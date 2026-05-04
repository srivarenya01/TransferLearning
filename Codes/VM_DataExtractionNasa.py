import os
import shutil
import pandas as pd
import numpy as np
import xarray as xr
import fsspec
import s3fs
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
PARENT_DIR: Path = Path(__file__).parent
DATA_DIR: Path = PARENT_DIR / "Data"

# Set these to the file you want to process

INPUT_FILE: Path = DATA_DIR / "Processed/Rice/Auto-Rice_long02_v1.csv" 
OUTPUT_FILE: Path = DATA_DIR / "Weather_Data_Rice_full.xlsx"

TEMP_DIR: Path = DATA_DIR / "temp_weather_v3" # Directory for temporary weather files
NUM_ROWS: int = -1  # Set to -1 for all rows, or a positive integer for a subset
# =============================================================================

class NasaWeatherV3:
    """
    Handles optimized weather data extraction from NASA POWER's S3 Zarr Datastore.
    - We are using S3 Datastore because of the ratelimiting on API and faster data retrieval.
    """

    def __init__(self) -> None:
        """
        Initializes S3 store paths and parameter mappings.
        """
        self.met_store: str = "s3://nasa-power/merra2/temporal/power_merra2_daily_temporal_utc.zarr"
        self.sol_store: str = "s3://nasa-power/syn1deg/temporal/power_syn1deg_daily_temporal_utc.zarr"
        
        self.param_mapping: Dict[str, str] = {
            'T2M': 'Temperature',            
            'T2M_MAX': 'Temperature Max',
            'T2M_MIN': 'Temperature Min',
            'RH2M': 'Relative Humidity',    
            'WS2M': 'Wind Speed',           
            'PRECTOTCORR': 'Precipitation', 
            'ALLSKY_SFC_SW_DWN': 'Solar Radiation' 
        }
        
        self.ds_met: Optional[xr.Dataset] = None
        self.ds_sol: Optional[xr.Dataset] = None
        self.cache: Dict[Tuple[float, float], Tuple[xr.Dataset, xr.Dataset]] = {}

    def connect(self) -> None:
        """
        Connects to NASA S3 Zarr datastores and ensures coordinate monotonicity.
        """
        print("Connecting to NASA S3 Zarr Datastore...")
        try:
            self.ds_met = xr.open_zarr(self.met_store, storage_options={"anon": True}, consolidated=True)
            self.ds_sol = xr.open_zarr(self.sol_store, storage_options={"anon": True}, consolidated=True)
            
            # Ensure unique and monotonic coordinates for nearest-neighbor selection
            self.ds_met = self.ds_met.drop_duplicates('lat').drop_duplicates('lon').sortby(['lat', 'lon'])
            self.ds_sol = self.ds_sol.drop_duplicates('lat').drop_duplicates('lon').sortby(['lat', 'lon'])
            
            met_vars = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PRECTOTCORR']
            sol_vars = ['ALLSKY_SFC_SW_DWN']
            
            self.ds_met = self.ds_met[[v for v in met_vars if v in self.ds_met.data_vars]]
            self.ds_sol = self.ds_sol[[v for v in sol_vars if v in self.ds_sol.data_vars]]
            print("Connected successfully.")
        except Exception as e:
            print(f"Error connecting to NASA S3: {e}")
            raise

    def cache_locations(self, unique_coords: pd.DataFrame, start_year: str = "2014", end_year: str = "2025") -> None:
        """
        Fetches and caches full time-series data for unique locations to minimize S3 I/O.

        Args:
            unique_coords: DataFrame containing unique 'lat' and 'lon' pairs.
            start_year: Start year for the historical window.
            end_year: End year for the historical window.
        """
        print(f"Pre-caching time-series ({start_year}-{end_year}) for {len(unique_coords)} locations...")
        time_slice = slice(f"{start_year}-01-01", f"{end_year}-12-31")
        
        def _fetch_coord(lat: float, lon: float) -> Tuple[Tuple[float, float], Optional[Tuple[xr.Dataset, xr.Dataset]]]:
            try:
                m = self.ds_met.sel(lat=lat, lon=lon, method='nearest').sel(time=time_slice).load()
                s = self.ds_sol.sel(lat=lat, lon=lon, method='nearest').sel(time=time_slice).load()
                return (lat, lon), (m, s)
            except Exception as e:
                print(f"Error caching ({lat}, {lon}): {e}")
                return (lat, lon), None

        with ThreadPoolExecutor(max_workers=min(10, len(unique_coords))) as executor:
            futures = [executor.submit(_fetch_coord, row['lat'], row['lon']) for _, row in unique_coords.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching Locations"):
                coord, datasets = future.result()
                if datasets:
                    self.cache[coord] = datasets

    def get_weather_for_row(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Extracts weather data for a specific location and period from the local cache.

        Args:
            lat: Latitude coordinate.
            lon: Longitude coordinate.
            start_date: Start of the daily weather period.
            end_date: End of the daily weather period.

        Returns:
            A DataFrame with weather parameters or None if retrieval fails.
        """
        coord = (lat, lon)
        if coord not in self.cache:
            return None
            
        m_ds, s_ds = self.cache[coord]
        try:
            m_df = m_ds.sel(time=slice(start_date, end_date)).to_dataframe().drop(columns=['lat', 'lon'], errors='ignore')
            s_df = s_ds.sel(time=slice(start_date, end_date)).to_dataframe().drop(columns=['lat', 'lon'], errors='ignore')
        except:
            return None
            
        df = pd.concat([m_df, s_df], axis=1)
        if df.empty:
            return None
            
        if 'ALLSKY_SFC_SW_DWN' in df.columns:
            df['ALLSKY_SFC_SW_DWN'] = (df['ALLSKY_SFC_SW_DWN'] * 0.0864).round(2)
            
        df = df.reset_index().set_index('time')
        df = df.rename(columns=self.param_mapping)
        
        for col in ['Temperature', 'Temperature Max', 'Temperature Min', 'Relative Humidity', 'Wind Speed', 'Precipitation']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        cols = [c for c in self.param_mapping.values() if c in df.columns]
        return df[cols]

def parse_gps(gps_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses a combined GPS string into latitude and longitude floats.

    Args:
        gps_str: A string in the format "lat, lon".

    Returns:
        A tuple of (latitude, longitude) or (None, None) if parsing fails.
    """
    try:
        s = str(gps_str).strip('"').strip()
        lat, lon = map(float, s.split(','))
        return lat, lon
    except:
        return None, None

def doy_to_date(year: int, doy: int) -> Optional[datetime]:
    """
    Converts Year and Day of Year to a datetime object.

    Args:
        year: Integer year.
        doy: Day of year (1-366).

    Returns:
        A datetime object or None if invalid.
    """
    try:
        return datetime(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    except:
        return None

def prepare_dataframe(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """
    Standardizes input columns and preserves original index for alignment.
    """
    df = df.copy()
    
    # STRICT ALIGNMENT: Use 'Sl' column
    if 'Sl' not in df.columns:
        raise ValueError("Dataset MUST have an 'Sl' column for alignment.")
    
    # Ensure Sl is integer
    df['Sl'] = df['Sl'].astype(int)
    
    if 'GPS' in df.columns:
        df['lat'], df['lon'] = zip(*df['GPS'].apply(parse_gps))
    else:
        raise ValueError("Dataset missing 'GPS' column.")

    if 'Planting_Date' in df.columns and 'Harvesting_Date' in df.columns:
        df['Planting_DT'] = pd.to_datetime(df['Planting_Date'])
        df['Harvesting_DT'] = pd.to_datetime(df['Harvesting_Date'])
    elif 'days_from_year_start_Planting' in df.columns and 'Year' in df.columns:
        df['Planting_DT'] = df.apply(lambda x: doy_to_date(x['Year'], x['days_from_year_start_Planting']), axis=1)
        df['Harvesting_DT'] = df.apply(lambda x: doy_to_date(x['Year'], x['days_from_year_start_harvest']), axis=1)
    else:
        raise ValueError("Could not resolve Planting/Harvesting dates from columns.")

    is_rice = "rice" in str(file_path).lower() or 'Rice' in str(file_path)
    offset = 10 if is_rice else 5
    df['Emergence_DT'] = pd.NaT
    
    if 'days_from_year_start_Emergence' in df.columns and 'Year' in df.columns:
        valid_mask = df['days_from_year_start_Emergence'].notna()
        if valid_mask.any():
            df.loc[valid_mask, 'Emergence_DT'] = df[valid_mask].apply(
                lambda x: doy_to_date(x['Year'], x['days_from_year_start_Emergence']), axis=1
            )
    
    if 'Emergence_Date' in df.columns:
        df['Emergence_DT'] = df['Emergence_DT'].fillna(pd.to_datetime(df['Emergence_Date']))
        
    df['Emergence_DT'] = df['Emergence_DT'].fillna(df['Planting_DT'] + timedelta(days=offset))
    df['Emergence_Offset_Days'] = (df['Emergence_DT'] - df['Planting_DT']).dt.days

    return df.dropna(subset=['lat', 'lon', 'Planting_DT', 'Harvesting_DT'])

def process_and_save_temp(args: Tuple[int, pd.Series, NasaWeatherV3]) -> bool:
    """
    Processes a single dataset row and saves to temporary parquet using Original_Index.
    """
    idx, row, nasa = args
    # Use Sl for unique identification
    sl_no = row['Sl']
    
    df_w = nasa.get_weather_for_row(row['lat'], row['lon'], row['Planting_DT'], row['Harvesting_DT'])
    if df_w is not None:
        date_label = row['Planting_DT'].strftime('%Y-%m-%d')
        # PREFER Sl-based naming
        sheet_name = f"Sl_{sl_no}_{date_label}"
        safe_name = "".join([c for c in sheet_name if c.isalnum() or c in (' ', '_', '-')])[:31]
        temp_file = TEMP_DIR / f"{idx:05d}_{safe_name}.parquet"
        df_w.to_parquet(temp_file)
        return True
    return False

if __name__ == "__main__":
    print(f"Loading input from: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        exit(1)

    if INPUT_FILE.suffix.lower() in ['.xlsx', '.xls']:
        df_raw = pd.read_excel(INPUT_FILE)
    else:
        df_raw = pd.read_csv(INPUT_FILE)
        
    df_input = prepare_dataframe(df_raw, INPUT_FILE)
    
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    unique_coords = df_input[['lat', 'lon']].drop_duplicates()
    
    nasa = NasaWeatherV3()
    nasa.connect()
    nasa.cache_locations(unique_coords)

    df_proc = df_input.head(NUM_ROWS).copy() if NUM_ROWS > 0 else df_input.copy()

    num_workers = max(1, int((os.cpu_count() or 1) * 0.8))
    print(f"Starting parallel processing ({num_workers} threads) for {len(df_proc)} rows...")

    tasks = [(idx, row, nasa) for idx, row in df_proc.iterrows()]
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_and_save_temp, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Rows"):
            results.append(future.result())

    print(f"Successfully processed {sum(results)} / {len(results)} rows.")
    
    temp_files = sorted(list(TEMP_DIR.glob("*.parquet")))
    if not temp_files:
        print("No weather data extracted.")
        exit(0)

    print(f"Merging results into {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_input.to_excel(writer, sheet_name="Summary_Input", index=False)
        
        for f in tqdm(temp_files, desc="Writing Excel Sheets"):
            sheet_name = f.stem[6:] 
            df = pd.read_parquet(f)
            df.to_excel(writer, sheet_name=sheet_name)
    
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    print(f"DONE! Output saved to: {OUTPUT_FILE}")
