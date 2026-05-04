import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
PARENT_DIR: Path = Path(__file__).parent
DATA_DIR: Path = PARENT_DIR / "Data"

# Process both datasets
CONFIGS: List[Dict[str, Union[str, Path]]] = [
    {
        'crop': 'Soybean',
        'weather_file': DATA_DIR / "Weather_Data_soy_full.xlsx",
        'output_file': DATA_DIR / "Weather_Aggregated_Soybean.csv"
    },
    {
        'crop': 'Rice',
        'weather_file': DATA_DIR / "Weather_Data_Rice_full.xlsx",
        'output_file': DATA_DIR / "Weather_Aggregated_Rice.csv"
    }
]
# =============================================================================

def c_to_f(c: float) -> float:
    """
    Converts Celsius to Fahrenheit.

    Args:
        c: Temperature in Celsius.

    Returns:
        Temperature in Fahrenheit.
    """
    return (c * 1.8) + 32

def calculate_gdd(t_max_c: float, t_min_c: float, crop_type: str) -> float:
    """
    Calculates Growing Degree Days (GDD) based on crop-specific scientific formulas.

    Args:
        t_max_c: Daily maximum temperature in Celsius.
        t_min_c: Daily minimum temperature in Celsius.
        crop_type: Type of crop ('Rice' or others assumed as 'Soybean').

    Returns:
        Calculated GDD value for the day.
    """
    t_max_f = c_to_f(t_max_c)
    t_min_f = c_to_f(t_min_c)
    t_avg_f = (t_max_f + t_min_f) / 2
    
    if crop_type.lower() == 'rice':
        # Rice Formula: (max(max_temp, 107.6) + min(min_temp, 46.4))/2 if avg > 50
        if t_avg_f > 50:
            return (max(t_max_f, 107.6) + min(t_min_f, 46.4)) / 2
        return 0.0
    else:
        # Soybean Formula: (max(max_temp, 86) + min(min_temp, 50))/2 if avg > 50
        if t_avg_f > 50:
            return (max(t_max_f, 86) + min(t_min_f, 50)) / 2
        return 0.0

class WeatherAggregatorUnified:
    """
    Aggregates daily NASA weather data into seasonal cumulative metrics for crop modeling.
    """

    def __init__(self, crop_config: Dict[str, Any]) -> None:
        """
        Initializes the aggregator with a specific crop configuration.

        Args:
            crop_config: Dictionary containing 'crop', 'weather_file', and 'output_file'.
        """
        self.crop: str = crop_config['crop']
        self.weather_file: Path = crop_config['weather_file']
        self.output_file: Path = crop_config['output_file']

    def run(self) -> None:
        """
        Executes the aggregation pipeline: loads weather data, calculates metrics, and saves CSV.
        """
        print(f"\nProcessing {self.crop} from {self.weather_file}...")
        
        try:
            df_summary = pd.read_excel(self.weather_file, sheet_name='Summary_Input', engine='openpyxl')
        except Exception as e:
            print(f"Error loading Summary_Input for {self.crop}: {e}")
            return

        mapping = {
            'Planting_Date': 'Planting_DT',
            'Harvesting_Date': 'Harvesting_DT',
            'Emergence_Date': 'Emergence_DT'
        }
        for orig, new in mapping.items():
            if new not in df_summary.columns and orig in df_summary.columns:
                df_summary[new] = pd.to_datetime(df_summary[orig])

        results: List[Dict[str, Any]] = []
        
        print(f"Loading daily weather sheets...")
        all_sheets = pd.read_excel(self.weather_file, sheet_name=None, engine='openpyxl')
        
        for idx, row in tqdm(df_summary.iterrows(), total=len(df_summary), desc=f"Aggregating {self.crop}"):
            res_row = {
                'Sl': row.get('Sl', idx),
                'GPS': row.get('GPS', ''),
                'lat': row.get('lat', np.nan),
                'long': row.get('lon', row.get('long', np.nan)),
                'Planting_Date': str(row.get('Planting_DT', ''))[:10],
                'Harvesting_Date': str(row.get('Harvesting_DT', ''))[:10],
                'Emergence_Date': str(row.get('Emergence_DT', ''))[:10]
            }
            
            # Construct sheet name (logic must match extraction script)
            date_label = pd.to_datetime(row['Planting_DT']).strftime('%Y-%m-%d')
            
            # Use preserved Original_Index from Summary_Input if available, else loop idx
            ref_idx = row.get('Original_Index', idx)
            sheet_name = f"Row_{ref_idx}_{date_label}"
            safe_name = "".join([c for c in sheet_name if c.isalnum() or c in (' ', '_', '-')])[:31]
            
            if safe_name in all_sheets:
                df_daily = all_sheets[safe_name]
                if df_daily.empty:
                    results.append(self._fill_nan(res_row))
                    continue
                
                df_daily['time'] = pd.to_datetime(df_daily['time'])
                emergence_dt = pd.to_datetime(row['Emergence_DT'])
                
                # --- Planting to Harvest Calculations ---
                res_row['Cumulative_Precipitation'] = round(df_daily['Precipitation'].sum(), 2)
                res_row['Flooding_Days'] = int((df_daily['Precipitation'] > 15).sum())
                res_row['Avg_Humidity'] = round(df_daily['Relative Humidity'].mean(), 2)
                res_row['Heavy_Wind_Occurrences'] = int((df_daily['Wind Speed'] > 2).sum())
                
                gdds = df_daily.apply(lambda r: calculate_gdd(r['Temperature Max'], r['Temperature Min'], self.crop), axis=1)
                res_row['GDD_Cumulative'] = round(gdds.sum(), 2)
                res_row['GDD_Max'] = round(gdds.max(), 2)
                res_row['GDD_Min'] = round(gdds.min(), 2)
                res_row['GDD_Avg'] = round(gdds.mean(), 2)
                
                # 6. Heavy Heats (Tmax > 80°F)
                res_row['Heavy_Heats'] = int(((df_daily['Temperature Max'] * 1.8) + 32 > 80).sum())
                
                # --- Emergence to Harvest Calculations ---
                df_emergence = df_daily[df_daily['time'] >= emergence_dt]
                if not df_emergence.empty:
                    res_row['Solar_Cumulative'] = round(df_emergence['Solar Radiation'].sum(), 2)
                    res_row['Solar_Max'] = round(df_emergence['Solar Radiation'].max(), 2)
                    res_row['Solar_Avg'] = round(df_emergence['Solar Radiation'].mean(), 2)
                else:
                    res_row['Solar_Cumulative'] = np.nan
                    res_row['Solar_Max'] = np.nan
                    res_row['Solar_Avg'] = np.nan
                
                results.append(res_row)
            else:
                results.append(self._fill_nan(res_row))

        df_out = pd.DataFrame(results)
        df_out.to_csv(self.output_file, index=False)
        print(f"Saved: {self.output_file}")

    def _fill_nan(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fills a result row with NaNs when weather data is missing.

        Args:
            row: The metadata dictionary for the row.

        Returns:
            The dictionary updated with empty weather metrics.
        """
        cols = ['Cumulative_Precipitation', 'Flooding_Days', 'Avg_Humidity', 
                'Heavy_Wind_Occurrences', 'GDD_Cumulative', 'GDD_Max', 'GDD_Min', 'GDD_Avg',
                'Heavy_Heats', 'Solar_Cumulative', 'Solar_Max', 'Solar_Avg']
        for c in cols:
            row[c] = np.nan
        return row

if __name__ == "__main__":
    for config in CONFIGS:
        agg = WeatherAggregatorUnified(config)
        agg.run()
