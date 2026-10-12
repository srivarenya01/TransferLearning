"""
Enrich rice and soybean trial tables with more *shared* predictors.

Theory: transfer learning needs a thicker shared feature space (weather stress
indices + soil). Crop-specific GDD formulas and empty soil columns currently
leave only ~20 overlapping inputs; rice-only columns get random warm-start
weights. This script:

  1. Pulls SSURGO topsoil / site properties for unique lat/lon (cached).
  2. Pulls NASA POWER daily weather for unique (lat, lon, year) (cached).
  3. Aggregates planting→harvest with *identical* shared stress formulas
     (same rain / heat / dry-spell / shared-base GDD rules for both crops).
  4. Splits the season into GDD-fraction stages (S1–S4) and computes precip /
     GDD / stress / high-radiation *per stage* (timing, not only season totals).
  5. Adds harvest-day POWER snapshots so rice `_H` columns have soybean peers.
  6. Writes Data/VM_*_TESTING_STUFF_v2.csv (stage features). Also refreshes
     Data/VM_*_TESTING_STUFF.csv for backward compatibility.

Does not modify the original harvesting-only CSVs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "Data"
CACHE_DIR = DATA_DIR / "enrichment_cache"
SOIL_CACHE = CACHE_DIR / "ssurgo_by_latlon.json"
POWER_CACHE = CACHE_DIR / "nasapower_daily"
SOY_IN = DATA_DIR / "soybean_with_harvesting_only.csv"
RICE_IN = DATA_DIR / "rice_with_harvesting_only.csv"
SOY_OUT = DATA_DIR / "VM_SOYBEAN_TESTING_STUFF.csv"
RICE_OUT = DATA_DIR / "VM_RICE_TESTING_STUFF.csv"
SOY_OUT_V2 = DATA_DIR / "VM_SOYBEAN_TESTING_STUFF_v2.csv"
RICE_OUT_V2 = DATA_DIR / "VM_RICE_TESTING_STUFF_v2.csv"

SSURGO_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = (
    "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"
)

# Empty SoilGrids stubs in the soybean file — drop and replace with SSURGO.
DROP_SOY_EMPTY_SOIL = ("phh2o_0_15cm", "clay_0_15cm", "soc_0_15cm")

# NASA POWER ALLSKY_SFC_SW_DWN is MJ/m^2/day. ~20 is near the upper quartile
# in LA growing seasons in our cache (~22 MJ median summer) — "bright" days.
HIGH_RAD_MJ = 20.0
# Phenology bins as fractions of season cumulative Shared GDD (base 10 C).
STAGE_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0)
STAGE_NAMES = ("S1", "S2", "S3", "S4")  # early → late by GDD progress

REQUEST_SLEEP_S = 0.15
MAX_RETRIES = 4


def _round_coord(value: float, ndigits: int = 5) -> float:
    """Round GPS for cache keys without collapsing nearby stations."""
    return float(round(float(value), ndigits))


def _coord_key(lat: float, lon: float) -> str:
    return f"{_round_coord(lat)}_{_round_coord(lon)}"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# SSURGO
# ---------------------------------------------------------------------------

def ssurgo_query(lat: float, lon: float) -> str:
    """
    Weighted-mean topsoil (0–30 cm) and site properties at a point.

    Uses component.comppct_r weights. Categorical hydricrating is mapped to
    Yes=1 / Partial=0.5 / No=0 before averaging.
    """
    return f"""
SELECT
  SUM(ch.sandtotal_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.sandtotal_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_sand_pct,
  SUM(ch.silttotal_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.silttotal_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_silt_pct,
  SUM(ch.claytotal_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.claytotal_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_clay_pct,
  SUM(ch.om_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.om_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_om_pct,
  SUM(ch.cec7_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.cec7_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_cec,
  SUM(ch.ph1to1h2o_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.ph1to1h2o_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_ph,
  SUM(ch.awc_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.awc_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_awc,
  SUM(ch.dbthirdbar_r * c.comppct_r) / NULLIF(SUM(CASE WHEN ch.dbthirdbar_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_bd,
  SUM(c.slope_r * c.comppct_r) / NULLIF(SUM(CASE WHEN c.slope_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_slope_pct,
  SUM(c.elev_r * c.comppct_r) / NULLIF(SUM(CASE WHEN c.elev_r IS NOT NULL THEN c.comppct_r END), 0) AS soil_elev_m,
  SUM(
    CASE
      WHEN LOWER(c.hydricrating) LIKE 'yes%' THEN 1.0
      WHEN LOWER(c.hydricrating) LIKE 'partial%' THEN 0.5
      WHEN LOWER(c.hydricrating) LIKE 'no%' THEN 0.0
      ELSE NULL
    END * c.comppct_r
  ) / NULLIF(SUM(
    CASE
      WHEN c.hydricrating IS NULL OR c.hydricrating = '' THEN NULL
      ELSE c.comppct_r
    END
  ), 0) AS soil_hydric_score
FROM mapunit mu
INNER JOIN component c ON mu.mukey = c.mukey
LEFT JOIN chorizon ch ON c.cokey = ch.cokey
  AND ISNULL(ch.hzdept_r, 999) < 30
WHERE mu.mukey IN (
  SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT({lon} {lat})')
)
GROUP BY mu.mukey
"""


def fetch_ssurgo(lat: float, lon: float, session: requests.Session) -> Dict[str, float]:
    """Query Soil Data Access; return first mapunit row as floats."""
    payload = {"query": ssurgo_query(lat, lon), "format": "JSON+COLUMNNAME"}
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(SSURGO_URL, data=payload, timeout=90)
            response.raise_for_status()
            table = response.json().get("Table", [])
            if len(table) < 2:
                return {}
            headers, values = table[0], table[1]
            out: Dict[str, float] = {}
            for key, value in zip(headers, values):
                if value in (None, "", "NULL"):
                    out[key] = np.nan
                else:
                    try:
                        out[key] = float(value)
                    except (TypeError, ValueError):
                        out[key] = np.nan
            return out
        except Exception as exc:  # noqa: BLE001 — network; retry then empty
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"SSURGO failed ({lat}, {lon}): {last_error}")
    return {}


def collect_ssurgo(coords: Iterable[Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    """Fill soil cache for unique coordinates."""
    cache = load_json(SOIL_CACHE)
    session = requests.Session()
    todo = [(_round_coord(lat), _round_coord(lon)) for lat, lon in coords]
    todo = sorted(set(todo))
    for lat, lon in tqdm(todo, desc="SSURGO unique sites"):
        key = _coord_key(lat, lon)
        if key in cache and cache[key]:
            continue
        cache[key] = fetch_ssurgo(lat, lon, session)
        save_json(SOIL_CACHE, cache)
        time.sleep(REQUEST_SLEEP_S)
    return cache


# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------

def power_cache_path(lat: float, lon: float, year: int) -> Path:
    POWER_CACHE.mkdir(parents=True, exist_ok=True)
    return POWER_CACHE / f"{_coord_key(lat, lon)}_{int(year)}.csv"


def fetch_power_year(lat: float, lon: float, year: int, session: requests.Session) -> pd.DataFrame:
    """
    Daily NASA POWER AG community variables for one calendar year at a point.

    Cached on disk so re-runs do not re-hit the API.
    """
    path = power_cache_path(lat, lon, year)
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    params = {
        "parameters": POWER_PARAMS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(POWER_URL, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            properties = payload["properties"]["parameter"]
            dates = sorted(properties["T2M"].keys())
            frame = pd.DataFrame({"date": pd.to_datetime(dates, format="%Y%m%d")})
            for name, series in properties.items():
                frame[name] = [series[d] for d in dates]
            # POWER uses -999 for missing
            frame = frame.replace(-999, np.nan)
            frame.to_csv(path, index=False)
            time.sleep(REQUEST_SLEEP_S)
            return frame
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"NASA POWER failed ({lat}, {lon}, {year}): {last_error}")


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest True streak in a boolean array."""
    best = cur = 0
    for flag in mask:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _stage_masks(gdd: np.ndarray) -> List[np.ndarray]:
    """
    Boolean masks for GDD-fraction stages S1–S4.

    Stages are quartiles of *cumulative* shared GDD within the season so early
    vs late stress lines up by thermal progress, not calendar date. If total
    GDD is zero, falls back to equal day-count quarters.
    """
    n = len(gdd)
    if n == 0:
        return [np.array([], dtype=bool) for _ in STAGE_NAMES]
    cum = np.cumsum(np.nan_to_num(gdd, nan=0.0))
    total = float(cum[-1])
    masks: List[np.ndarray] = []
    if total <= 1e-6:
        idx = np.arange(n)
        for i in range(len(STAGE_NAMES)):
            lo = int(round(STAGE_EDGES[i] * n))
            hi = int(round(STAGE_EDGES[i + 1] * n))
            masks.append((idx >= lo) & (idx < hi) if i < len(STAGE_NAMES) - 1 else (idx >= lo) & (idx <= hi))
        return masks
    for i in range(len(STAGE_NAMES)):
        lo = STAGE_EDGES[i] * total
        hi = STAGE_EDGES[i + 1] * total
        if i == 0:
            masks.append(cum <= hi)
        elif i == len(STAGE_NAMES) - 1:
            masks.append(cum > lo)
        else:
            masks.append((cum > lo) & (cum <= hi))
    return masks


def _stage_block(
    prefix: str,
    mask: np.ndarray,
    precip: np.ndarray,
    tmax: np.ndarray,
    tmin: np.ndarray,
    gdd: np.ndarray,
    solar: np.ndarray,
) -> Dict[str, float]:
    """Shared stress aggregates for one phenology stage (or empty if no days)."""
    if not np.any(mask):
        return {
            f"{prefix}_Days": 0.0,
            f"{prefix}_Cum_Precip": 0.0,
            f"{prefix}_GDD_Base10": 0.0,
            f"{prefix}_Flooding_Days_15mm": 0.0,
            f"{prefix}_Heat_Days_Tmax35C": 0.0,
            f"{prefix}_Dry_Days": 0.0,
            f"{prefix}_HighRad_Days": 0.0,
            f"{prefix}_HighRad_Sum": 0.0,
            f"{prefix}_Mean_Solar": np.nan,
        }
    p, tx, tn, g, s = precip[mask], tmax[mask], tmin[mask], gdd[mask], solar[mask]
    high = s > HIGH_RAD_MJ
    return {
        f"{prefix}_Days": float(np.sum(mask)),
        f"{prefix}_Cum_Precip": float(np.nansum(p)),
        f"{prefix}_GDD_Base10": float(np.nansum(g)),
        f"{prefix}_Flooding_Days_15mm": float(np.nansum(p > 15)),
        f"{prefix}_Heat_Days_Tmax35C": float(np.nansum(tx > 35)),
        f"{prefix}_Dry_Days": float(np.nansum(p < 1.0)),
        f"{prefix}_HighRad_Days": float(np.nansum(high)),
        f"{prefix}_HighRad_Sum": float(np.nansum(np.where(high, s, 0.0))),
        f"{prefix}_Mean_Solar": float(np.nanmean(s)),
    }


def aggregate_season(daily: pd.DataFrame, plant: pd.Timestamp, harvest: pd.Timestamp) -> Dict[str, float]:
    """
    Shared seasonal + stage-wise indices from daily POWER rows (plant→harvest).

    Uses the same thresholds for rice and soybean so columns are transferable.
    Shared GDD base is 10 C (not crop-specific caps). Stages S1–S4 are
    quartiles of cumulative shared GDD. High-radiation days use HIGH_RAD_MJ.
    """
    window = daily[(daily["date"] >= plant) & (daily["date"] <= harvest)].copy()
    empty: Dict[str, float] = {
        "Shared_Season_Days": np.nan,
        "Shared_Cum_Precip": np.nan,
        "Shared_Flooding_Days_15mm": np.nan,
        "Shared_Heavy_Rain_Days_25mm": np.nan,
        "Shared_Extreme_Rain_Days_50mm": np.nan,
        "Shared_Dry_Days": np.nan,
        "Shared_Max_Consecutive_Dry_Days": np.nan,
        "Shared_Max_Consecutive_Wet_Days": np.nan,
        "Shared_Mean_Rain_Intensity": np.nan,
        "Shared_Heat_Days_Tmax35C": np.nan,
        "Shared_Heat_Days_Tmax38C": np.nan,
        "Shared_Cold_Days_Tmin10C": np.nan,
        "Shared_Frost_Days_Tmin0C": np.nan,
        "Shared_Mean_Diurnal_Range": np.nan,
        "Shared_GDD_Base10_Cumulative": np.nan,
        "Shared_GDD_Base10_Avg": np.nan,
        "Shared_Mean_Tavg": np.nan,
        "Shared_Mean_Tmax": np.nan,
        "Shared_Mean_Tmin": np.nan,
        "Shared_Mean_RH": np.nan,
        "Shared_Mean_Wind": np.nan,
        "Shared_Cum_Solar": np.nan,
        "Shared_Mean_Solar": np.nan,
        "Shared_Mean_VPD_proxy": np.nan,
        "Shared_HighRad_Days": np.nan,
        "Shared_HighRad_Sum": np.nan,
        "Shared_HighRad_Frac": np.nan,
        "Temperature_H": np.nan,
        "WindSpeed_H": np.nan,
        "SolarRadiation_H": np.nan,
        "RelativeHumidity_H": np.nan,
        "Precipitation_H": np.nan,
    }
    for name in STAGE_NAMES:
        for suffix in (
            "Days", "Cum_Precip", "GDD_Base10", "Flooding_Days_15mm",
            "Heat_Days_Tmax35C", "Dry_Days", "HighRad_Days", "HighRad_Sum", "Mean_Solar",
        ):
            empty[f"Shared_{name}_{suffix}"] = np.nan

    if window.empty:
        return empty

    precip = window["PRECTOTCORR"].to_numpy(dtype=float)
    tmax = window["T2M_MAX"].to_numpy(dtype=float)
    tmin = window["T2M_MIN"].to_numpy(dtype=float)
    tavg = window["T2M"].to_numpy(dtype=float)
    rh = window["RH2M"].to_numpy(dtype=float)
    wind = window["WS2M"].to_numpy(dtype=float)
    solar = window["ALLSKY_SFC_SW_DWN"].to_numpy(dtype=float)

    wet = precip > 1.0
    dry = precip < 1.0
    gdd = np.maximum(0.0, (tmax + tmin) / 2.0 - 10.0)
    high_rad = solar > HIGH_RAD_MJ
    # Simple vapor-pressure-deficit proxy (kPa-ish scale, not exact ASCE).
    es = 0.6108 * np.exp(17.27 * tavg / (tavg + 237.3))
    vpd = es * (1.0 - np.clip(rh, 0, 100) / 100.0)

    out: Dict[str, float] = {
        "Shared_Season_Days": float(len(window)),
        "Shared_Cum_Precip": float(np.nansum(precip)),
        "Shared_Flooding_Days_15mm": float(np.nansum(precip > 15)),
        "Shared_Heavy_Rain_Days_25mm": float(np.nansum(precip > 25)),
        "Shared_Extreme_Rain_Days_50mm": float(np.nansum(precip > 50)),
        "Shared_Dry_Days": float(np.nansum(dry)),
        "Shared_Max_Consecutive_Dry_Days": float(_longest_run(np.nan_to_num(dry, nan=False).astype(bool))),
        "Shared_Max_Consecutive_Wet_Days": float(_longest_run(np.nan_to_num(wet, nan=False).astype(bool))),
        "Shared_Mean_Rain_Intensity": float(np.nansum(precip) / max(np.nansum(wet), 1)),
        "Shared_Heat_Days_Tmax35C": float(np.nansum(tmax > 35)),
        "Shared_Heat_Days_Tmax38C": float(np.nansum(tmax > 38)),
        "Shared_Cold_Days_Tmin10C": float(np.nansum(tmin < 10)),
        "Shared_Frost_Days_Tmin0C": float(np.nansum(tmin < 0)),
        "Shared_Mean_Diurnal_Range": float(np.nanmean(tmax - tmin)),
        "Shared_GDD_Base10_Cumulative": float(np.nansum(gdd)),
        "Shared_GDD_Base10_Avg": float(np.nanmean(gdd)),
        "Shared_Mean_Tavg": float(np.nanmean(tavg)),
        "Shared_Mean_Tmax": float(np.nanmean(tmax)),
        "Shared_Mean_Tmin": float(np.nanmean(tmin)),
        "Shared_Mean_RH": float(np.nanmean(rh)),
        "Shared_Mean_Wind": float(np.nanmean(wind)),
        "Shared_Cum_Solar": float(np.nansum(solar)),
        "Shared_Mean_Solar": float(np.nanmean(solar)),
        "Shared_Mean_VPD_proxy": float(np.nanmean(vpd)),
        "Shared_HighRad_Days": float(np.nansum(high_rad)),
        "Shared_HighRad_Sum": float(np.nansum(np.where(high_rad, solar, 0.0))),
        "Shared_HighRad_Frac": float(np.nansum(high_rad) / max(len(window), 1)),
    }

    for name, mask in zip(STAGE_NAMES, _stage_masks(gdd)):
        out.update(_stage_block(f"Shared_{name}", mask, precip, tmax, tmin, gdd, solar))

    # Harvest-day snapshot (soy lacked rice's Temperature_H etc.)
    harvest_rows = daily[daily["date"] == harvest]
    if harvest_rows.empty:
        harvest_rows = window.tail(1)
    hr = harvest_rows.iloc[0]
    out.update(
        {
            "Temperature_H": float(hr["T2M"]),
            "WindSpeed_H": float(hr["WS2M"]),
            "SolarRadiation_H": float(hr["ALLSKY_SFC_SW_DWN"]),
            "RelativeHumidity_H": float(hr["RH2M"]),
            "Precipitation_H": float(hr["PRECTOTCORR"]),
        }
    )
    return out


def enrich_crop(df: pd.DataFrame, soil_cache: Dict[str, Dict[str, float]], label: str) -> pd.DataFrame:
    """Attach SSURGO + shared seasonal POWER aggregates to one crop table."""
    out = df.copy()
    for col in DROP_SOY_EMPTY_SOIL:
        if col in out.columns:
            out = out.drop(columns=[col])

    out["Planting_Date"] = pd.to_datetime(out["Planting_Date"], errors="coerce")
    out["Harvesting_Date"] = pd.to_datetime(out["Harvesting_Date"], errors="coerce")
    if "Emergence_Date" in out.columns:
        out["Emergence_Date"] = pd.to_datetime(out["Emergence_Date"], errors="coerce")
        out["days_from_year_start_Emergence"] = out["Emergence_Date"].dt.dayofyear.astype(float)
    elif "days_from_year_start_Emergence" not in out.columns:
        out["days_from_year_start_Emergence"] = np.nan

    # Soil join by rounded lat/lon
    soil_cols = [
        "soil_sand_pct", "soil_silt_pct", "soil_clay_pct", "soil_om_pct",
        "soil_cec", "soil_ph", "soil_awc", "soil_bd", "soil_slope_pct",
        "soil_elev_m", "soil_hydric_score",
    ]
    for col in soil_cols:
        out[col] = np.nan
    for idx, row in out.iterrows():
        key = _coord_key(row["lat"], row["long"])
        props = soil_cache.get(key, {})
        for col in soil_cols:
            if col in props:
                out.at[idx, col] = props[col]

    # Unique season windows for POWER aggregation
    season_keys = (
        out[["lat", "long", "Planting_Date", "Harvesting_Date"]]
        .dropna()
        .drop_duplicates()
    )
    session = requests.Session()
    season_lookup: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for _, season in tqdm(season_keys.iterrows(), total=len(season_keys), desc=f"POWER seasons ({label})"):
        lat = float(season["lat"])
        lon = float(season["long"])
        plant = pd.Timestamp(season["Planting_Date"])
        harvest = pd.Timestamp(season["Harvesting_Date"])
        key = (_round_coord(lat), _round_coord(lon), plant.date().isoformat(), harvest.date().isoformat())
        if key in season_lookup:
            continue
        years = range(plant.year, harvest.year + 1)
        frames = [fetch_power_year(lat, lon, year, session) for year in years]
        daily = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
        season_lookup[key] = aggregate_season(daily, plant, harvest)

    shared_cols = list(next(iter(season_lookup.values())).keys()) if season_lookup else []
    for col in shared_cols:
        if col == "days_from_year_start_Emergence":
            continue
        # Rice already has harvest-day weather; only fill gaps / add Shared_* .
        if col.endswith("_H") and col in out.columns:
            continue
        out[col] = np.nan

    for idx, row in out.iterrows():
        if pd.isna(row["Planting_Date"]) or pd.isna(row["Harvesting_Date"]):
            continue
        key = (
            _round_coord(row["lat"]),
            _round_coord(row["long"]),
            pd.Timestamp(row["Planting_Date"]).date().isoformat(),
            pd.Timestamp(row["Harvesting_Date"]).date().isoformat(),
        )
        props = season_lookup.get(key, {})
        for col, value in props.items():
            if col == "days_from_year_start_Emergence":
                continue
            if col.endswith("_H") and col in df.columns and col in out.columns:
                # Preserve original rice harvest-day values when present.
                if pd.notna(out.at[idx, col]):
                    continue
            out.at[idx, col] = value

    # Dates back to string for CSV compatibility with the main pipeline.
    for date_col in ("Planting_Date", "Harvesting_Date", "Emergence_Date"):
        if date_col in out.columns:
            out[date_col] = out[date_col].dt.strftime("%Y-%m-%d")

    return out


def feature_report(soy: pd.DataFrame, rice: pd.DataFrame) -> None:
    """Print how many numeric columns now overlap after enrichment."""
    ignore = {
        "Sl", "Yield", "Year", "GermplasmName_or_Variety", "Maturity_Group",
        "Station_or_Location", "Location", "Variety", "State", "Spacing", "GPS",
        "Planting_Date", "Harvesting_Date", "Emergence_Date",
        "Harvest.moisture.LSU_01.0000136", "Harvest.weight.LSU_01.0000137",
        "Lodging.LSU_01.0000135", "Plant.height.LSU_01.0000132",
        "Plot.stand.near.flood.LSU_01.0000133", "Seedling.vigor.LSU_01.0000131",
        "latitude", "longitude",
    }
    soy_num = {
        c for c in soy.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(soy[c])
    }
    rice_num = {
        c for c in rice.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(rice[c])
    }
    common = sorted(soy_num & rice_num)
    print(f"\nNumeric common columns: {len(common)}")
    print(common)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    soy = pd.read_csv(SOY_IN)
    rice = pd.read_csv(RICE_IN)

    coords = list(zip(soy["lat"], soy["long"])) + list(zip(rice["lat"], rice["long"]))
    soil_cache = collect_ssurgo(coords)

    soy_out = enrich_crop(soy, soil_cache, "soybean")
    rice_out = enrich_crop(rice, soil_cache, "rice")

    soy_out.to_csv(SOY_OUT, index=False)
    rice_out.to_csv(RICE_OUT, index=False)
    soy_out.to_csv(SOY_OUT_V2, index=False)
    rice_out.to_csv(RICE_OUT_V2, index=False)
    print(f"Saved {SOY_OUT} and {SOY_OUT_V2}")
    print(f"Saved {RICE_OUT} and {RICE_OUT_V2}")
    print(f"High-radiation threshold: {HIGH_RAD_MJ} MJ/m^2/day | stages: {STAGE_NAMES} by GDD fraction")
    feature_report(soy_out, rice_out)


if __name__ == "__main__":
    main()
