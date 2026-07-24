"""
Gully parameterisation for Dynamic SedNet.

Python port of the linear-growth path through
``Dynamic_SedNet/Parameterisation/Models/GullyParameterisationModel.cs`` and
``Dynamic_SedNet/Tools/ToolsModel.cs::createGullyTimeSeriesInclCalcVolume``
(plus its helper ``getEventRunoffFromFilesMM``).

For each (subcatchment, FU) combination the workflow is:

1. Resolve five spatial inputs (gully density, soil bulk density, B-horizon
   clay %, year of disturbance, gully cross-sectional area) to a per-FU value.
   Each input may be supplied as a raster or as a uniform scalar fallback.
2. Compute total gully volume, integration window, and long-term annual
   sediment supply.
3. Load FU-level event runoff (daily runoff minus daily baseflow, floored at
   zero), aggregate to annual, and produce an annual gully load timeseries
   modulated by yearly runoff anomalies and the linear growth curve.

Only the linear growth model is implemented. The exponential / sigmoidal /
Gompertz branches in the C# source carry hardcoded ``b1=b2=1`` and a buggy
local override of ``b1`` (``-ln(1-year)/year``) and were not used in
production -- they are intentionally omitted here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping, Optional, Union
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename conventions (mirroring the C# StringConstants used by the plugin)
# ---------------------------------------------------------------------------

ELEMENT_SEPARATOR = "$"
RUNOFF_VAR = "Runoff"
BASEFLOW_VAR = "Baseflow"
MM_PER_DAY_SUFFIX = "_mmPerDay"
GULLY_ANNUAL_LOAD = "AnnualLoad_KG"
GULLY_ANNUAL_RUNOFF = "AnnualRunoff_MM"
PARAM_STATS_FILENAME = "GullyParamStats.csv"

# Raster / parameter names (preserved from the C# Stats_Name constants so the
# stats DataFrame column ordering is unambiguous and self-documenting).
GULLY_DENSITY = "Gully_Density"
SOIL_BD = "Soil_BD_Raster_kg_per_m3"
B_HORIZON_CLAY_PCT = "B_Horizon_Clay_Percent"
YEAR_OF_DISTURBANCE = "Year_Of_Disturbance"
GULLY_CSA = "Gully_CSA"

RASTER_PARAMS = (
    GULLY_DENSITY,
    SOIL_BD,
    B_HORIZON_CLAY_PCT,
    YEAR_OF_DISTURBANCE,
    GULLY_CSA,
)

# Year-of-disturbance is summarised by majority; everything else by mean.
MAJORITY_PARAMS = frozenset({YEAR_OF_DISTURBANCE})


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class GullyParameters:
    """Parameters and derived statistics for a single (catchment, FU) gully model.

    Field names match the C# ``SedNet_Gully_Model`` properties so that the
    resulting parameter table can be consumed unchanged by the OpenWater
    migration / parameteriser layer.
    """
    # Spatial / per-FU inputs
    Gully_Density: float = 0.0                     # km / km^2
    Gully_Soil_Bulk_Density: float = 0.0           # g / cm^3
    Gully_Percent_Fine: float = 0.0                # %
    Gully_Year_Disturb: int = 1900
    Gully_Cross_Section_Area: float = 0.0          # m^2
    areaInSquareMeters: float = 0.0                # FU area, m^2

    # Scenario-wide settings
    Gully_SDR_Fine: float = 100.0
    Gully_SDR_Coarse: float = 50.0
    Gully_Year_Density_Raster: int = 2003
    Gully_End_Year: int = 2007
    Average_Gully_Activity_Factor: float = 1.0
    Gully_Management_Practice_Factor: float = 1.0
    Gully_Daily_Runoff_Power_Factor: float = 1.4
    gullyGrowthModel: str = "Linear"
    gullyModelType: str = "SEDNET"

    # Derived (filled in by compute_gully_volume_and_supply / annual_gully_load)
    Total_Gully_Volume: float = 0.0
    Gully_Annual_Average_Sediment_Supply: float = 0.0
    Gully_Long_Term_Runoff_Factor: float = 0.0
    Gully_Average_Daily_Runoff: float = 0.0
    Annual_Average_Runoff: float = 0.0


# ---------------------------------------------------------------------------
# Volume / supply / growth-curve maths
# ---------------------------------------------------------------------------

def integration_finish(year_density_raster: int,
                       year_disturb: int,
                       year_end: int,
                       activity_factor: float,
                       integration_start: int = 1) -> float:
    """Length of the integration window (in years) used by the linear model.

    Mirrors the ``Integration_Finish`` calculation in
    ``createGullyTimeSeriesInclCalcVolume``: if the gully became inactive
    before the density-raster year, the post-end years are scaled by the
    average gully activity factor.
    """
    if year_density_raster > year_end:
        normal_years = year_end - year_disturb
        altered_years = year_density_raster - year_end
        return normal_years + altered_years * activity_factor + integration_start
    return year_density_raster - year_disturb + integration_start


KM_PER_KM2_TO_M_PER_M2 = 1.0e-3


def total_gully_volume(gully_density_km_per_km2: float,
                       fu_area_m2: float,
                       csa_m2: float) -> float:
    """Total gully volume in the FU (m^3).

    ``gully_density`` is in km/km^2 (as labelled in the Source
    ``GullyParamStats`` export), so multiply by 1e-3 to get m/m^2 -- 1000 m/km
    over 1e6 m^2/km^2. This is the ``KmperSquareKm_to_MetresperSquareMeter``
    constant in C#.

    Confirmed against the FI reference run (``FI-gully-parameteriser-output-
    source-20260522``): solving ``volume / (density * area * csa)`` over the
    3816 FUs with non-zero gully volume gives 1.000e-3 with a standard
    deviation of 5.6e-9, using vector polygon areas from
    ``FI_Subcat-FU_Intersection.shp``.
    """
    return (gully_density_km_per_km2 * KM_PER_KM2_TO_M_PER_M2) * fu_area_m2 * csa_m2


def linear_growth(gully_year: np.ndarray,
                  integration_start: int,
                  integration_end: float) -> np.ndarray:
    """Per-year mass fraction for the linear growth model.

    Replicates ``Gully_Function`` for the linear branch: sums to 1 across years
    1..integration_end, with year 1 returning ``integration_start/integration_end``
    and subsequent years returning ``1/integration_end``.
    """
    y = np.asarray(gully_year, dtype=float)
    out = np.where(
        y <= integration_start,
        integration_start / integration_end,
        y / integration_end - np.maximum(integration_start, y - 1) / integration_end,
    )
    return out


# ---------------------------------------------------------------------------
# Runoff loading
# ---------------------------------------------------------------------------

def _runoff_csv_candidates(directory: Path, var: str, catchment: str, fu: str):
    base = f"FU{ELEMENT_SEPARATOR}{var}"
    suffix = f"{ELEMENT_SEPARATOR}{catchment}{ELEMENT_SEPARATOR}{fu}.csv"
    yield directory / f"{base}{MM_PER_DAY_SUFFIX}{suffix}"
    yield directory / f"{base}{suffix}"


def _read_runoff_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if df.shape[1] == 0:
        raise ValueError(f"No data column in runoff CSV: {path}")
    return df.iloc[:, 0].astype(float)


def event_runoff(runoff: Union[pd.Series, str, os.PathLike],
                 baseflow: Optional[Union[pd.Series, str, os.PathLike]] = None) -> pd.Series:
    """Return event runoff (mm/day) = max(runoff - baseflow, 0).

    Either argument may be a pandas Series (DatetimeIndex) or a path to a
    CSV. If ``baseflow`` is ``None`` the runoff series is returned unchanged.

    The C# implementation hard-fails when the baseflow directory is missing;
    here we let the caller decide by allowing a ``None`` baseflow.
    """
    if not isinstance(runoff, pd.Series):
        runoff = _read_runoff_csv(Path(runoff))
    if baseflow is None:
        return runoff.astype(float)
    if not isinstance(baseflow, pd.Series):
        baseflow = _read_runoff_csv(Path(baseflow))
    aligned_bf = baseflow.reindex(runoff.index).fillna(0.0)
    return (runoff.astype(float) - aligned_bf.astype(float)).clip(lower=0.0)


def load_event_runoff_from_directory(runoff_dir: Union[str, os.PathLike],
                                     catchment: str,
                                     fu: str,
                                     baseflow_dir: Optional[Union[str, os.PathLike]] = None
                                     ) -> pd.Series:
    """Locate runoff (and optional baseflow) CSVs and return event runoff.

    Filename conventions match the C# ``getEventRunoffFromFilesMM`` helper.
    If ``baseflow_dir`` is not given it is derived by replacing ``runoff``
    with ``baseflow`` in the runoff directory's path components.
    """
    runoff_dir = Path(runoff_dir)
    if baseflow_dir is None:
        baseflow_dir = Path(str(runoff_dir).replace(RUNOFF_VAR, BASEFLOW_VAR))
    else:
        baseflow_dir = Path(baseflow_dir)

    def _pick(d: Path, var: str) -> Optional[Path]:
        for candidate in _runoff_csv_candidates(d, var, catchment, fu):
            if candidate.exists():
                return candidate
        return None

    runoff_path = _pick(runoff_dir, RUNOFF_VAR)
    if runoff_path is None:
        raise FileNotFoundError(
            f"No runoff CSV for {catchment}/{fu} under {runoff_dir}")

    baseflow_path = _pick(baseflow_dir, BASEFLOW_VAR) if baseflow_dir.exists() else None
    if baseflow_path is None:
        logger.warning("No baseflow file for %s/%s; using runoff as event runoff",
                       catchment, fu)
    return event_runoff(runoff_path, baseflow_path)


# ---------------------------------------------------------------------------
# Per-FU computation
# ---------------------------------------------------------------------------

def compute_gully_volume_and_supply(params: GullyParameters) -> GullyParameters:
    """Fill in derived volume and long-term sediment supply fields.

    Mutates and returns ``params`` for convenience.
    """
    params.Total_Gully_Volume = total_gully_volume(
        params.Gully_Density, params.areaInSquareMeters,
        params.Gully_Cross_Section_Area)

    finish = integration_finish(
        params.Gully_Year_Density_Raster, params.Gully_Year_Disturb,
        params.Gully_End_Year, params.Average_Gully_Activity_Factor)

    if finish <= 0:
        params.Gully_Annual_Average_Sediment_Supply = 0.0
    else:
        params.Gully_Annual_Average_Sediment_Supply = (
            params.Total_Gully_Volume * params.Gully_Soil_Bulk_Density / finish)
    return params


def annual_runoff_complete_years(daily: pd.Series) -> pd.Series:
    """Calendar-year runoff totals, excluding partial years at either end.

    Labelled by year start, matching the Source ``AnnualRunoff_MM`` export.

    Source only emits complete calendar years. Including a partial year pulls
    down ``annual_avg``, which normalises the whole load series: on the FI
    reference run, whose daily data runs 1993-07-01 to 2023-06-30, keeping the
    two partial years scaled every year of every FU's load by 5-6%. Trimming
    them brings the load to within 1e-15 of Source.
    """
    annual = daily.resample("YS").sum()
    if not len(annual):
        return annual
    starts = annual.index
    ends = starts + pd.offsets.YearEnd(0)
    complete = (starts >= daily.index.min()) & (ends <= daily.index.max())
    return annual[complete]


def annual_gully_load(event_runoff_mm_per_day: pd.Series,
                      params: GullyParameters) -> tuple[pd.Series, pd.Series]:
    """Build annual gully load (kg) and annual runoff (mm/yr) series.

    Returns ``(annual_load, annual_runoff)``. Mutates ``params`` to record
    long-term runoff stats (``Gully_Long_Term_Runoff_Factor``,
    ``Gully_Average_Daily_Runoff``, ``Annual_Average_Runoff``).

    The unit constant of 1000 converts soil bulk density from g/cm^3 to
    kg/m^3, giving a load in kg. The bulk density raster really is in g/cm^3:
    ``BulkDensity_SubSurf2014_FI_v2.tif`` spans 0-1.70 (mean 1.378) and the FI
    reference run reports ``Gully Soil Bulk Density`` over the same 1.005-1.70
    range, matching the ``(g/cm^3)`` unit in the Beckers ``ExpectedResults.csv``
    header.
    """
    daily = event_runoff_mm_per_day.astype(float)

    # Daily runoff stats stored on the model
    power = params.Gully_Daily_Runoff_Power_Factor
    if not np.isfinite(power) or power < 0.5:
        power = 1.4
        params.Gully_Daily_Runoff_Power_Factor = power
    # The two daily statistics are taken over the whole series, including any
    # partial years -- verified against the FI reference run to 8 decimal
    # places. Only the annual series drops them.
    params.Gully_Long_Term_Runoff_Factor = float(np.power(daily, power).mean())
    params.Gully_Average_Daily_Runoff = float(daily.mean())

    annual_runoff = annual_runoff_complete_years(daily)
    annual_avg = float(annual_runoff.mean()) if len(annual_runoff) else 0.0
    params.Annual_Average_Runoff = annual_avg

    integration_start = 1
    finish = integration_finish(
        params.Gully_Year_Density_Raster, params.Gully_Year_Disturb,
        params.Gully_End_Year, params.Average_Gully_Activity_Factor,
        integration_start)

    if (params.Gully_Density <= 0
            or annual_avg == 0
            or finish <= 0):
        annual_load = pd.Series(0.0, index=annual_runoff.index, name=GULLY_ANNUAL_LOAD)
        annual_runoff.name = GULLY_ANNUAL_RUNOFF
        return annual_load, annual_runoff

    years = annual_runoff.index.year.to_numpy()
    gully_year = years - params.Gully_Year_Disturb + integration_start
    pre_disturbance = years < params.Gully_Year_Disturb

    year_prop = linear_growth(gully_year, integration_start, finish)
    year_prop = np.where(pre_disturbance | (gully_year == 0), 0.0, year_prop)

    G_PER_CM3_TO_KG_PER_M3 = 1000.0
    load = (annual_runoff.to_numpy() / annual_avg) * year_prop \
        * params.Total_Gully_Volume \
        * (params.Gully_Soil_Bulk_Density * G_PER_CM3_TO_KG_PER_M3)

    annual_load = pd.Series(load, index=annual_runoff.index, name=GULLY_ANNUAL_LOAD)
    annual_runoff.name = GULLY_ANNUAL_RUNOFF
    return annual_load, annual_runoff


# ---------------------------------------------------------------------------
# Zonal-stats orchestration
# ---------------------------------------------------------------------------

def _ensure_zonal_table(zonal_stats: pd.DataFrame,
                        catchment_col: str,
                        fu_col: str) -> pd.DataFrame:
    missing = {catchment_col, fu_col} - set(zonal_stats.columns)
    if missing:
        raise KeyError(f"zonal_stats is missing required columns: {missing}")
    return zonal_stats


def _fill_uniform_columns(zonal_stats: pd.DataFrame,
                          uniform_values: Mapping[str, float]) -> pd.DataFrame:
    out = zonal_stats.copy()
    for name, value in uniform_values.items():
        if name not in out.columns:
            logger.info("Using uniform value %s = %s for all FUs", name, value)
            out[name] = value
        else:
            out[name] = out[name].where(out[name].notna(), value)
    return out


def _scenario_defaults(zonal_stats: pd.DataFrame,
                       default_year_disturb: int) -> dict:
    """Compute scenario-wide fallback values used when an FU has no zonal hit.

    Mean for continuous parameters; minimum (oldest year) for year-of-disturbance.
    Zero year-of-disturbance entries are first remapped to the supplied default.
    """
    yod = zonal_stats[YEAR_OF_DISTURBANCE].replace(0, default_year_disturb)
    return {
        GULLY_DENSITY: float(zonal_stats[GULLY_DENSITY].mean()),
        SOIL_BD: float(zonal_stats[SOIL_BD].mean()),
        B_HORIZON_CLAY_PCT: float(zonal_stats[B_HORIZON_CLAY_PCT].mean()),
        GULLY_CSA: float(zonal_stats[GULLY_CSA].mean()),
        YEAR_OF_DISTURBANCE: int(yod.min()) if len(yod) else default_year_disturb,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def parameterise_gullies(zonal_stats: pd.DataFrame,
                         fu_areas: Mapping[tuple, float],
                         runoff: Union[pd.DataFrame, str, os.PathLike],
                         *,
                         baseflow: Optional[Union[pd.DataFrame, str, os.PathLike]] = None,
                         catchment_col: str = "catchment",
                         fu_col: str = "fu",
                         uniform_values: Optional[Mapping[str, float]] = None,
                         scenario: Optional[Mapping] = None,
                         output_dir: Optional[Union[str, os.PathLike]] = None,
                         ) -> dict:
    """Parameterise gully models for every (catchment, FU) in ``zonal_stats``.

    Parameters
    ----------
    zonal_stats
        DataFrame with one row per (catchment, FU). Must contain ``catchment_col``
        and ``fu_col`` plus any subset of the columns named by ``RASTER_PARAMS``.
        Missing columns must be supplied via ``uniform_values``.
    fu_areas
        Mapping of ``(catchment, fu) -> area_m2``. FUs with zero or missing
        area are skipped (their parameters are not produced).
    runoff
        Either a long-form DataFrame indexed by date with columns
        ``MultiIndex[(catchment, fu)]`` of daily runoff (mm/day), or a path
        to a directory of per-FU CSVs following the C# naming convention.
    baseflow
        Same shape as ``runoff``; used to derive event runoff. If ``None``
        and ``runoff`` is a directory, a sibling ``baseflow`` directory is
        looked up.
    catchment_col, fu_col
        Column names within ``zonal_stats`` identifying the catchment and FU.
    uniform_values
        Fallback scalars for any of the five spatial parameters not present
        as a column in ``zonal_stats``.
    scenario
        Mapping of scenario-wide settings. Recognised keys correspond to
        :class:`GullyParameters` fields (e.g. ``Gully_Year_Density_Raster``,
        ``Gully_End_Year``, ``Gully_SDR_Fine``, ``Gully_SDR_Coarse``,
        ``Average_Gully_Activity_Factor``, ``Gully_Management_Practice_Factor``,
        ``Gully_Daily_Runoff_Power_Factor``, ``gullyModelType``,
        ``gullyGrowthModel``, ``Default_Gully_Start_Year``).
    output_dir
        If provided, per-FU annual load and annual runoff CSVs are written
        here, plus a long-format ``GullyParamStats.csv``. Filenames match
        the C# convention.

    Returns
    -------
    dict
        ``{
            "parameters": DataFrame indexed by (catchment, fu),
            "annual_load": dict[(catchment, fu) -> Series],
            "annual_runoff": dict[(catchment, fu) -> Series],
        }``
    """
    scenario = dict(scenario or {})
    default_year_disturb = scenario.pop("Default_Gully_Start_Year", 1900)
    uniform_values = dict(uniform_values or {})

    table = _ensure_zonal_table(zonal_stats, catchment_col, fu_col)
    table = _fill_uniform_columns(table, uniform_values)

    missing_params = [p for p in RASTER_PARAMS if p not in table.columns]
    if missing_params:
        raise KeyError(
            "No raster column or uniform_values fallback for: "
            + ", ".join(missing_params))

    # Replace zero years-of-disturbance with the user's default *before*
    # computing scenario-wide fallbacks, matching fixStartYearForGullies.
    table[YEAR_OF_DISTURBANCE] = (
        table[YEAR_OF_DISTURBANCE].replace(0, default_year_disturb))

    defaults = _scenario_defaults(table, default_year_disturb)

    runoff_loader = _make_runoff_loader(runoff, baseflow)

    rows = []
    annual_load = {}
    annual_runoff = {}

    for _, row in table.iterrows():
        catchment = row[catchment_col]
        fu = row[fu_col]
        area = float(fu_areas.get((catchment, fu), 0.0) or 0.0)
        if area <= 0:
            logger.debug("Skipping zero-area FU %s/%s", catchment, fu)
            continue

        params = GullyParameters(areaInSquareMeters=area)
        for k, v in scenario.items():
            if hasattr(params, k):
                setattr(params, k, v)

        for col, attr in (
                (GULLY_DENSITY, "Gully_Density"),
                (SOIL_BD, "Gully_Soil_Bulk_Density"),
                (B_HORIZON_CLAY_PCT, "Gully_Percent_Fine"),
                (GULLY_CSA, "Gully_Cross_Section_Area")):
            value = row.get(col)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = defaults[col]
            setattr(params, attr, float(value))

        yod = row.get(YEAR_OF_DISTURBANCE)
        if yod is None or (isinstance(yod, float) and np.isnan(yod)) or yod == 0:
            yod = defaults[YEAR_OF_DISTURBANCE]
        params.Gully_Year_Disturb = int(yod)

        compute_gully_volume_and_supply(params)

        try:
            event = runoff_loader(catchment, fu)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s/%s: %s", catchment, fu, exc)
            continue

        load, run = annual_gully_load(event, params)
        annual_load[(catchment, fu)] = load
        annual_runoff[(catchment, fu)] = run

        rec = asdict(params)
        rec[catchment_col] = catchment
        rec[fu_col] = fu
        rows.append(rec)

        if output_dir is not None:
            _write_fu_outputs(output_dir, catchment, fu, load, run)

    parameters = pd.DataFrame(rows).set_index([catchment_col, fu_col])

    if output_dir is not None:
        _write_param_stats(output_dir, parameters, catchment_col, fu_col)

    return {
        "parameters": parameters,
        "annual_load": annual_load,
        "annual_runoff": annual_runoff,
    }


def _make_runoff_loader(runoff, baseflow):
    if isinstance(runoff, pd.DataFrame):
        runoff_df = runoff
        baseflow_df = baseflow if isinstance(baseflow, pd.DataFrame) else None

        def load(catchment, fu):
            try:
                r = runoff_df[(catchment, fu)]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"No runoff column for ({catchment}, {fu})") from exc
            b = None
            if baseflow_df is not None and (catchment, fu) in baseflow_df.columns:
                b = baseflow_df[(catchment, fu)]
            return event_runoff(r, b)
        return load

    runoff_dir = Path(runoff)
    baseflow_dir = Path(baseflow) if baseflow is not None else None

    def load(catchment, fu):
        return load_event_runoff_from_directory(
            runoff_dir, catchment, fu, baseflow_dir)
    return load


def _write_fu_outputs(output_dir, catchment, fu, load, run):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{catchment}{ELEMENT_SEPARATOR}{fu}{ELEMENT_SEPARATOR}"
    load.to_csv(output_dir / f"{base}{GULLY_ANNUAL_LOAD}.csv", header=True)
    run.to_csv(output_dir / f"{base}{GULLY_ANNUAL_RUNOFF}.csv", header=True)


def _write_param_stats(output_dir, parameters: pd.DataFrame,
                       catchment_col: str, fu_col: str):
    long = parameters.reset_index().melt(
        id_vars=[catchment_col, fu_col], var_name="Parameter", value_name="Value")
    long.insert(2, "Constituent", "Sediment - Fine")
    long = long.rename(columns={catchment_col: "SubCat", fu_col: "FU"})
    long.to_csv(Path(output_dir) / PARAM_STATS_FILENAME, index=False)
