"""
Runoff adjustment: distribute monthly loads to daily timesteps proportional
to daily runoff, and compute runoff ratio between two sources.

Translates GBRToolsGeneral.calculateIntraMonthlyFlows (GBRToolsGeneral.cs
lines 36-87) and the runoff ratio calculation from ApsimAdjustor
(APSIMparameterisationModel.cs lines 1591-1634, 2428-2483).
"""
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_intra_monthly_flows(daily_event_runoff, monthly_load):
    """Distribute monthly loads across days proportional to daily runoff.

    For each month in the overlapping period:
    - If monthly runoff > 0: daily_load = (monthly_load + carryover) * (daily_runoff / monthly_runoff)
    - If monthly runoff == 0: carryover += monthly_load

    Translated from GBRToolsGeneral.calculateIntraMonthlyFlows
    (GBRToolsGeneral.cs lines 36-87).

    Parameters
    ----------
    daily_event_runoff : Series
        Daily runoff time series with DatetimeIndex.
    monthly_load : Series
        Monthly load time series with DatetimeIndex (period start dates).
        If the index is daily or has more than monthly resolution, it will
        be resampled to monthly sums.

    Returns
    -------
    Series
        Daily adjusted load time series.
    """
    runoff_idx = pd.to_datetime(daily_event_runoff.index)
    load_idx = pd.to_datetime(monthly_load.index)

    # Resample to monthly if needed
    if not _is_monthly_fast(load_idx):
        monthly_load = pd.Series(monthly_load.values, index=load_idx).resample("MS").sum()
        load_idx = monthly_load.index

    # Determine overlapping date range
    start = max(runoff_idx.min(), load_idx.min())
    end = min(runoff_idx.max(), load_idx.max())

    date_range = pd.date_range(start, end, freq="D")
    n_days = len(date_range)

    # Align daily runoff to the date range as a numpy array
    daily_ro = pd.Series(daily_event_runoff.values, index=runoff_idx).reindex(
        date_range, fill_value=0.0
    ).values.astype(np.float64)

    # Compute month labels for each day (year*12 + month) for grouping
    years = date_range.year.values
    months = date_range.month.values
    month_labels = years * 12 + months

    # Get unique months and their boundaries
    unique_labels, first_idx = np.unique(month_labels, return_index=True)
    # End indices (exclusive) for each month
    end_idx = np.empty_like(first_idx)
    end_idx[:-1] = first_idx[1:]
    end_idx[-1] = n_days

    # Compute monthly runoff sums
    monthly_ro_sums = np.empty(len(unique_labels), dtype=np.float64)
    for i in range(len(unique_labels)):
        monthly_ro_sums[i] = daily_ro[first_idx[i]:end_idx[i]].sum()

    # Build lookup for monthly load values
    load_vals = np.zeros(len(unique_labels), dtype=np.float64)
    load_index_labels = load_idx.year.values * 12 + load_idx.month.values
    load_lookup = dict(zip(load_index_labels, monthly_load.values))
    for i, label in enumerate(unique_labels):
        load_vals[i] = load_lookup.get(label, 0.0)

    # Compute effective loads with carryover (sequential scan over months)
    effective_loads = np.empty(len(unique_labels), dtype=np.float64)
    carryover = 0.0
    for i in range(len(unique_labels)):
        if monthly_ro_sums[i] > 0:
            effective_loads[i] = load_vals[i] + carryover
            carryover = 0.0
        else:
            effective_loads[i] = 0.0
            carryover += load_vals[i]

    # Vectorised daily distribution
    out = np.zeros(n_days, dtype=np.float64)
    for i in range(len(unique_labels)):
        if effective_loads[i] > 0 and monthly_ro_sums[i] > 0:
            s, e = first_idx[i], end_idx[i]
            out[s:e] = daily_ro[s:e] * (effective_loads[i] / monthly_ro_sums[i])

    return pd.Series(out, index=date_range)


def event_flow_from_totals(runoff_dir=None, baseflow_dir=None, parent_dir=None,
                           separator='$'):
    """Compute event flow (runoff - baseflow) from directories of CSV files.

    Returns a DataFrame with MultiIndex columns ``(catchment, fu)`` containing
    daily event flow time series.

    Parameters
    ----------
    runoff_dir : str, optional
        Directory containing runoff CSVs named
        ``FU$Runoff_mmPerDay$<catchment>$<fu>.csv``.
    baseflow_dir : str, optional
        Directory containing baseflow CSVs named
        ``FU$Baseflow_mmPerDay$<catchment>$<fu>.csv``.
    parent_dir : str, optional
        Parent directory containing ``Runoff`` and ``Baseflow`` subdirectories.
        Used when *runoff_dir* and *baseflow_dir* are not provided.
    separator : str
        Separator used in file naming (default '$').

    Returns
    -------
    DataFrame
        Daily event flow with MultiIndex columns ``(catchment, fu)``.
    """
    import os

    if parent_dir is not None:
        if runoff_dir is None:
            runoff_dir = os.path.join(parent_dir, 'Runoff')
        if baseflow_dir is None:
            baseflow_dir = os.path.join(parent_dir, 'Baseflow')

    if runoff_dir is None:
        raise ValueError("Either parent_dir or runoff_dir must be provided")

    runoff_prefix = f'FU{separator}Runoff_mmPerDay{separator}'
    baseflow_prefix = f'FU{separator}Baseflow_mmPerDay{separator}'

    # Parse catchment/fu from runoff filenames
    combos = {}
    for fname in os.listdir(runoff_dir):
        if not fname.lower().endswith('.csv') or not fname.startswith(runoff_prefix):
            continue
        stem = fname[len(runoff_prefix):-4]
        parts = stem.split(separator, 1)
        if len(parts) == 2:
            combos[(parts[0], parts[1])] = fname

    if not combos:
        logger.warning("No runoff files found in %s", runoff_dir)
        return pd.DataFrame()

    collected = {}
    for (catchment, fu), ro_fname in sorted(combos.items()):
        ro_path = os.path.join(runoff_dir, ro_fname)
        ro = pd.read_csv(ro_path, index_col=0, parse_dates=True).iloc[:, 0]

        if baseflow_dir is not None:
            bf_fname = f'{baseflow_prefix}{catchment}{separator}{fu}.csv'
            bf_path = os.path.join(baseflow_dir, bf_fname)
            if os.path.exists(bf_path):
                bf = pd.read_csv(bf_path, index_col=0, parse_dates=True).iloc[:, 0]
                event = (ro - bf.reindex(ro.index, fill_value=0.0)).clip(lower=0.0)
            else:
                logger.debug("No baseflow file for %s / %s, using total runoff", catchment, fu)
                event = ro
        else:
            event = ro

        collected[(catchment, fu)] = event

    columns = pd.MultiIndex.from_tuples(
        list(collected.keys()), names=['catchment', 'fu']
    )
    return pd.DataFrame(collected, columns=columns)


def compute_runoff_ratio(ds_runoff_daily, apsim_runoff):
    """Compute ratio of APSIM runoff to DS runoff over their common period.

    runoff_ratio = sum(apsim_runoff) / sum(ds_runoff)

    If ds_runoff total is 0, returns 1.0 (or 0.0 if apsim is also 0).

    Translated from ApsimAdjustor.CalculateMonthlyFlowRatios
    (APSIMparameterisationModel.cs lines 2428-2483) and the inline
    runoff ratio calculation (lines 1591-1634).

    Parameters
    ----------
    ds_runoff_daily : Series
        Daily DS (Dynamic SedNet) runoff time series.
    apsim_runoff : Series
        APSIM or HowLeaky runoff time series (daily or monthly).

    Returns
    -------
    float
        Runoff ratio (apsim_total / ds_total).
    """
    ds = ds_runoff_daily.copy()
    apsim = apsim_runoff.copy()

    ds.index = pd.to_datetime(ds.index)
    apsim.index = pd.to_datetime(apsim.index)

    # Convert both to monthly for comparison
    ds_monthly = ds.resample("MS").sum()
    if not _is_monthly(apsim):
        apsim_monthly = apsim.resample("MS").sum()
    else:
        apsim_monthly = apsim

    # Common period
    start = max(ds_monthly.index.min(), apsim_monthly.index.min())
    end = min(ds_monthly.index.max(), apsim_monthly.index.max())

    common = pd.date_range(start, end, freq="MS")
    ds_total = ds_monthly.reindex(common, fill_value=0.0).sum()
    apsim_total = apsim_monthly.reindex(common, fill_value=0.0).sum()

    if ds_total > 0:
        return apsim_total / ds_total
    elif apsim_total > 0:
        return 1.0
    else:
        return 0.0


def _is_monthly(series):
    """Check if a series has monthly-ish frequency."""
    return _is_monthly_fast(pd.to_datetime(series.index))


def _is_monthly_fast(dt_index):
    """Check if a DatetimeIndex has monthly-ish frequency."""
    if len(dt_index) < 2:
        return False
    # Check a few diffs rather than computing full median
    diffs_ns = np.diff(dt_index.values[:min(5, len(dt_index))]).astype(np.int64)
    median_days = np.median(diffs_ns) / (24 * 3600 * 1e9)
    return 25 <= median_days <= 35
