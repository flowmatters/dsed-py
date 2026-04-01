"""
HowLeaky time series import with runoff adjustment.

Reads pre-accumulated HowLeaky CSVs, applies runoff adjustment, and writes
adjusted daily CSVs. Similar to APSIM import but with HowLeaky-specific
constituent handling and Rattray file naming variants.

Translated from ImportHowLeakyTimeseriesForm.doRunoffAdjusting()
(ImportHowLeakyTimeseriesForm.cs lines 137-462) and
ImportHowLeakyTimeSeriesModel (lines 654-881).
"""
import os
import logging
import shutil

import pandas as pd

from . import cropping_const as cc
from .runoff_adjust import calculate_intra_monthly_flows
from .apsim_timeseries import scan_raw_directory, _build_output_dataframe, _make_column_label, _make_variable_name
from .area_weighted import load_shapefile_data, area_weighted_average, simple_average

logger = logging.getLogger(__name__)

# Map HowLeaky/Rattray file-naming variants to canonical constituent names.
# scan_raw_directory extracts the raw 3rd-part of the filename, which for
# HowLeaky files uses different naming conventions to APSIM.  This mapping
# is applied to scanned constituent names so the dispatch logic in
# import_howleaky_timeseries matches correctly.
HL_CONSTITUENT_ALIASES = {
    cc.HL_EROSION_RATTRAY: cc.FINE_SED,                       # Erosion -> Sediment - Fine
    cc.HL_N03_RUNOFF_RATTRAY: cc.N_DIN,                       # N03NRunoffLoad -> N_DIN
    cc.HL_NO3_RUNOFF_CORRECT_RATTRAY: cc.N_DIN,               # NO3NRunoffLoad -> N_DIN
    cc.HL_DIN_DRAINAGE_RATTRAY: cc.N_DIN,                     # DIN_Drainage -> N_DIN
    cc.HL_DISSOLVED_P_OUTPUT: cc.P_DOP,                       # Dissolved P Export -> P_DOP
    cc.HL_PEST_SED_PHASE_RATTRAY: None,                       # handled via pesticide dispatch
    cc.HL_PEST_WATER_PHASE_RATTRAY: None,                     # handled via pesticide dispatch
}


def _load_timeseries(filepath):
    """Load a single time series CSV with date index."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True,dayfirst=True,header=None,names=['Date','Value'])
    return df.iloc[:, 0]


def _load_runoff(runoff_source, catchment, fu, separator=cc.SEPARATOR):
    """Load daily event runoff time series for a catchment/FU.

    Parameters
    ----------
    runoff_source : str or DataFrame
        Either a directory path containing runoff CSVs, or a DataFrame with
        MultiIndex columns ``(catchment, fu)`` or ``(catchment, hru)``.
    catchment : str
        Catchment name.
    fu : str
        Functional unit name.
    separator : str
        Separator used in file naming (only used when *runoff_source* is a path).

    Returns
    -------
    Series or None
    """
    if isinstance(runoff_source, pd.DataFrame):
        from .apsim_timeseries import _load_runoff_from_dataframe
        return _load_runoff_from_dataframe(runoff_source, catchment, fu)

    patterns = [
        f"{catchment}{separator}{fu}.csv",
        f"{catchment}{separator}{fu}{separator}Runoff.csv",
    ]
    for pat in patterns:
        path = os.path.join(runoff_source, pat)
        if os.path.exists(path):
            return _load_timeseries(path)
    return None


def _find_file(raw_dir, *candidates):
    """Return the first candidate path that exists, or None."""
    for name in candidates:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            return path
    return None


def _write_adjusted(output_dir, filename, series):
    """Write an adjusted time series to CSV."""
    path = os.path.join(output_dir, filename)
    series.to_csv(path, header=True)


def _make_output_name(catchment, fu, constituent, suffix, units, separator, phase=None):
    """Build output filename."""
    parts = [catchment, fu, constituent]
    if phase:
        parts.append(phase)
    parts.append(suffix)
    return separator.join(parts) + f"_({units}).csv"


def import_howleaky_timeseries(
    raw_dir,
    runoff_source,
    output_dir=None,
    constituents=None,
    catchments=None,
    fus=None,
    separator=cc.SEPARATOR,
    start_date=None,
    end_date=None,
):
    """Import pre-accumulated HowLeaky time series with runoff adjustment.

    Handles constituent-specific file naming including Rattray 2018 variants.

    Any of *constituents*, *catchments* and *fus* may be ``None``, in which
    case they are inferred by scanning the CSV filenames in *raw_dir*.  When
    provided, they act as filters restricting which combinations are processed.

    Translated from ImportHowLeakyTimeseriesForm.doRunoffAdjusting()
    (ImportHowLeakyTimeseriesForm.cs lines 137-462).

    Parameters
    ----------
    raw_dir : str
        Directory containing pre-accumulated CSVs.
    runoff_source : str or DataFrame
        Either a directory path containing daily runoff CSVs, or a DataFrame
        with MultiIndex columns ``(catchment, fu)`` or ``(catchment, hru)``.
    output_dir : str, optional
        Directory to write adjusted daily CSVs.  If ``None``, results are
        returned as a single DataFrame instead of being written to disk.
    constituents : list of str, optional
        Constituent names to process.  Inferred from *raw_dir* if ``None``.
    catchments : list of str, optional
        Catchment names.  Inferred from *raw_dir* if ``None``.
    fus : list of str, optional
        Functional unit names.  Inferred from *raw_dir* if ``None``.
    separator : str
        Separator used in file naming.
    start_date : str or Timestamp, optional
        If given, restrict output time series to dates on or after this date.
    end_date : str or Timestamp, optional
        If given, restrict output time series to dates on or before this date.

    Returns
    -------
    DataFrame or dict
        If *output_dir* is ``None``, returns a DataFrame with a MultiIndex
        column ``(catchment, fu, variable)`` containing all adjusted daily
        time series.  Otherwise, returns a dict tallying processed files per
        constituent.
    """
    # Infer any missing parameters from raw directory contents
    scanned_catchments, scanned_fus, scanned_constituents = scan_raw_directory(
        raw_dir, separator
    )

    if catchments is None:
        catchments = scanned_catchments
        logger.info("Inferred %d catchments from %s", len(catchments), raw_dir)
    else:
        catchments = [c for c in catchments if c in set(scanned_catchments)] or list(catchments)

    if fus is None:
        fus = scanned_fus
        logger.info("Inferred %d FUs from %s", len(fus), raw_dir)
    else:
        fus = [f for f in fus if f in set(scanned_fus)] or list(fus)

    if constituents is None:
        # Map HowLeaky/Rattray naming variants to canonical constituent names
        mapped = set()
        for c in scanned_constituents:
            canonical = HL_CONSTITUENT_ALIASES.get(c, c)
            if canonical is not None:
                mapped.add(canonical)
        constituents = sorted(mapped)
        logger.info("Inferred %d constituents from %s", len(constituents), raw_dir)
    else:
        constituents = list(constituents)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Build units map
    units_map = dict(cc.CONSTITUENT_UNITS)
    for c in constituents:
        if c not in units_map:
            units_map[c] = cc.UNITS_G_PER_HA

    tally = {c: 0 for c in constituents}
    if cc.P_DOP in constituents:
        tally[cc.P_FRP] = 0
    counts_by_catchment = {c: 0 for c in catchments}
    counts_by_fu = {f: 0 for f in fus}
    collected = {}  # (catchment, fu, variable) -> Series

    # Dispatch table: constituent name -> handler function.
    # P_FRP maps to None (handled by P_DOP). Unknown constituents use
    # _process_hl_pesticide as the default.
    handlers = {
        cc.FINE_SED: _process_hl_fine_sed,
        cc.N_DIN: _process_hl_din,
        cc.P_PARTICULATE: _process_hl_particulate_p,
        cc.P_DOP: _process_hl_dissolved_p,
        cc.P_FRP: None,
        cc.AGGREGATED_RUNOFF: _process_hl_runoff_copy,
    }

    for catchment in catchments:
        for fu in fus:
            runoff_ts = _load_runoff(runoff_source, catchment, fu, separator)
            if runoff_ts is None:
                logger.warning("No runoff found for %s %s; skipping", catchment, fu)
                continue

            for const_name in constituents:
                units = units_map.get(const_name, cc.UNITS_G_PER_HA)
                units_bit = "_" + units

                handler = handlers.get(const_name, _process_hl_pesticide)
                if handler is None:
                    continue

                results = handler(
                    raw_dir, runoff_ts, catchment, fu,
                    const_name, units_bit, units, separator,
                )

                if not results:
                    logger.warning("No data found for %s %s %s; skipping",
                                   catchment, fu, const_name)
                    continue

                # Update tallies
                if const_name == cc.P_DOP:
                    tally[cc.P_DOP] += 1
                    tally[cc.P_FRP] += 1
                else:
                    tally[const_name] += 1
                counts_by_catchment[catchment] += 1
                counts_by_fu[fu] += 1

                for entry in results:
                    series = entry["series"]
                    if start_date is not None or end_date is not None:
                        series = series.loc[start_date:end_date]
                    s_units = entry.get("units", units)
                    if output_dir is not None:
                        fname = _make_output_name(
                            catchment, fu, entry["constituent"],
                            entry["suffix"], s_units, separator,
                            phase=entry.get("phase"),
                        )
                        series = series.rename(
                            _make_column_label(
                                entry["constituent"], s_units,
                            )
                        )
                        _write_adjusted(output_dir, fname, series)
                    else:
                        var = _make_variable_name(
                            entry["constituent"], entry["suffix"],
                            phase=entry.get("phase"),
                        )
                        collected[(catchment, fu, var)] = series

    # Report zero-count dimensions
    for catchment, count in counts_by_catchment.items():
        if count == 0:
            logger.error("No files processed for catchment %s", catchment)
    for fu, count in counts_by_fu.items():
        if count == 0:
            logger.error("No files processed for FU %s", fu)
    for const_name, count in tally.items():
        if count == 0:
            logger.error("No files processed for constituent %s", const_name)

    logger.info("HowLeaky timeseries import complete: %s", tally)

    if output_dir is None:
        return _build_output_dataframe(collected)
    return tally


def _process_hl_fine_sed(raw_dir, runoff_ts, catchment, fu,
                         const_name, units_bit, units, separator):
    """Process HowLeaky Fine Sediment with Rattray naming fallback.

    Returns list of entry dicts, or empty list.
    """
    base = f"{catchment}{separator}{fu}{separator}{cc.FINE_SED}{units_bit}.csv"
    rattray = f"{catchment}{separator}{fu}{separator}{cc.HL_EROSION_RATTRAY}_{units.replace(' ', '')}.csv"

    path = _find_file(raw_dir, base, rattray)
    if path is None:
        logger.debug("No Fine Sed file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    return [{"constituent": cc.FINE_SED, "suffix": cc.TS_DAILY_ADJUSTED,
             "series": adjusted}]


def _process_hl_din(raw_dir, runoff_ts, catchment, fu,
                    const_name, units_bit, units, separator):
    """Process HowLeaky DIN with Rattray N03/NO3 naming fallbacks and DIN drainage.

    Returns list of entry dicts, or empty list.
    """
    base = f"{catchment}{separator}{fu}{separator}{cc.N_DIN}{units_bit}.csv"
    rattray_n03 = f"{catchment}{separator}{fu}{separator}{cc.HL_N03_RUNOFF_RATTRAY}_{units.replace(' ', '')}.csv"
    rattray_no3 = f"{catchment}{separator}{fu}{separator}{cc.HL_NO3_RUNOFF_CORRECT_RATTRAY}_{units.replace(' ', '')}.csv"

    path = _find_file(raw_dir, base, rattray_n03, rattray_no3)
    if path is None:
        logger.debug("No DIN file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(path)
    try:
        monthly = raw_ts.resample("MS").sum()
    except Exception as e:
        logger.error("Error resampling DIN time series for %s %s: %s",
                     catchment, fu, e)
        logger.error(raw_ts.head())
        raise
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    results = [{"constituent": cc.N_DIN, "suffix": cc.TS_DAILY_ADJUSTED,
                "series": adjusted}]

    # Check for DIN Drainage (leached) if we used Rattray naming
    do_leached = (base not in os.path.basename(path))
    if do_leached:
        drain_name = os.path.basename(path)
        for rattray_str in [cc.HL_N03_RUNOFF_RATTRAY, cc.HL_NO3_RUNOFF_CORRECT_RATTRAY]:
            drain_name = drain_name.replace(rattray_str, cc.HL_DIN_DRAINAGE_RATTRAY)
        drain_path = os.path.join(raw_dir, drain_name)

        if os.path.exists(drain_path):
            drain_ts = _load_timeseries(drain_path)
            drain_monthly = drain_ts.resample("MS").sum()
            drain_adjusted = calculate_intra_monthly_flows(runoff_ts, drain_monthly)

            results.append({"constituent": cc.APSIM_NLEACHED_FIELD,
                            "suffix": cc.TS_DAILY_ADJUSTED,
                            "series": drain_adjusted})

    return results


def _process_hl_particulate_p(raw_dir, runoff_ts, catchment, fu,
                              const_name, units_bit, units, separator):
    """Process HowLeaky Particulate P.

    Returns list of entry dicts, or empty list.
    """
    base = f"{catchment}{separator}{fu}{separator}{cc.P_PARTICULATE}{units_bit}.csv"
    rattray = f"{catchment}{separator}{fu}{separator}{cc.P_PARTICULATE}_{units.replace(' ', '')}.csv"

    path = _find_file(raw_dir, base, rattray)
    if path is None:
        logger.debug("No Particulate P file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    return [{"constituent": cc.P_PARTICULATE, "suffix": cc.TS_DAILY_ADJUSTED,
             "series": adjusted}]


def _process_hl_dissolved_p(raw_dir, runoff_ts, catchment, fu,
                            const_name, units_bit, units, separator):
    """Process HowLeaky Dissolved P: produce both DOP and FRP from same input.

    Returns list of entry dicts, or empty list.
    """
    base = f"{catchment}{separator}{fu}{separator}{cc.HL_DISSOLVED_P_OUTPUT}{units_bit}.csv"
    rattray = f"{catchment}{separator}{fu}{separator}{cc.P_DOP}_{units.replace(' ', '')}.csv"

    path = _find_file(raw_dir, base, rattray)
    if path is None:
        logger.debug("No Dissolved P file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    return [
        {"constituent": cc.P_DOP, "suffix": cc.TS_DAILY_ADJUSTED,
         "series": adjusted},
        {"constituent": cc.P_FRP, "suffix": cc.TS_DAILY_ADJUSTED,
         "series": adjusted.copy()},
    ]


def _process_hl_runoff_copy(raw_dir, runoff_ts, catchment, fu,
                            const_name, units_bit, units, separator):
    """Process HowLeaky Aggregated Runoff: load as-is.

    Returns list of entry dicts, or empty list.
    """
    base = f"{catchment}{separator}{fu}{separator}{cc.AGGREGATED_RUNOFF}{units_bit}.csv"
    rattray = f"{catchment}{separator}{fu}{separator}{cc.AGGREGATED_RUNOFF}_{units.replace(' ', '')}.csv"

    path = _find_file(raw_dir, base, rattray)
    if path is None:
        logger.debug("No Aggregated Runoff file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(path)
    return [{"constituent": cc.AGGREGATED_RUNOFF, "suffix": cc.TS_DAILY_NOT_ADJUSTED,
             "series": raw_ts}]


def _process_hl_pesticide(raw_dir, runoff_ts, catchment, fu,
                          const_name, units_bit, units, separator):
    """Process HowLeaky pesticide with Rattray naming variants.

    Returns list of entry dicts, or empty list.
    """
    input_name = const_name
    if const_name == cc.TWOFOUR_D:
        input_name_rattray = cc.RATTRAY_TWOFOUR_D
    else:
        input_name_rattray = const_name.lower()

    # Sed phase
    base_sed = f"{catchment}{separator}{fu}{separator}{input_name}{separator}{cc.SED_PHASE}{units_bit}.csv"
    rattray_sed = (
        f"{catchment}{separator}{fu}{separator}"
        f"{cc.HL_PEST_SED_PHASE_RATTRAY}_{input_name_rattray}_{units.replace(' ', '')}.csv"
    )

    sed_path = _find_file(raw_dir, base_sed, rattray_sed)
    if sed_path is None:
        logger.debug("No pesticide sed phase for %s / %s / %s", catchment, fu, const_name)
        return []

    raw_ts = _load_timeseries(sed_path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    results = [{"constituent": const_name, "suffix": cc.TS_DAILY_ADJUSTED,
                "phase": cc.SED_PHASE, "series": adjusted}]

    # Water phase - derive from sed phase path
    water_path = sed_path.replace("Sediment", "Water")
    if os.path.exists(water_path):
        raw_ts_w = _load_timeseries(water_path)
        monthly_w = raw_ts_w.resample("MS").sum()
        adjusted_w = calculate_intra_monthly_flows(runoff_ts, monthly_w)

        results.append({"constituent": const_name, "suffix": cc.TS_DAILY_ADJUSTED,
                        "phase": cc.WATER_PHASE, "series": adjusted_w})

    return results


def import_howleaky_timeseries_model(
    shapefile,
    ts_dir,
    fu_combos,
    constituent_type,
    use_weighted_averages=True,
    subcatch_field=cc.SUBCATCH_FIELD,
    fu_field=cc.FU_FIELD,
    area_field=cc.AREA_FIELD,
):
    """Build spatially-varied parameter table from shapefile for HowLeaky TS import.

    Translated from ImportHowLeakyTimeSeriesModel (lines 654-881).

    Parameters
    ----------
    shapefile : str or GeoDataFrame
        Path to intersected shapefile.
    ts_dir : str
        Directory containing time series files.
    fu_combos : list of tuple
        List of (catchment, fu) tuples to process.
    constituent_type : str
        Type of constituent being processed (affects which fields are used).
    use_weighted_averages : bool
        If True, use area-weighted averages; otherwise simple averages.
    subcatch_field : str
        Shapefile field for subcatchment names.
    fu_field : str
        Shapefile field for FU names.
    area_field : str
        Shapefile field for polygon areas.

    Returns
    -------
    DataFrame
        Parameter values per catchment/FU with columns for clay, SDR, DWC, etc.
    """
    gdf = load_shapefile_data(shapefile)
    group_cols = [subcatch_field, fu_field]

    # Determine which fields to compute
    value_cols = []
    available = set(gdf.columns)

    for field in [cc.CLAYPERC_FIELD, cc.SDR_FINE_FIELD, cc.SDR_COARSE_FIELD,
                  cc.DELIV_RATIO_FIELD, cc.CONV_FACT_FIELD,
                  cc.DWC_FINE_FIELD, cc.DWC_COARSE_FIELD, cc.DWC_FIELD]:
        if field in available:
            value_cols.append(field)

    if use_weighted_averages:
        stats = area_weighted_average(gdf, group_cols, value_cols, area_field)
    else:
        stats = simple_average(gdf, group_cols, value_cols)

    return stats
