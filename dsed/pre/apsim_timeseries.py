"""
APSIM time series import with runoff adjustment.

Reads pre-accumulated CSVs (Catchment$FU$Constituent.csv), applies
calculateIntraMonthlyFlows to distribute monthly loads across days
proportional to daily runoff, and writes adjusted daily CSVs.

Translated from ImportAPSIMTimeSeries_Form.doRunoffAdjusting()
(ImportAPSIMTimeSeries_Form.cs lines 153-397).
"""
import os
import logging
import shutil

import pandas as pd

from . import cropping_const as cc
from .runoff_adjust import calculate_intra_monthly_flows

logger = logging.getLogger(__name__)


def scan_raw_directory(raw_dir, separator=cc.SEPARATOR):
    """Scan a directory of pre-accumulated CSVs and infer catchments, FUs and constituents.

    Filenames are expected to follow the pattern:
        Catchment$FU$Constituent[...].csv
    where the first two $-separated parts are catchment and FU, and the third
    is the constituent (possibly followed by units, phase, etc.).

    Parameters
    ----------
    raw_dir : str
        Directory containing accumulated CSV files.
    separator : str
        Separator used in file naming (default '$').

    Returns
    -------
    catchments : sorted list of str
    fus : sorted list of str
    constituents : sorted list of str
    """
    catchments = set()
    fus = set()
    constituents = set()

    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".csv"):
            continue
        stem = fname[:-4]  # strip .csv
        parts = stem.split(separator)
        if len(parts) < 3:
            continue

        catchments.add(parts[0])
        fus.add(parts[1])

        # The third part is the constituent, but may have trailing units or
        # phase info.  Strip known unit suffixes and whitespace.
        raw_const = parts[2]
        # Remove trailing unit tokens like " TperHa", " KgPerHa", " gPerHa", " mm"
        for suffix in [
            " " + cc.UNITS_T_PER_HA,
            " " + cc.UNITS_KG_PER_HA,
            " " + cc.UNITS_G_PER_HA,
            " " + cc.UNITS_MM,
        ]:
            if raw_const.endswith(suffix):
                raw_const = raw_const[: -len(suffix)]
                break

        constituents.add(raw_const)

    return sorted(catchments), sorted(fus), sorted(constituents)


def find_accumulated_file(raw_dir, catchment, fu, constituent, units=None,
                          phase=None, alt_name=None, separator=cc.SEPARATOR):
    """Try to find a pre-accumulated CSV with various naming fallbacks.

    Naming patterns tried (in order):
    1. Catchment$FU$Constituent.csv
    2. Catchment$FU$Constituent units.csv  (with space)
    3. Catchment$FU$AltName$Phase.csv       (if phase given)
    4. Catchment$FU$AltName$Phase units.csv

    Parameters
    ----------
    raw_dir : str
        Directory containing accumulated files.
    catchment : str
        Catchment name.
    fu : str
        Functional unit name.
    constituent : str
        Constituent name.
    units : str, optional
        Unit string (e.g. 'TperHa', 'KgPerHa', 'gPerHa').
    phase : str, optional
        Phase name (e.g. 'SedimentPhase', 'WaterPhase') for pesticides.
    alt_name : str, optional
        Alternative constituent name to try.
    separator : str
        Separator in file naming.

    Returns
    -------
    str or None
        Full path to found file, or None if not found.
    """
    candidates = []

    base = separator.join([catchment, fu, constituent])
    if phase:
        base_phase = separator.join([catchment, fu, constituent, phase])
    else:
        base_phase = None

    # Without phase
    if not phase:
        candidates.append(base + ".csv")
        if units:
            candidates.append(base + " " + units + ".csv")
    else:
        # With phase
        candidates.append(base_phase + ".csv")
        if units:
            candidates.append(base_phase + " " + units + ".csv")

    # Try alternative name
    if alt_name and alt_name != constituent:
        alt_base = separator.join([catchment, fu, alt_name])
        if not phase:
            candidates.append(alt_base + ".csv")
            if units:
                candidates.append(alt_base + " " + units + ".csv")
        else:
            alt_base_phase = separator.join([catchment, fu, alt_name, phase])
            candidates.append(alt_base_phase + ".csv")
            if units:
                candidates.append(alt_base_phase + " " + units + ".csv")

    for candidate in candidates:
        full_path = os.path.join(raw_dir, candidate)
        if os.path.exists(full_path):
            return full_path

    return None


def _load_timeseries(filepath):
    """Load a single time series CSV with date index."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
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
        return _load_runoff_from_dataframe(runoff_source, catchment, fu)

    # Directory-based lookup
    patterns = [
        f"{catchment}{separator}{fu}.csv",
        f"{catchment}{separator}{fu}{separator}Runoff.csv",
        f'FU$Runoff_mmPerDay{separator}{catchment}{separator}{fu}.csv',
    ]
    for pat in patterns:
        path = os.path.join(runoff_source, pat)
        if os.path.exists(path):
            return _load_timeseries(path)

    logger.warning("Runoff file not found for %s / %s in %s", catchment, fu, runoff_source)
    return None


def _load_runoff_from_dataframe(df, catchment, fu):
    """Extract a runoff series from a DataFrame with MultiIndex columns.

    Supports column MultiIndex levels named ``(catchment, fu)``,
    ``(catchment, hru)``, or positional two-level columns.
    """
    cols = df.columns
    if not isinstance(cols, pd.MultiIndex) or cols.nlevels < 2:
        raise ValueError(
            "Runoff DataFrame must have MultiIndex columns with at least two levels"
        )

    # Try to find the column by (catchment, fu) tuple
    key = (catchment, fu)
    if key in cols:
        return df[key]

    # Try matching by level names if they differ
    level_names = [n.lower() if n else "" for n in cols.names]
    catch_level = None
    fu_level = None
    for i, name in enumerate(level_names):
        if name in ("catchment", "subcatchment"):
            catch_level = i
        elif name in ("fu", "hru", "functional_unit"):
            fu_level = i

    if catch_level is not None and fu_level is not None:
        for col_key in cols:
            if col_key[catch_level] == catchment and col_key[fu_level] == fu:
                return df[col_key]

    logger.warning("Runoff column not found for %s / %s in DataFrame", catchment, fu)
    return None


def import_apsim_timeseries(
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
    """Import pre-accumulated APSIM time series with runoff adjustment.

    For each catchment/FU/constituent combination:
    - Load the raw accumulated CSV
    - Load daily runoff
    - Apply calculate_intra_monthly_flows
    - Write adjusted daily CSVs *or* collect into a single DataFrame

    Any of *constituents*, *catchments* and *fus* may be ``None``, in which
    case they are inferred by scanning the CSV filenames in *raw_dir*.  When
    provided, they act as filters restricting which combinations are processed.

    Translated from ImportAPSIMTimeSeries_Form.doRunoffAdjusting()
    (ImportAPSIMTimeSeries_Form.cs lines 153-397).

    Parameters
    ----------
    raw_dir : str
        Directory containing pre-accumulated monthly CSVs.
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
        constituents = scanned_constituents
        logger.info("Inferred %d constituents from %s", len(constituents), raw_dir)
    else:
        constituents = list(constituents)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Build units map
    units_map = dict(cc.CONSTITUENT_UNITS)
    for c in constituents:
        if c not in units_map:
            units_map[c] = cc.UNITS_G_PER_HA  # pesticide default

    tally = {c: 0 for c in constituents}
    collected = {}  # (catchment, fu, variable) -> Series

    for catchment in catchments:
        for fu in fus:
            runoff_ts = _load_runoff(runoff_source, catchment, fu, separator)
            if runoff_ts is None:
                continue

            for const_name in constituents:
                units = units_map.get(const_name, cc.UNITS_G_PER_HA)

                if const_name == cc.FINE_SED:
                    results = _process_fine_sed(
                        raw_dir, runoff_ts, catchment, fu,
                        const_name, units, separator
                    )
                elif const_name == cc.N_DIN:
                    results = _process_din(
                        raw_dir, runoff_ts, catchment, fu,
                        const_name, units, separator
                    )
                elif const_name == cc.AGGREGATED_RUNOFF:
                    results = _process_runoff_copy(
                        raw_dir, catchment, fu,
                        const_name, units, separator
                    )
                else:
                    results = _process_pesticide(
                        raw_dir, runoff_ts, catchment, fu,
                        const_name, units, separator
                    )

                if results:
                    tally[const_name] += 1
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

    logger.info("APSIM timeseries import complete: %s", tally)

    if output_dir is None:
        return _build_output_dataframe(collected)
    return tally


def _build_output_dataframe(collected):
    """Build a DataFrame from collected results.

    Parameters
    ----------
    collected : dict
        Mapping of ``(catchment, fu, variable)`` to Series.

    Returns
    -------
    DataFrame
        With MultiIndex columns ``(catchment, fu, variable)``.
    """
    if not collected:
        return pd.DataFrame()
    columns = pd.MultiIndex.from_tuples(
        list(collected.keys()), names=["catchment", "fu", "variable"]
    )
    return pd.DataFrame(collected, columns=columns)


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


def _make_column_label(constituent, units):
    """Build a descriptive column label for CSV output, e.g. 'NLeached KgPerHa'."""
    return f"{constituent} {units}"


def _make_variable_name(constituent, suffix, phase=None):
    """Build a variable name for DataFrame column output."""
    parts = [constituent]
    if phase:
        parts.append(phase)
    parts.append(suffix)
    return "_".join(parts)


def _process_fine_sed(raw_dir, runoff_ts, catchment, fu,
                      const_name, units, separator):
    """Process Fine Sediment: load raw, adjust.

    Returns list of entry dicts, or empty list.
    """
    raw_path = find_accumulated_file(raw_dir, catchment, fu, const_name, units,
                                     separator=separator)
    if raw_path is None:
        logger.debug("No Fine Sed file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(raw_path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    return [{"constituent": const_name, "suffix": cc.TS_DAILY_ADJUSTED,
             "series": adjusted}]


def _process_din(raw_dir, runoff_ts, catchment, fu,
                 const_name, units, separator):
    """Process DIN: load raw DIN + NLeached, adjust both.

    Returns list of entry dicts, or empty list.
    """
    raw_path = find_accumulated_file(raw_dir, catchment, fu, const_name, units,
                                     separator=separator)
    if raw_path is None:
        logger.debug("No DIN file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(raw_path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    results = [{"constituent": const_name, "suffix": cc.TS_DAILY_ADJUSTED,
                "series": adjusted}]

    # NLeached
    leach_units = cc.UNITS_KG_PER_HA
    leach_path = find_accumulated_file(raw_dir, catchment, fu, cc.APSIM_NLEACHED_FIELD,
                                       leach_units, separator=separator)
    if leach_path is None:
        leach_path = find_accumulated_file(raw_dir, catchment, fu, cc.APSIM_NLEACHED_FIELD,
                                           separator=separator)
    if leach_path is not None:
        leach_ts = _load_timeseries(leach_path)
        leach_monthly = leach_ts.resample("MS").sum()
        leach_adjusted = calculate_intra_monthly_flows(runoff_ts, leach_monthly)

        results.append({"constituent": cc.APSIM_NLEACHED_FIELD,
                        "suffix": cc.TS_DAILY_ADJUSTED,
                        "units": leach_units, "series": leach_adjusted})

    return results


def _process_runoff_copy(raw_dir, catchment, fu,
                         const_name, units, separator):
    """Process Aggregated Runoff: load as-is (no adjustment).

    Returns list of entry dicts, or empty list.
    """
    raw_path = find_accumulated_file(raw_dir, catchment, fu, const_name, units,
                                     separator=separator)
    if raw_path is None:
        logger.debug("No runoff file for %s / %s", catchment, fu)
        return []

    raw_ts = _load_timeseries(raw_path)
    return [{"constituent": const_name, "suffix": cc.TS_DAILY_NOT_ADJUSTED,
             "series": raw_ts}]


def _process_pesticide(raw_dir, runoff_ts, catchment, fu,
                       const_name, units, separator):
    """Process pesticide: load sed+water phase, adjust both.

    Returns list of entry dicts, or empty list.
    """
    alt_name = cc.PESTICIDE_ALT_NAMES.get(const_name)

    # Sediment phase
    sed_path = find_accumulated_file(
        raw_dir, catchment, fu, const_name, units,
        phase=cc.SED_PHASE, alt_name=alt_name, separator=separator
    )
    if sed_path is None:
        logger.debug("No pesticide sed phase for %s / %s / %s", catchment, fu, const_name)
        return []

    raw_ts = _load_timeseries(sed_path)
    monthly = raw_ts.resample("MS").sum()
    adjusted = calculate_intra_monthly_flows(runoff_ts, monthly)

    results = [{"constituent": const_name, "suffix": cc.TS_DAILY_ADJUSTED,
                "phase": cc.SED_PHASE, "series": adjusted}]

    # Water phase - derive path from sed phase path
    water_path = sed_path.replace(cc.SED_PHASE, cc.WATER_PHASE)
    if not os.path.exists(water_path):
        water_path = sed_path.replace("Sediment", "Water")

    if os.path.exists(water_path):
        raw_ts_w = _load_timeseries(water_path)
        monthly_w = raw_ts_w.resample("MS").sum()
        adjusted_w = calculate_intra_monthly_flows(runoff_ts, monthly_w)

        results.append({"constituent": const_name, "suffix": cc.TS_DAILY_ADJUSTED,
                        "phase": cc.WATER_PHASE, "series": adjusted_w})

    return results
