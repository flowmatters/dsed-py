"""
Area-weighted and simple averaging from shapefile data.

Replaces the GIS row loops in both APSIM (APSIMparameterisationModel.cs
lines 1219-1506) and HowLeaky (HowLeakyParameterisationModel.cs lines 304-384)
models, and the weighted/unweighted stats methods in ImportHowLeakyTimeSeriesModel.cs
(lines 654-881).
"""
import logging

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


def load_shapefile_data(source):
    """Load shapefile data as a GeoDataFrame.

    Parameters
    ----------
    source : str or GeoDataFrame
        Path to shapefile or an existing GeoDataFrame.

    Returns
    -------
    GeoDataFrame
    """
    if isinstance(source, gpd.GeoDataFrame):
        return source
    return gpd.read_file(source)


def area_weighted_average(gdf, group_cols, value_cols, area_col):
    """Compute area-weighted averages of value columns grouped by group columns.

    Translated from getWeightedStatsDictionary (ImportHowLeakyTimeSeriesModel.cs
    lines 805-847) and the GIS row loops in APSIMparameterisationModel.cs.

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Input data with area and value columns.
    group_cols : str or list of str
        Column(s) to group by (e.g. subcatchment, FU).
    value_cols : str or list of str
        Column(s) to compute weighted averages for.
    area_col : str
        Column containing polygon areas.

    Returns
    -------
    DataFrame
        Weighted averages indexed by group columns, with value columns plus
        a 'total_area' column.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    df = pd.DataFrame(gdf)

    def _weighted_avg(group):
        areas = group[area_col]
        total_area = areas.sum()
        result = {}
        for col in value_cols:
            result[col] = (group[col] * areas).sum() / total_area if total_area > 0 else 0.0
        result["total_area"] = total_area
        return pd.Series(result)

    return df.groupby(group_cols).apply(_weighted_avg, include_groups=False)


def simple_average(gdf, group_cols, value_cols):
    """Compute simple (unweighted) averages of value columns.

    Translated from getUnWeightedStatsDictionary (ImportHowLeakyTimeSeriesModel.cs
    lines 849-881).

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Input data.
    group_cols : str or list of str
        Column(s) to group by.
    value_cols : str or list of str
        Column(s) to compute means for.

    Returns
    -------
    DataFrame
        Simple averages indexed by group columns.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    df = pd.DataFrame(gdf)
    return df.groupby(group_cols)[value_cols].mean()


def dominant_category(gdf, group_cols, category_col, area_col):
    """Find the dominant category (largest total area) per group.

    Translated from getDomSoilName / getdomSoil in APSIMparameterisationModel.cs
    (lines 1471-1487, 1508-1509).

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Input data.
    group_cols : str or list of str
        Column(s) to group by.
    category_col : str
        Column containing category labels.
    area_col : str
        Column containing polygon areas.

    Returns
    -------
    Series
        Dominant category per group, indexed by group columns.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    df = pd.DataFrame(gdf)
    area_by_cat = df.groupby(group_cols + [category_col])[area_col].sum()
    return area_by_cat.groupby(level=group_cols).idxmax().apply(lambda x: x[-1])
