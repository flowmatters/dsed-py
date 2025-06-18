import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from glob import glob
import os
import shutil
import rasterio
from geocube.api.core import make_geocube
import logging
logger = logging.getLogger(__name__)


def apply_projection(grids,prj_file=None):
    prj_filenames = [os.path.splitext(fn)[0] + '.prj' for fn in grids]
    if prj_file is None:
        for p in prj_filenames:
            if os.path.exists(p):
                prj_file = p
                logger.info(f'Using first available example prj file: {p}')
                break
    assert prj_file is not None and os.path.exists(prj_file)
    for prj_filename in prj_filenames:
        if not os.path.exists(prj_filename):
            logger.info(f'Copying projection to {prj_filename}')
            shutil.copyfile(prj_file,prj_filename)

def apply_timestamp_from_filename(ds):
    fn = os.path.splitext(os.path.basename(ds.encoding['source']))[0]
    y = int(fn[2:6])
    m = int(fn[6:])
    ds['time'] = pd.Timestamp(y,m,1)
    return ds

def load_cfactor_ascii_grids(directory_or_file_pattern):
    '''
    Load C-factor ascii grids from a directory or a file pattern.
    If a directory is given, it will look for files with the pattern '*.asc'.
    If a file pattern is given, it will use that directly.
    The files are expected to be in the same projection and at least one should have a .prj file

    Returns an xarray Dataset with a time dimension.
    '''
    if '*' in directory_or_file_pattern:
        file_pattern = directory_or_file_pattern
    else:
        file_pattern = os.path.join(directory_or_file_pattern,'*.asc')
    grids = list(glob(file_pattern))
    apply_projection(grids)
    dataset = xr.open_mfdataset(grids,preprocess=apply_timestamp_from_filename,concat_dim='time',combine='nested',parallel=True)
    return dataset

def convert_cfactor_ascii_grids(file_pattern,dest_fn):
    '''
    Convert C-factor ascii grids to a NetCDF file.

    See `load_cfactor_ascii_grids` for the expected input format.
    The output is a NetCDF file with a time dimension.
    '''
    dataset = load_cfactor_ascii_grids(file_pattern)
    dataset.to_netcdf(dest_fn)

def load_cfactor_nc(fn):
    '''
    Load C-factor NetCDF file, and reinstate the CRS from the spatial_ref attribute.
    '''
    dataset = xr.open_dataset(fn)
    data = dataset.band_data
    data.rio.set_crs(dataset.spatial_ref.attrs['crs_wkt'])
    return data

def load_matching(fn,match):
    '''
    Load a raster file and reproject it to match the provided raster.
    '''
    data = rxr.open_rasterio(fn,masked=True)
    matched = data.rio.reproject_match(match)
    return matched

def filter_areas(areas,fus=None,catchments=None):
    if fus is not None:
        areas = areas[areas.IntFUs.isin(fus)]
    if catchments is not None:
        areas = areas[areas.IntSCs.isin(catchments)]
    return areas

def load_areas(fn,match=None,fus=None,catchments=None):
    '''
    Load areas (intersected SC/FU) from a shapefile, filtering by FUs and/or catchments if specified and rasterising the result.

    Rasterisation is done to match the CRS and resolution of the provided `match` raster.

    Returns a GeoDataFrame of areas and an xarray DataArray of the rasterised areas.
    '''
    areas = gpd.read_file(fn)
    if fus is not None or catchments is not None:
        areas = filter_areas(areas,fus,catchments)
    if areas.crs is None:
        assert match.rio.crs is not None
        areas.set_crs(match.rio.crs,inplace=True)
    areas_raster = make_geocube(areas,like=match)    
    return areas, areas_raster.IntSCFU

def make_zone_names(areas):
    zone_names = areas.reset_index()
    zone_names['name'] = zone_names['IntSCs'] + '$' + zone_names['IntFUs']
    return zone_names['name'].to_dict()

def compute_zonal_timeseries(data,zones,names,suffix=''):
    '''
    Compute zonal timeseries from a DataArray (x,y,t), using the provided zones (x,y),
    taking the time dimension from data and the names from the provided names dictionary.
    '''
    grouped = data.groupby(zones)
    timeseries = grouped.mean()
    timeseries = timeseries.sel(band=1)
    result = pd.DataFrame(timeseries)
    result.index = [pd.Timestamp(d) for d in data.time.values]
    
    result = result.rename(columns=lambda c:names[c]+suffix)
    return result

def compare(computed,path):
    pc_error = {}
    for col in computed:
        fn = os.path.join(path,f'{col}.csv')
        if not os.path.exists(fn):
            print(f'No source version: {fn}')
            continue
        source_version = pd.read_csv(fn,index_col=0,parse_dates=True,dayfirst=True)
        source_version = source_version[source_version.index.isin(computed.index)]
        source_version = source_version[source_version.columns[0]]
        pc_error[col] = 100.0 * (source_version - computed[col]) / source_version
    pc_error = pd.DataFrame(pc_error)
    return pc_error

def static_stats(areas,names,**grids):
    rows = []
    for label,grid in grids.items():
        if isinstance(grid,float):
            rows += [name.split('$')+[label,grid] for name in names.values()]
            continue
        stats = grid.groupby(areas).mean()
        stats = stats.sel(band=1)
        rows += [names[ix].split('$')+[label,v] for ix,v in enumerate(stats.values)]
    result = pd.DataFrame(rows,columns=['Catchment','FU','StatsName','Value'])
    result = result.sort_values(['Catchment','FU','StatsName']).reset_index(drop=True)
    return result

def fill_seasonal(df,period):
    means = df.mean()
    df = df.reindex(period,method='ffill',limit=2)
    df = df.fillna(means)
    return df

def standard_usle(cfactor_fn, kls_fn, scald_fn, fines_fn, areas_shp_fn,fus=None,time_period=None,start=None,end=None):
    '''
    Standard USLE preprocessing for Dynamic Sednet

    Computes spatiallly averaged KLSC for each subcatchment/fu (optionally limiting to a set of FUs)

    Fills the time series with long term means if a longer time period is specified.
    '''
    if os.path.isdir(cfactor_fn) or '*' in cfactor_fn:
        logger.info(f'Converting cfactor from ascii grids to NetCDF: {cfactor_fn}')
        cfactor_nc = os.path.join(cfactor_fn,'cfactor.nc')
        convert_cfactor_ascii_grids(cfactor_fn,cfactor_nc)
        cfactor_fn = cfactor_nc
    cfactor = load_cfactor_nc(cfactor_fn)
    kls = load_matching(kls_fn,cfactor)
    scald = load_matching(scald_fn,cfactor)
    fines = load_matching(fines_fn,cfactor)
    klsc = kls * cfactor
    klsc_fines = 1e-2 * fines * klsc
    areas,areas_raster = load_areas(areas_shp_fn,kls,fus=fus)
    zone_names = make_zone_names(areas)

    klsc_timeseries = compute_zonal_timeseries(klsc,areas_raster,zone_names,'$USLE_KLSC_Total')
    cfactor_timeseries = compute_zonal_timeseries(cfactor,areas_raster,zone_names,'$CFactor')
    klsc_fines_timeseries = compute_zonal_timeseries(klsc_fines,areas_raster,zone_names,'$USLE_KLSC_FinePerc')
    if start is not None and end is not None:
        time_period = pd.date_range(start,end,freq='MS')
    if time_period is not None:
        klsc_timeseries = fill_seasonal(klsc_timeseries,time_period)
        klsc_fines_timeseries = fill_seasonal(klsc_fines_timeseries, time_period)
        cfactor_timeseries = fill_seasonal(cfactor_timeseries, time_period)

    stats = static_stats(areas_raster,zone_names,Mean_Fines=fines,Mean_KLS=kls,Means_Scald=scald,Mean_LS=1.0)

    return klsc_timeseries, klsc_fines_timeseries, cfactor_timeseries, stats

def write_source_inputs(dest,klsc_timeseries, klsc_fines_timeseries, cfactor_timeseries, stats):
    '''
    Write out the results of `standard_usle` in the format expected by the Source implementation of Dynamic Sednet.
    '''
    os.makedirs(dest,exist_ok=True)
    sets = [
        (cfactor_timeseries,'Cfact','C-Factor'),
        (klsc_timeseries,'KLSC','KLSC_Total'),
        (klsc_fines_timeseries,'KLSC_Fines','KLSC_Fines')
    ]
    for df, folder, column_name in sets:
        var_dest = os.path.join(dest,folder)
        os.makedirs(var_dest,exist_ok=True)
        for col in df.columns:
            series = df[col].copy()
            series.name = column_name
            series.to_csv(os.path.join(var_dest,f'{col}.csv'),date_format='%d/%m/%Y')

