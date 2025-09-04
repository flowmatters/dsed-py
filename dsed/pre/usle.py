from deprecated import deprecated
import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from glob import glob
import dask
import os
import math
import shutil
import gc
import rasterio
from geocube.api.core import make_geocube
import logging
logger = logging.getLogger(__name__)

CELL_SIZE=30
GB=1024*1024*1024

def choose_number_of_parallel_workers(example_cover_file=None,cores=None,ram_gb=None,n_cells=None):
    '''
    Attempt to determine a reasonable number of parallel workers to use for processing cover data.

    The cover data processing is easily parallelised into independent jobs, but individual jobs
    can be memory intensive, limiting the number of parallel jobs that can be run at once.

    This function function recommends a number of parallel jobs based on the available RAM and the
    size of a single cover grid. The function assumes that the compute is otherwise idle and that
    the user is not running other CPU or memory-intensive processes at the same time. You may wish
    to reduce the number of parallel jobs if you will be running other processes at the same time.
    '''
    FLOAT_SIZE=4
    RASTER_MULTIPLIER=6

    if cores is None:
        cores = os.cpu_count()
        logger.info('Using system report CPU count (cores/threads): %d',cores)
    if ram_gb is None:
        try:
            import psutil
            ram = psutil.virtual_memory()
            ram_gb = int(ram.total/GB)
            logger.info('Assuming total available RAM: %dGb',ram_gb)
        except:
            ram_gb = 2 * cores
            logger.info('Cannot detect RAM. Assuming system has 2Gb/core = %dGb',ram_gb)
    if n_cells is None:
        if example_cover_file is None:
            n_cells = 320 * 1e6 # Largest cover grid in the to date is 320 million cells
            logger.info('No example cfactor provided. Assuming large. n_cells=%d',n_cells)
        else:
            raster = rasterio.open(example_cover_file)
            n_cells = raster.width * raster.height
            logger.info('Using example cover file %s with n_cells=%d',example_cover_file,n_cells)

    raster_size=n_cells*FLOAT_SIZE/GB
    ram_per_job = math.ceil(RASTER_MULTIPLIER*raster_size)
    max_jobs = int(ram_gb/ram_per_job)
    logger.info('Sufficient RAM for %d jobs',max_jobs)
    if max_jobs > cores:
        max_jobs = cores
        logger.info('Constraining to number cores: %d',max_jobs)
    return max_jobs


def apply_projection_to_all_grids(grids,prj_file=None):
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
    '''
    Extract timestamp from the filename of the dataset and add it as a time coordinate.
    The filename is expected to be in the format '<prefix>YYYYMM<suffix>.<extension>'.
    For example, 'cfactor_20220101.asc' will extract the timestamp as
    2022-01-01 and add it to the dataset as a time coordinate.
    If the filename does not contain a valid timestamp, it will raise a ValueError.
    '''
    fn = os.path.splitext(os.path.basename(ds.encoding['source']))[0]
    while len(fn) and not fn[0].isdigit():
        fn = fn[1:]
    if len(fn) < 6:
        raise ValueError(f'Filename {ds.encoding["source"]} does not contain a valid timestamp in the expected format (e.g. "cfactor_20220101.asc")')
    y = int(fn[:4])
    m = int(fn[4:6])
    ds['time'] = pd.Timestamp(y,m,1)
    return ds

def match_grid_files(directory_or_file_pattern):
    if '*' in directory_or_file_pattern:
        file_pattern = directory_or_file_pattern
    else:
        for ext in ['*.asc', '*.tif']:
            file_pattern = os.path.join(directory_or_file_pattern, ext)
            grids = list(glob(file_pattern))
            if grids:
                break
    return list(glob(file_pattern))

def load_cfactor_grids(directory_or_file_pattern):
    '''
    Load C-factor ascii grids from a directory or a file pattern.
    If a directory is given, it will look for files with the pattern '*.asc' or '*.tif'.
    If a file pattern is given, it will use that directly.
    The files are expected to be in the same projection and at least one should have a .prj file

    Returns an xarray Dataset with a time dimension.
    '''
    if isinstance(directory_or_file_pattern, list):
        grids = directory_or_file_pattern
    else:
        grids = match_grid_files(directory_or_file_pattern)
    if grids[0].endswith('.asc'):
        apply_projection_to_all_grids(grids)
    dataset = xr.open_mfdataset(grids,preprocess=apply_timestamp_from_filename,concat_dim='time',combine='nested',parallel=False)
    return dataset

def convert_cfactor_grids(file_pattern,dest_fn):
    '''
    Convert C-factor ascii grids to a NetCDF file.

    See `load_cfactor_ascii_grids` for the expected input format.
    The output is a NetCDF file with a time dimension.
    '''
    dataset = load_cfactor_grids(file_pattern)
    dataset = dataset.rename({'band_data':'cfactor'})
    dataset.to_netcdf(dest_fn,encoding={'cfactor':{'chunksizes':(1,1,100,100)}})

def apply_crs_to_cfactor(dataset):
    if 'cfactor' in dataset:
        # If the dataset has a 'cfactor' variable, we assume it's the main variable
        data = dataset['cfactor']
    else:
        # Otherwise, we assume the dataset is the main variable
        data = dataset['band_data']
    data.rio.write_crs(dataset.spatial_ref.attrs['crs_wkt'],inplace=True)
    return data

def load_cfactor_nc(fn,**kwargs):
    '''
    Load C-factor NetCDF file, and reinstate the CRS from the spatial_ref attribute.
    '''
    dataset = xr.open_dataset(fn,**kwargs)
    return apply_crs_to_cfactor(dataset)

def create_cfactor_nc_per_year(src_dir, dest_dir,replace_existing=False):
    '''
    Create a NetCDF file for each year in the source directory, containing the all C-factor grids for that year.
    The source directory is expected to contain C-factor files with a timestamp in the filename.
    The destination directory will contain the created NetCDF files, which are chunked for efficient processing.
    '''
    cfactor_files = match_grid_files(src_dir)
    def get_ym(fn):
        fn = os.path.splitext(os.path.basename(fn))[0]
        ym = fn.split('_')[-1][:6]
        return ym
    ym = [get_ym(fn) for fn in cfactor_files]
    df = pd.DataFrame([ym,cfactor_files]).transpose().rename(columns={0:'ym',1:'fn'}).sort_values('ym')
    df['y'] = df['ym'].str.slice(0,4).astype('int')
    df['m'] = df['ym'].str.slice(4).astype('int')

    cpu_count = dask.system.cpu_count()

    jobs = []
    convert_cfactor_grids_ = dask.delayed(convert_cfactor_grids)
    os.makedirs(dest_dir, exist_ok=True)
    if not len(df):
        logger.warning('No C-factor grids found in the source directory. Check the file pattern or directory.')
        return []
    filenames = []
    for y, data in df.groupby('y'):
        files = list(data.sort_values('m')['fn'].values)
        dest_fn = os.path.join(dest_dir,f'cfactor-{y}.nc')
        filenames.append(dest_fn)
        if os.path.exists(dest_fn) and not replace_existing:
            logger.info(f'Skipping existing file {dest_fn}')
            continue
        res = convert_cfactor_grids_(files,dest_fn)
        jobs.append(res)

    if len(jobs):
        logger.info(f'Running %d jobs across %d processes to convert C-factor grids to NetCDF', len(jobs), cpu_count)
        _ = dask.compute(*jobs,scheduler='processes', num_workers=cpu_count)
    else:
        logger.info('No missing C-factor grids to convert.')

    return filenames

def load_matching(fn,match):
    '''
    Load a raster file and reproject it to match the provided raster.
    '''
    data = rxr.open_rasterio(fn,masked=True)
    matched = data.rio.reproject_match(match).sel(band=1)
    return matched

def cluster_polygons_by_centroid(df,n_clusters):
    '''
    Cluster polygons in a GeoDataFrame by their centroid coordinates using KMeans clustering.

    Used for creating batches of polygons that are geographically close to each other, such
    that each batch can be processed in parallel without excessive memory usage.
    '''
    from sklearn.cluster import KMeans
    df['cx'] = df.geometry.centroid.x
    df['cy'] = df.geometry.centroid.y
    X = df[['cx', 'cy']].values
    kmeans = KMeans(n_clusters=n_clusters,random_state=0,n_init='auto')
    df['cluster_label'] = kmeans.fit_predict(X)
    return [batch for _,batch in df.groupby('cluster_label')]

def filter_geodataframe(gdf,**kwargs):
    '''
    Filter a GeoDataFrame by the provided keyword arguments.
    The keyword arguments are expected to be column names and values to filter by.
    For example, `filter_geodataframe(gdf, IntFUs='FU1', IntSCs='SC1')` will filter the GeoDataFrame
    to only include rows where IntFUs is 'FU1' and IntSCs is 'SC1'.
    '''
    null_geometry = gdf[pd.isnull(gdf.bounds.minx)]
    if len(null_geometry) > 0:
        logger.warning(f'Found {len(null_geometry)} geometries with null bounds, dropping them.')
        logger.debug(f'Null geometries: {null_geometry}')
        gdf = gdf.drop(null_geometry.index)

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, list):
            gdf = gdf[gdf[key].isin(value)]
        else:
            gdf = gdf[gdf[key] == value]
    return gdf

def filter_areas(areas,fus=None,catchments=None):
    return filter_geodataframe(areas, IntFUs=fus, IntSCs=catchments)

def load_filter_rasterise(fn,match=None,**kwargs):
    '''
    Load a vector file, filter it by the provided keyword arguments, and generate rasterised versions
    to match the provided raster.

    The keyword arguments are passed to the `filter_geodataframe` function to filter the areas.
    If no match is provided, the CRS of the loaded data is used.
    '''
    areas = gpd.read_file(fn)
    areas = filter_geodataframe(areas, **kwargs)
    if areas.crs is None or \
       fn.endswith('.json'): # JSON files may not have a CRS defined and are assumed (by Geopandas) to be WGS84
        assert match.rio.crs is not None
        areas.set_crs(match.rio.crs,inplace=True,allow_override=True)
    assert len(areas) > 0, f'No areas found in {fn} with the specified filters: {kwargs}'
    areas_raster = make_geocube(areas,like=match)
    return areas, areas_raster

def load_areas(fn):
    areas = gpd.read_file(fn)
    if ('IntSCFU' not in areas.columns) or areas['IntSCFU'].dtype != int:
        areas['IntSCFU'] = areas.index.astype(int)
    return areas

def load_areas_matched(fn,match=None,fus=None,catchments=None):
    '''
    Load areas (intersected SC/FU) from a shapefile, filtering by FUs and/or catchments if specified and rasterising the result.

    Rasterisation is done to match the CRS and resolution of the provided `match` raster.

    Returns a GeoDataFrame of areas and an xarray DataArray of the rasterised areas.
    '''
    areas = load_areas(fn)
    areas = filter_geodataframe(areas, IntFUs=fus,IntSCs=catchments)
    if areas.crs is None or \
       fn.endswith('.json'): # JSON files may not have a CRS defined and are assumed (by Geopandas) to be WGS84
        logger.info(f'No CRS defined for {fn}, using CRS from match raster')
        assert match.rio.crs is not None
        areas.set_crs(match.rio.crs,inplace=True,allow_override=True)
    assert len(areas) > 0, f'No areas found in {fn} with the specified filters: {fus}, {catchments}'
    rasters = make_geocube(areas,like=match)
    # Ensure the areas are filtered
    return areas, rasters.IntSCFU

def make_zone_names(areas):
    zone_names = areas.reset_index()
    zone_names['name'] = zone_names['IntSCs'] + '$' + zone_names['IntFUs']
    return zone_names['name'].to_dict()

def load_change(fn,match=None,fus=None,catchments=None,a='NonLinearA',b='NonLinearB'):
    '''
    Load change (intersected SC/FU) from a shapefile, filtering by FUs and/or catchments if specified and rasterising the result.

    Rasterisation is done to match the CRS and resolution of the provided `match` raster.

    Returns a GeoDataFrame of areas and an xarray DataArray of the rasterised areas.
    '''
    areas, rasters = load_filter_rasterise(fn,match=match,IntFUs=fus,IntSCs=catchments)
    rasters[a] = rasters[a].fillna(1.0)
    rasters[b] = rasters[b].fillna(1.0)
    return areas, rasters[a], rasters[b]

@deprecated
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

def compare_to_source(computed,path):
    '''
    Compare zonal timeseries computed using this library to values computed by Source plugin.

    Parameters:
    - computed: DataFrame with computed values, indexed by time and with columns for each catchment/FU combination.
    - path: Path to the directory containing the source CSV files with individual files with naming convention '<catchment>$<FU>$<variable>.csv'.

    Returns:
    - DataFrame with timeseries of percentage errors for each catchment/FU combination.
    '''
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

@deprecated
def standard_usle(cfactor_or_cfactor_fn, kls_fn, scald_fn, fines_fn, areas_shp_fn,fus=None,time_period=None,start=None,end=None):
    '''
    Standard USLE preprocessing for Dynamic Sednet

    Computes spatiallly averaged KLSC for each subcatchment/fu (optionally limiting to a set of FUs)

    Fills the time series with long term means if a longer time period is specified.
    '''
    # global areas, areas_raster, zone_names, kls, cfactor, scald, fines
    if isinstance(cfactor_or_cfactor_fn, str):
        cfactor_fn = cfactor_or_cfactor_fn
        if os.path.isdir(cfactor_fn) or '*' in cfactor_fn:
            logger.info(f'Converting cfactor from ascii grids to NetCDF: {cfactor_fn}')
            cfactor_nc = os.path.join(cfactor_fn,'cfactor.nc')
            convert_cfactor_grids(cfactor_fn,cfactor_nc)
            cfactor_fn = cfactor_nc
        cfactor = load_cfactor_nc(cfactor_fn)
    else:
        logger.info(f'Using provided C-factor dataset')
        cfactor = cfactor_or_cfactor_fn
    kls = load_matching(kls_fn,cfactor)
    if scald_fn is None:
        scald = 1.0
    else:
        scald = load_matching(scald_fn,cfactor)
    fines = load_matching(fines_fn,cfactor)
    logger.info('Grids loaded, computing KLSC')
    klsc = kls * cfactor
    klsc_fines = 1e-2 * fines * klsc
    logger.info('Loading and rasterising SC/FU areas')
    areas,areas_raster = load_areas(areas_shp_fn,kls,fus=fus)
    zone_names = make_zone_names(areas)

    logger.info('Computing zonal timeseries for klsc')
    klsc_timeseries = compute_zonal_timeseries(klsc,areas_raster,zone_names,'$USLE_KLSC_Total')
    logger.info('Computing zonal timeseries for cfactor')
    cfactor_timeseries = compute_zonal_timeseries(cfactor,areas_raster,zone_names,'$CFactor')
    logger.info('Computing zonal timeseries for klsc fines')
    klsc_fines_timeseries = compute_zonal_timeseries(klsc_fines,areas_raster,zone_names,'$USLE_KLSC_FinePerc')
    if start is not None and end is not None:
        time_period = pd.date_range(start,end,freq='MS')
    if time_period is not None:
        logger.info(f'Filling timeseries with long term means for period {time_period[0]} to {time_period[-1]}')
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

def load_change_scenario(fn,match=None,fus=None,catchments=None,a='NonLinearA',b='NonLinearB'):
    '''
    Load change (intersected SC/FU) from a shapefile, filtering by FUs and/or catchments if specified and rasterising the result.

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
    areas_raster[a] = areas_raster[a].fillna(1.0)
    areas_raster[b] = areas_raster[b].fillna(1.0)
    return areas, areas_raster[a], areas_raster[b]


def compute_usle_zonal_statistics(cfactor_files,
                                  kls_fn,scald_fn,fines_fn,
                                  areas_shp_fn,
                                  dest_prefix=None,
                                  change_map=None,
                                  buffer=CELL_SIZE,
                                  log_level=logging.DEBUG):
    '''
    USLE preprocessing for Dynamic Sednet

    Computes spatiallly averaged cfactor and KLSC for each subcatchment/fu (optionally limiting to a set of FUs)

    Can be called directly, but more often called by `parallel_usle_zonal_statistics`.

    If `dest_prefix` is provided, the results are written to CSV files with the prefix `dest_prefix`.
    If `dest_prefix` is None, the results are returned as a tuple
    (failed_regions, cfactor_timeseries, klsc_timeseries, klsc_fines_timeseries).
    If `change_map` is provided, it is used to apply a change scenario
    to the cfactor before computing the KLSC.
    The change scenario is expected to be a shapefile with two attributes: NonLinearA and NonLinearB.
    These attributes are used to apply a non-linear change to the cfactor:
    cfactor = max(cfactor, change_A * cfactor ** change_B)
    If no change map is provided, the cfactor is not modified.

    Returns:
    - A list of failed regions (if any)
    - A DataFrame of cfactor timeseries (if `dest_prefix` is None)
    - A DataFrame of KLSC timeseries (if `dest_prefix` is None)
    - A DataFrame of KLSC fines timeseries (if `dest_prefix` is None)
    '''
    gc.collect()
    klsc_timeseries_all = []
    cfactor_timeseries_all = []
    klsc_fines_timeseries_all = []
    cfactor_eg = cfactor_files[0]
    if cfactor_eg.endswith('.nc'):
        cfactor_eg = load_cfactor_nc(cfactor_eg)
    else:
        cfactor_eg = load_cfactor_grids(cfactor_eg)
    # cfactor = apply_crs_to_cfactor(cfactor)
    logging.log(log_level,'cfactor full: %s, %s',str(cfactor_eg.shape),str(cfactor_eg.rio.crs))
    crs = cfactor_eg.rio.crs
    logging.log(log_level,'Loading SC/FU areas')
    areas = gpd.read_file(areas_shp_fn)
    areas.set_crs(cfactor_eg.rio.crs,inplace=True,allow_override=True)
    zone_names = make_zone_names(areas)

    minx, miny, maxx, maxy = areas.total_bounds
    minx = minx-buffer
    maxx = maxx+buffer
    miny = miny-buffer
    maxy = maxy+buffer
    x_range = slice(minx,maxx)
    y_range = slice(maxy,miny)
    cfactor_eg = cfactor_eg.sel(x=x_range,y=y_range)
    kls = load_matching(kls_fn,cfactor_eg)
    if scald_fn is None:
        scald = 1.0
    else:
        scald = load_matching(scald_fn,cfactor_eg)
    fines = load_matching(fines_fn,cfactor_eg)

    if change_map is None:
        logger.log(log_level,'No change map')
        change_A=1.0
        change_B=1.0
    else:
        logger.log(log_level,'Using change scenario from %s',change_map)
        _, change_A, change_B = load_change_scenario(change_map,cfactor_eg)

    cfactor = cfactor_eg
    cfactor_eg = None
    failed_regions = set()
    region_rasters = {}
    for ix,cfactor_fn in enumerate(cfactor_files):
        gc.collect()
        logger.log(log_level,'Processing cfactor: %s (%d/%d)',cfactor_fn,ix+1,len(cfactor_files))
        klsc_timeseries = {}
        cfactor_timeseries = {}
        klsc_fines_timeseries = {}

        if ix > 0:
            cfactor = load_cfactor_nc(cfactor_fn)
            cfactor = cfactor.sel(x=x_range,y=y_range)
            cfactor.rio.write_crs(crs,inplace=True)

        if 'band' in cfactor.dims:
            cfactor = cfactor.sel(band=1)

        if change_map is not None:
            cfactor = xr.ufuncs.minimum(cfactor,change_A* cfactor**change_B)

        index = [pd.Timestamp(d) for d in cfactor.time.values]
        for ix in range(len(areas)):
            # gc.collect()
            subset = areas.iloc[ix:(ix+1)]
            zone_name= zone_names[ix]
            bounds = subset.bounds.iloc[0]
            minx = bounds.minx-buffer
            maxx = bounds.maxx+buffer
            miny = bounds.miny-buffer
            maxy = bounds.maxy+buffer

            _cfactor = cfactor.sel(x=slice(minx,maxx),y=slice(maxy,miny))
            _cfactor.rio.write_crs(crs,inplace=True)

            if zone_name not in region_rasters:
                region_rasters[zone_name] = make_geocube(subset,like=_cfactor).IntSCFU
            area_raster = region_rasters[zone_name]

            def get_zonal(data):
                if zone_name in failed_regions:
                    # If we have already failed for this zone, return NaN
                    result = pd.DataFrame([np.nan]*len(data.time),columns=[zone_name],
                                          index=index)
                    return result

                try:
                    zonal_ts = data.groupby(area_raster).mean()
                    result = pd.DataFrame(zonal_ts,index=index)
                    result = result.rename(columns={0:zone_name})
                except Exception as e:
                    result = pd.DataFrame([np.nan]*len(data.time),columns=[zone_name],
                                          index=index)
                    failed_regions.add(zone_name)
                return result

            cfactor_timeseries[zone_name] = get_zonal(_cfactor)

            _kls = kls.sel(x=slice(minx,maxx),y=slice(maxy,miny))
            klsc = _kls * _cfactor
            klsc_timeseries[zone_name] = get_zonal(klsc)

            _fines = fines.sel(x=slice(minx,maxx),y=slice(maxy,miny))
            klsc_fines = 1e-2 * _fines * klsc
            klsc_fines_timeseries[zone_name] = get_zonal(klsc_fines)

        def tidy_results(coll):
            return pd.concat(coll,axis=1).droplevel(0,axis=1)

        klsc_timeseries_all.append(tidy_results(klsc_timeseries))
        klsc_fines_timeseries_all.append(tidy_results(klsc_fines_timeseries))
        cfactor_timeseries_all.append(tidy_results(cfactor_timeseries))#, stats

    klsc_final = pd.concat(klsc_timeseries_all)
    klsc_fines_final = pd.concat(klsc_fines_timeseries_all)
    cfactor_final = pd.concat(cfactor_timeseries_all)
    if dest_prefix is None:
        return list(failed_regions), cfactor_final, klsc_final, klsc_fines_final

    def write(df,label):
        df.sort_index().to_csv(f'{dest_prefix}-{label}.csv')
    write(klsc_final,'klsc')
    write(klsc_fines_final,'klsc-fines')
    write(cfactor_final,'cfactor')
    return list(failed_regions)

def parallel_usle_zonal_statistics(cfactor_src,
                                   kls_fn,scald_fn,fines_fn,
                                   areas,
                                   change_map_fn=None,
                                   buffer=CELL_SIZE,
                                   num_workers=None,
                                   keep_intermediate_files=False,
                                   batch_directory=None,
                                   job_results_directory=None,
                                   time_period=None,
                                   start=None,
                                   end=None,
                                   source_style_outputs=None):
    '''
    Compute USLE zonal statistics in parallel for a set of areas.

    Parameters:
    - cfactor_src: List of C-factor files - should be yearly netCDF files with a time dimension.
    - kls_fn: Path to the KLS raster file.
    - scald_fn: Path to the scald raster file (optional, can be None).
    - fines_fn: Path to the fines raster file.
    - areas: GeoDataFrame of areas to compute statistics for.
    - change_map_fn: Path to a change scenario shapefile (optional).
    - buffer: Buffer size to apply to the areas (default: CELL_SIZE).
    - num_workers: Number of parallel workers to use (default: None, which will be determined automatically).
    - keep_intermediate_files: If True, keep the intermediate files created during processing.
    - batch_directory: Directory to store the batches of areas (default: 'tmp-area-batches').
    - job_results_directory: Directory to store the results of the jobs (default: 'tmp-results-usle-jobs').
    - source_style_outputs: If provided, the results will be written in the format expected by the Source plugin for Dynamic Sednet.
    '''
    if num_workers is None:
        num_workers = choose_number_of_parallel_workers(cores=None,ram_gb=None,n_cells=None)
    logger.info('Using %d parallel workers',num_workers)

    # Create batches of areas to process in parallel
    n_batches = 10 * num_workers
    if batch_directory is None:
        batch_directory = 'tmp-area-batches'
        clear_batches = True
    else:
        clear_batches = False
    batch_filenames = create_area_batches(areas, n_batches, dest=batch_directory, clear=clear_batches)

    if job_results_directory is None:
        job_results_directory = 'tmp-results-usle-jobs'
        if os.path.exists(job_results_directory):
            logger.info(f'Clearing existing job results directory: {job_results_directory}')
            shutil.rmtree(job_results_directory)
    os.makedirs(job_results_directory,exist_ok=False)

    compute_usle_zonal_statistics_ = dask.delayed(compute_usle_zonal_statistics)
    delayed = []
    for ix,batch in enumerate(batch_filenames):
        res = compute_usle_zonal_statistics_(cfactor_src,
                                             kls_fn,scald_fn,fines_fn,
                                             batch,
                                             f'{job_results_directory}/{ix}',
                                             change_map=change_map_fn,
                                             buffer=buffer)
        delayed.append(res)
    batch_issues = dask.compute(*delayed, scheduler='processes', num_workers=18)
    for ix, issues in enumerate(batch_issues):
        if len(issues):
            logger.warning(f'Batch {ix} had issues: {issues}. Possibly due to very small areas or no data in the area.')
    results = load_results(job_results_directory,apply_source_naming_convention=True)

    if start is not None and end is not None:
        time_period = pd.date_range(start,end,freq='MS')
    if time_period is not None:
        logger.info(f'Filling timeseries with long term means for period {time_period[0]} to {time_period[-1]}')
        results = {k: fill_seasonal(v, time_period) for k,v in results.items()}

    return results

def create_area_batches(areas,num_batches,dest,clear=False):
    '''
    Create batches of areas to process in parallel.
    The areas are grouped by their centroid coordinates and saved to a GeoJSON file.
    The number of batches is determined by the `num_batches` parameter.
    The output files are saved to the `dest` directory.
    If `clear` is True, the destination directory is cleared before saving the files.
    '''
    if os.path.exists(dest):
        if clear:
            logger.info(f'Clearing existing destination directory: {dest}')
            shutil.rmtree(dest)
        else:
            logger.error(f'Destination directory {dest} already exists. Set clear=True to clear it.')
            raise FileExistsError(f'Destination directory {dest} already exists. Set clear=True to clear it.')
    os.makedirs(dest, exist_ok=False)

    batches = cluster_polygons_by_centroid(areas, num_batches)

    batch_filenames = []
    for i, batch in enumerate(batches):
        batch_fn = os.path.join(dest, f'batch_{i}.geojson')
        txt = batch.to_json()
        with open(batch_fn,'w') as fp:
            fp.write(txt)
        batch_filenames.append(batch_fn)

    logger.info(f'Created {len(batches)} batches of areas in {dest}')
    return batch_filenames

def load_results(loc,apply_source_naming_convention=True):
    files = glob(os.path.join(loc,'*.csv'))
    results = {}
    for fn in files:
        df = pd.read_csv(fn,index_col=0,parse_dates=True)
        variable = os.path.splitext(os.path.basename(fn))[0].split('-',1)[1]
        if not variable in results:
            results[variable]=[]
        results[variable].append(df)
    result = {v:pd.concat(dfs,axis=1) for v,dfs in results.items()}
    if apply_source_naming_convention:
        COL_RENAMES={
            'klsc':'USLE_KLSC_Total',
            'klsc-fines':'USLE_KLSC_FinePerc',
            'cfactor':'CFactor'
        }
        result = {COL_RENAMES[v]:df.rename(columns=lambda c: f'{c}${COL_RENAMES[v]}') for v,df in result.items()}
    return result
