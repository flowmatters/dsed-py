import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from glob import glob
import dask
import os
import math
import shutil
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
    for y, data in df.groupby('y'):
        files = list(data.sort_values('m')['fn'].values)
        dest_fn = os.path.join(dest_dir,f'cfactor-{y}.nc')
        if os.path.exists(dest_fn) and not replace_existing:
            logger.info(f'Skipping existing file {dest_fn}')
            continue
        res = convert_cfactor_grids_(files,dest_fn)
        jobs.append(res)
    logger.info(f'Running %d jobs across %d processes to convert C-factor grids to NetCDF', len(jobs), cpu_count)
    _ = dask.compute(*jobs,scheduler='processes', num_workers=cpu_count)

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

def load_areas(fn,match=None,fus=None,catchments=None):
    '''
    Load areas (intersected SC/FU) from a shapefile, filtering by FUs and/or catchments if specified and rasterising the result.

    Rasterisation is done to match the CRS and resolution of the provided `match` raster.

    Returns a GeoDataFrame of areas and an xarray DataArray of the rasterised areas.
    '''
    areas = gpd.read_file(fn)
    areas = filter_geodataframe(areas, IntFUs=fus,IntSCs=catchments)
    if areas.crs is None or \
       fn.endswith('.json'): # JSON files may not have a CRS defined and are assumed (by Geopandas) to be WGS84
        logger.info(f'No CRS defined for {fn}, using CRS from match raster')
        assert match.rio.crs is not None
        areas.set_crs(match.rio.crs,inplace=True,allow_override=True)
    assert len(areas) > 0, f'No areas found in {fn} with the specified filters: {fus}, {catchments}'
    if ('IntSCFU' not in areas.columns) or areas['IntSCFU'].dtype != int:
        areas['IntSCFU'] = areas.index.astype(int)
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

