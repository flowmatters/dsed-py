import logging
from glob import glob
import os
import shutil
import subprocess
from time import sleep
import pandas as pd
import numpy as np
import hydrograph as hg
from string import Template
from . import RunDetails
from .const import M_TO_KM, PERCENT_TO_FRACTION
from .post import run_regional_contributor
import dask.dataframe as dd
import dask

logger = logging.getLogger(__name__)

PARAM_FN='ParameterTable.csv'
RAW_FN='RawResults.csv'
RESULTS_VALUE_COLUMN='Total_Load_in_Kg'
RUN_METADATA_FN='DSScenarioRunInfo.xml'
NEST_DASK_JOBS=False

AREAS_FN='fuAreasTable.csv'

TABLES={
    'parameters':['CONSTITUENT','PARAMETER'],
    'raw':['Constituent','Process','BudgetElement','Stage'],
    'areas':[],
    'run_metadata':[]
}

LOAD_COLUMNS=[
    'Total_Load_in_Kg',
    'kg_per_year',
    'Total_Load_in_Kg_per_ha',
    'kg_per_year_per_ha',
    'kg_per_km',
    'kg_per_km_per_year'
]
MODELS_WITHOUT_DELIVERY_RATIO = {
    'In Stream Dissolved Nutrient Model - SedNet',
    'EMC/DWC',
    'Reservoir Dissolved Constituent Decay - SedNet',
    'Lewis Trapping Model - GBR',
    'Dissolved Nutrient Generation - SedNet',
    'In Stream Coarse Sediment Model - SedNet',
    'In Stream Fine Sediment Model - SedNet',
    'In Stream Particulate Nutrient Model - SedNet'
}

def backcalc_gully_usle(results):
    def choose_sdr(row):
        is_gully = row.BudgetElement=='Gully'
        is_fine = row.Constituent=='Sediment - Fine'
        param = 'Gully SDR - ' if is_gully else 'USLE HSDR '
        param += 'Fine' if is_fine else 'Coarse'
        p = float(row[param])
        return p * PERCENT_TO_FRACTION

    results['effective_sdr'] = results.apply(choose_sdr,axis=1)
    for col in LOAD_COLUMNS:
        results[col] /= results['effective_sdr']
    return results

def backcalc_gully_emc(results):
    def choose_sdr(row):
        is_gully = row.BudgetElement=='Gully'
        if not is_gully:
            return 1.0

        is_fine = row.Constituent=='Sediment - Fine'
        param = 'Gully SDR - '
        param += 'Fine' if is_fine else 'Coarse'
        p = float(row[param])
        return p * PERCENT_TO_FRACTION

    results['effective_sdr'] = results.apply(choose_sdr,axis=1)
    for col in LOAD_COLUMNS:
        results[col] /= results['effective_sdr']
    return results

def backcalc_cropping_sediment(results):
    def choose_sdr(row):
        is_gully = row.BudgetElement=='Gully'
        is_fine = row.Constituent=='Sediment - Fine'
        sed = 'Fine' if is_fine else 'Coarse'

        if is_gully:
            param = 'Gully SDR - ' + sed
        else:
            param = 'Hillslope Sediment Delivery Ratio - %s Sediment' % sed

        p = float(row[param])
        return p * PERCENT_TO_FRACTION

    results['effective_sdr'] = results.apply(choose_sdr,axis=1)
    for col in LOAD_COLUMNS:
        results[col] /= results['effective_sdr']

    return results

def backacl_diss_n_ts(results):
    def choose_sdr(row):
        is_leached_seepage = ('Seepage' in row.BudgetElement) or (row.BudgetElement=='Leached')
        param = 'Delivery Ratio - Leached To Seepage' if is_leached_seepage else 'Delivery Ratio - Surface'

        p = float(row[param])
        return p * PERCENT_TO_FRACTION

    results['effective_sdr'] = results.apply(choose_sdr,axis=1)
    for col in LOAD_COLUMNS:
        results[col] /= results['effective_sdr']

    return results

def backalc_regular_dr_model(results):
    results['effective_sdr'] = results['Delivery Ratio'].astype('f') * PERCENT_TO_FRACTION
    for col in LOAD_COLUMNS:
        results[col] /= results['effective_sdr']

    return results

def backcalc_part_nut(results):
    def choose_sdr(row):
        if row.BudgetElement.startswith('Gully'):
            return float(row['Gully Delivery Ratio (Conversion Factor)'])*PERCENT_TO_FRACTION
        elif row.BudgetElement.startswith('Hillslope'):
            return float(row['Hillslope Delivery Ratio (Conversion Factor)'])*PERCENT_TO_FRACTION
        return 1.0

    results['effective_sdr'] = results.apply(choose_sdr,axis=1)
    for col in LOAD_COLUMNS:
        results[col] /= results['effective_sdr']

    return results

GENERATED_LOADS={
    'Sediment Generation (USLE & Gully) - SedNet' :backcalc_gully_usle,
    'Sediment Generation (EMC & Gully) - SedNet' :backcalc_gully_emc,
    'Cropping Sediment (Sheet & Gully) - GBR' :backcalc_cropping_sediment,
    'Dissolved Nitrogen TimeSeries Load Model - GBR' :backacl_diss_n_ts,
    'Dissolved Phosphorus Nutrient Model - GBR' :backalc_regular_dr_model,
    'Particulate Nutrient Generation - SedNet' :backcalc_part_nut,
    'TimeSeries Load Model - SedNet' :backalc_regular_dr_model
}

def add_per_year(df,row):
    df['kg_per_year'] = df[RESULTS_VALUE_COLUMN]/row['years']
    return df

def compute_derived_parameters(params,*args):
    logger.info('Computing derived parameters (Retreat Rate)')
    BEMF='Bank Erosion Management Factor'
    BEC='Bank Erosion Coefficent'
    SLOPE='Link Slope'
    BFD='Bank Full Flow'
    RR='Retreat Rate'
    MRR='Modelled Retreat Rate'
    SBD='Sediment Bulk Density'
    BH='Bank Height'
    LL='Link Length'
    SE = 'Soil Erodibility'
    RVP = 'Riparian Vegetation Percentage'
    MRVE = 'Maximum Riparian Vegetation Effectiveness'

    streambank_params = params[(params.ELEMENT!='Node')&params.PARAMETER.isin([BEMF,BEC,SLOPE,BFD,SBD,BH,LL,SE,RVP,MRVE])]

    MABE='Mean Annual Bank Erosion'
    MCF='Mass Conversion Factor'
    BE='Bank Erodibility'
    index_cols = set(streambank_params.columns)-{'PARAMETER','VALUE'}
    streambank_params = streambank_params.pivot(index=index_cols,columns='PARAMETER',values='VALUE')
    streambank_params = streambank_params.astype('f')
    density_water = 1000.0 # kg.m^-3
    gravity = 9.81        # m.s^-2

    streambank_params[RR] = streambank_params[BEC] * \
                            streambank_params[BEMF] * \
                            streambank_params[BFD] * \
                            streambank_params[SLOPE] * \
                            density_water * \
                            gravity
    soil_erodibility = streambank_params[SE] * 0.01
    riparian_efficacy = np.minimum(streambank_params[RVP],streambank_params[MRVE]) * 0.01
    streambank_params[BE] = (1 - riparian_efficacy) * soil_erodibility
    streambank_params[MCF] = streambank_params[SBD] * streambank_params[LL] * streambank_params[BH]
    streambank_params[MRR] = streambank_params[RR] * streambank_params[BE]
    streambank_params[MABE] = streambank_params[MRR] * streambank_params[MCF]

    streambank_params = pd.melt(streambank_params,ignore_index=False,value_name='VALUE').reset_index()
    computed_params = streambank_params[streambank_params.PARAMETER.isin([RR,BE,MABE,MRR,MCF])]

    params = pd.concat([params,computed_params])
    return params

def backcalc_loads(delivered,params):
    model = delivered.MODEL.iloc[0]
    params = params[params.MODEL==model]
    logger.info('Back-calculating loads for %s',delivered.MODEL.iloc[0])

    if model in MODELS_WITHOUT_DELIVERY_RATIO:
        return delivered

    if model not in GENERATED_LOADS:
        logger.warning('No back calculation method for %s. Assuming Generated=Delivered',model)
        return delivered

    backcalc_method = GENERATED_LOADS[model]

    params = params[params['MODEL']==model]
    param_table = params.pivot(index=['REGION','SCENARIO','CATCHMENT','ELEMENT','LINK','CONSTITUENT'],columns='PARAMETER',values='VALUE')
    delivered_columns = list(delivered.columns)

    delivered = pd.merge(delivered,param_table,
                         left_on=['REGION','SCENARIO','CATCHMENT','ELEMENT','Constituent'],
                         right_on=['REGION','SCENARIO','CATCHMENT','ELEMENT','CONSTITUENT'],
                         how='left')
    delivered = backcalc_method(delivered)
    delivered = delivered[delivered_columns]
    return delivered
    # n_elements = len(set(delivered.BudgetElement))
    # n_constituents = len(set(delivered.Constituent))
    # if n_elements>1 or n_constituents>1:
    #     logger.warning('Cannot back-calculate loads for multiple elements or constituents')
    #     print(delivered.MODEL.iloc[0])
    #     print(set(delivered.BudgetElement))
    #     print(set(delivered.Constituent))
    #     assert False
    # return delivered

def compute_generated_loads(raw,params):
    logger.info('Back-calculating generated loads')
    raw['Stage'] = 'Delivered'
    unrelated = raw[raw.Constituent.isin(['Flow'])]
    loads = raw[~raw.Constituent.isin(['Flow'])]
    generated = loads.copy()
    generated['Stage'] = 'Generated'
    generated = generated.groupby('MODEL').apply(lambda x: backcalc_loads(x,params))
    return pd.concat([loads,unrelated,generated])

TRANSFORMS={
    'raw':add_per_year,
    'parameters':compute_derived_parameters
}

def determine_num_years(results_dir:str):
    run_info = RunDetails(results_dir)
    return run_info.yearsOfRecording

def map_run(param_fn:str,base_dir:str)->dict:
    rel_path = os.path.relpath(param_fn,base_dir)
    result = dict(model=rel_path.replace(base_dir,'')[:2],
                  run=rel_path.replace(base_dir,'').split('/')[0],
                  parameters=param_fn,
                  raw=param_fn.replace(PARAM_FN,RAW_FN),
                  areas=param_fn.replace(PARAM_FN,AREAS_FN),
                  years=determine_num_years(os.path.dirname(param_fn)),
                  run_metadata=os.path.join(os.path.dirname(param_fn),RUN_METADATA_FN))
    name = result['run'].upper()
    if ('BASE' in name) or ('BL' in name):
        result['scenario'] = 'baseline'
    elif ('PREDEV' in name) or ('PD' in name):
        result['scenario'] = 'predev'
    else:
        result['scenario'] = 'unknown'
    logger.info('Detected run %s:%s (%d years)',result['model'],result['scenario'],result['years'])
    return result

def map_runs_in_directory(results_dir:str) -> list:
    pts = glob(os.path.join(results_dir,'**','ParameterTable.csv'),recursive=True)
    if len(pts)>1:
      base_dir = os.path.commonpath(pts)
    else:
      base_dir = results_dir
    param_map = [map_run(fn,base_dir) for fn in pts]
    return param_map

def run_label(run:dict)->str:
    return Template('${model}:${scenario} (${run})').substitute(run)

def runs_match(run1:dict,run2:dict)->bool:
    compare_keys = ['model','scenario']
    for k in compare_keys:
        if run1[k]!=run2[k]:
            return False
    return True

def contains_matching_run(all_runs,run):
    for r in all_runs:
        if runs_match(r,run):
            return True
    return False

def read_ragged_csv(fn):
    logger.warning('Reading ragged CSV file %s',fn)
    import csv

    with open(fn,'r') as fp:
      lines=list(csv.reader(fp))
      header = lines[0]
      extra_names = ['a','b','c']
      tbl = pd.read_csv(fn,skiprows=1,names=header+extra_names)
      tbl = tbl.drop(columns=extra_names)
      return tbl

def cache_filename(fn,data_cache):
    return os.path.abspath(os.path.join(data_cache,fn.replace('\\\\','_').replace(':','_')))

def cache_hit(cache_fn,source_fn):
    if not os.path.exists(cache_fn):
        return False
    cache_mtime = os.path.getmtime(cache_fn)
    source_mtime = os.path.getmtime(source_fn)
    hit = cache_mtime > source_mtime
    return hit

def ensure_cache(fn,data_cache):
    cache_fn = cache_filename(fn,data_cache)
    if not cache_hit(cache_fn,fn):
        logger.info('Caching %s to %s',fn,cache_fn)
        os.makedirs(os.path.dirname(cache_fn),exist_ok=True)
        shutil.copy(fn,cache_fn)
    return cache_fn

def read_xml(fn,data_cache):
    cache_fn = ensure_cache(fn,data_cache)
    # Read the xml and create a single row dataframe using the elements under the root element as columns
    import xml.etree.ElementTree as ET
    tree = ET.parse(cache_fn)
    root = tree.getroot()
    data = {child.tag:child.text for child in root}
    return pd.DataFrame([data])

def read_csv(fn,data_cache):
    cache_fn = ensure_cache(fn,data_cache)

    try:
        return pd.read_csv(cache_fn)
    except pd.errors.ParserError:
        return read_ragged_csv(cache_fn)

def find_all_runs(source_data_directories:list)->list:
    all_runs = []
    for source_dir in source_data_directories:
        dir_runs = map_runs_in_directory(source_dir)
        for run in dir_runs:
            if contains_matching_run(all_runs,run):
                logger.warning('Equivalent to run %s already in list',run_label(run))
                continue
            logger.info('Using run %s from %s',run_label(run),source_dir)
            all_runs.append(run)
    return all_runs

def load_tables(runs,data_cache):
    res = {}
    for table in TABLES.keys():
        logger.info(f'Loading {table} for each run.')
        loaded = []
        for mod in runs:
            logger.info(f'Loading %s for %s from %s',table,run_label(mod),mod[table])
            fn = mod[table]
            if fn.endswith('.xml'):
                tbl = read_xml(fn,data_cache)
            else:
                tbl = read_csv(fn,data_cache)
            tbl['REGION'] = mod['model']
            tbl['SCENARIO'] = mod['scenario']
            tbl.rename(columns=dict(ModelElement='CATCHMENT',FU='ELEMENT',ModelElementType='FEATURE_TYPE'),inplace=True)
            if table in TRANSFORMS:
                tbl = TRANSFORMS[table](tbl,mod)

            loaded.append(tbl)
        combined = pd.concat(loaded).reset_index().drop(columns='index')
        res[table] = combined
    return res

def split_fu_and_stream(df,fu_names):
    fu_rows = df[df.ELEMENT.isin(fu_names)]
    other_rows = df[~df.ELEMENT.isin(fu_names)]
    return fu_rows, other_rows

def clear_rows_for_zero_area_fus(df,fu_areas,column,keep_area=False):
    df = pd.merge(df,fu_areas,on=['SCENARIO','REGION','CATCHMENT','ELEMENT'])
    df.loc[df['AREA']==0.0,column] = np.nan
    if not keep_area:
        df = df.drop(columns='AREA')
    return df

def add_key(df):
    df['rcf'] = df['REGION']+'-'+df['CATCHMENT']+'-'+df['ELEMENT']
    df['rc'] = df['REGION']+'-'+df['CATCHMENT']
    return df

def classify_results(raw,parameters,model_param_index):
    logger.info('Classifying raw results')
    emc_dwc_parameters = model_param_index[model_param_index.PARAMETER.str.contains('EMC')|model_param_index.PARAMETER.str.contains('Event Mean Concentration')]
    # |model_param_index.PARAMETER.str.contains('DWC')|model_param_index.PARAMETER.str.contains('Dry Weather Concentration')]
    emc_dwc_parameters = emc_dwc_parameters[~emc_dwc_parameters.PARAMETER.str.endswith('model')]
    emc_dwc_models = set(emc_dwc_parameters.MODEL)
    ts_parameters = model_param_index[model_param_index.PARAMETER.str.contains('TimeSeries')|model_param_index.MODEL.str.contains('TimeSeries')]
    ts_models = set(ts_parameters.MODEL)

    lhs = parameters[['REGION','SCENARIO','ELEMENT','CONSTITUENT','MODEL']].drop_duplicates()
    raw = raw.rename(columns={'Constituent':'CONSTITUENT'})
    raw['ELEMENT'] = raw['ELEMENT'].apply(lambda e: 'Link' if e=='Stream' else e)
    raw = pd.merge(lhs,raw,on=['REGION','SCENARIO','ELEMENT','CONSTITUENT'],how='right')

    sed_fine_rows = lhs[lhs.CONSTITUENT=='Sediment - Fine']
    def match_sediment_model(row):
        if row.CONSTITUENT != 'Sediment - Coarse':
            return row.MODEL
        if not pd.isnull(row.MODEL):
            return row.MODEL
        matching = (sed_fine_rows.REGION==row.REGION) & \
                  (sed_fine_rows.SCENARIO==row.SCENARIO) & \
                  (sed_fine_rows.ELEMENT==row.ELEMENT) & \
                  (sed_fine_rows.CONSTITUENT=='Sediment - Fine')

        rows = sed_fine_rows[matching]
        if not len(rows):
            return np.nan
        return rows.iloc[0].MODEL

    if NEST_DASK_JOBS:
        logger.info('Matching sediment models (Coarse to Fine) in partitions')
        dask_raw = dd.from_pandas(raw,npartitions=20)
        logger.info('Got partitioned raw data')
        raw['MODEL'] = dask_raw.map_partitions(lambda df:df.apply(match_sediment_model,axis=1),meta=pd.Series(dtype='str')).compute()
    else:
        logger.info('Matching sediment models (Coarse to Fine) in single thread')
        raw['MODEL'] = raw.apply(match_sediment_model,axis=1)
    logger.info('Matched sediment models')
    raw['is_emc_dwc'] = raw['MODEL'].apply(lambda m: m in emc_dwc_models)
    raw['is_timeseries'] = raw['MODEL'].apply(lambda m: m in ts_models)
    logger.info('Mapped emc and ts properties')
    raw.loc[raw.BudgetElement=='Gully',['is_timeseries','is_emc_dwc']] = False
    # raw.loc[(raw.BudgetElement=='')&(raw.MODEL=='Cropping Sediment (Sheet & Gully) - GBR'),'is_emc_dwc'] = True
    raw.loc[(raw.BudgetElement=='Hillslope sub-surface soil')&(raw.MODEL=='Cropping Sediment (Sheet & Gully) - GBR'),'is_timeseries'] = False
    raw = raw.rename(columns=dict(CONSTITUENT='Constituent'))
    return raw

def proces_run_data(runs,data_cache):
    all_tables = load_tables(runs,data_cache)

    fu_areas = all_tables['areas']
    fu_areas = fu_areas.rename(columns=dict(Catchment='CATCHMENT',FU='ELEMENT',Area='AREA'))
    fu_names=set(fu_areas.ELEMENT)
    all_tables['areas'] = fu_areas

    logger.info('Processing parameters')
    parameters = all_tables['parameters']
    # parameters = compute_derived_parameters(parameters)
    fu_params, other_params = split_fu_and_stream(parameters,fu_names)
    fu_params = clear_rows_for_zero_area_fus(fu_params,fu_areas,'VALUE')
    parameters = pd.concat([fu_params,other_params])

    parameters = parameters[~parameters.PARAMETER.isin(['USLEmodel','GULLYmodel','Hydropower','OutletManager'])]
    parameters.loc[parameters.ELEMENT.str.startswith('link for catchment'),'ELEMENT']='Link'
    parameters.loc[parameters.CATCHMENT==parameters.ELEMENT,'ELEMENT']='Node'
    parameters.loc[parameters.CONSTITUENT.isna(),'CONSTITUENT']='Flow'
    parameters_without_model = parameters.drop(columns='MODEL')
    parameters_without_model = add_key(parameters_without_model)
    parameters_without_model = parameters_without_model.dropna()
    all_tables['parameters'] = parameters_without_model
    all_tables['parameters_orig'] = parameters

    model_parameter_index = parameters[['MODEL','PARAMETER']].drop_duplicates()

    logger.info('Processing raw results')
    raw = all_tables['raw']
    fu_results, other_results = split_fu_and_stream(raw,fu_names)
    fu_results = clear_rows_for_zero_area_fus(fu_results,fu_areas,[RESULTS_VALUE_COLUMN,'kg_per_year'],keep_area=True)
    fu_results['AREA_HA'] = fu_results['AREA'] * 1e-4
    for col in [RESULTS_VALUE_COLUMN,'kg_per_year']:
        fu_results[f'{col}_per_ha'] = fu_results[col]/fu_results['AREA_HA']
    fu_results = fu_results.drop(columns=['AREA','AREA_HA'])

    link_results = other_results[other_results.ELEMENT.isin(['Link','Stream'])]
    other_results = other_results[~other_results.ELEMENT.isin(['Link','Stream'])]

    link_length = parameters[parameters.PARAMETER=='Link Length'][['REGION','SCENARIO','CATCHMENT','VALUE']]
    link_length = link_length.rename(columns=dict(VALUE='length'))
    link_length['length'] = M_TO_KM * link_length['length'].astype('f')

    link_results = pd.merge(link_results,link_length,on=['REGION','SCENARIO','CATCHMENT'],how='left')
    link_results['kg_per_km'] = link_results[RESULTS_VALUE_COLUMN]/link_results['length']
    link_results['kg_per_km_per_year'] = link_results['kg_per_year']/link_results['length']
    link_results = link_results.drop(columns='length')

    raw = pd.concat([fu_results,link_results,other_results])
    raw = add_key(raw)
    raw = raw.dropna(subset=[RESULTS_VALUE_COLUMN])
    raw = classify_results(raw,parameters,model_parameter_index)
    raw = compute_generated_loads(raw,parameters)

    all_tables['raw'] = raw
    return all_tables

def concat_all_tables(all_tables):
    result = {}
    for k in all_tables[0].keys():
        result[k] = pd.concat([tbl[k] for tbl in all_tables])
    return result

def build_rsdr_dataset(dashboard_data_dir:str,runs:list,network_data_dir:str,reporting_regions:str):
    jobs = []
    regional_contributor_ = dask.delayed(run_regional_contributor)
    for run in runs:
        region = run['model']
        logger.info('Preparing regional contributor for %s (%s)',region,run['scenario'])
        network_fn = glob(os.path.join(network_data_dir,f'{region}*network.json'))[0]
        run_dir = os.path.dirname(run['parameters'])
        jobs.append(regional_contributor_(run['model'],network_fn,reporting_regions,run_dir))
    logger.info('Running %d RSDR jobs',len(jobs))
    job_results = dask.compute(*jobs)
    logger.info('Got %d RSDR results',len(job_results))

    for run, results in zip(runs,job_results):
        for key in ['scenario','model']:
            results[key] = run[key]
    all_results = pd.concat(job_results)
    all_results['rc'] = all_results['model'] + '-' + all_results['ModelElement']

    subcatchment_results = all_results.groupby(
        ['rc','Constituent','Rep_Region','scenario']).first().reset_index()
    subcatchment_pivot = subcatchment_results.pivot(
        columns='Constituent',
        index=['rc','Rep_Region','scenario'],
        values='RSDR').reset_index()

    ds = hg.open_dataset(os.path.join(dashboard_data_dir,'rsdr'),mode='w')
    for scenario in set(all_results.scenario):
        subset = subcatchment_pivot[subcatchment_pivot.scenario==scenario]
        for col in set(all_results.Constituent):
            logger.info('Storing RSDR results for %s/%s',scenario,col)
            columns = ['rc','Rep_Region','scenario',col]
            for keep, rr in [('last','local'),('first','final')]:
                table = subset.sort_values(col,ascending=True).drop_duplicates(
                    subset=['rc'],keep=keep)[columns]
                table = table.rename(columns={col:'rsdr'})
                ds.add_table(table,constituent=col,reporting_region=rr,scenario=scenario)

def prep(source_data_directories:list,dashboard_data_dir:str,data_cache:str=None,
         network_data_dir:str=None,reporting_regions:str=None):
    '''
    Prepares the dashboard data for the given source data directories

    source_data_directories: list of str
        List of directories containing the source data
    dashboard_data_dir: str
        Directory to store the dashboard data
    data_cache: str
        Directory to store the data cache
    network_data_dir: str
        Directory containing the node link networks in Veneer GeoJSON format
    reporting_regions: str
        Shapefile/GeoJSON containing the reporting regions (subcatchments with attributes for RepReg)

    Notes:
    * Processes RSDRs if network_data_dir and reporting_regions are provided
    '''

    if data_cache is None:
        data_cache = os.path.abspath('./results-data-cache')

    def open_hg(lbl):
        path = os.path.join(dashboard_data_dir,lbl)
        logger.info(f'Opening dataset for %s at %s',lbl,path)
        return hg.open_dataset(path,'w')

    runs = find_all_runs(source_data_directories)
    logger.info('Got %d runs',len(runs))

    if network_data_dir and reporting_regions:
        logger.info('Processing RSDRs')
        build_rsdr_dataset(dashboard_data_dir,runs,network_data_dir,reporting_regions)

    all_tables = []
    for run in runs:
        logger.info('Run %s',run_label(run))
        all_tables.append(dask.delayed(proces_run_data)([run],data_cache))
    all_tables = dask.compute(*all_tables)
    logger.info('Combining all tables')
    all_tables = concat_all_tables(all_tables)

    parameters = all_tables['parameters_orig']
    model_parameter_index = parameters[['MODEL','PARAMETER']].drop_duplicates()
    model_element_index = parameters[['MODEL','ELEMENT','SCENARIO']].drop_duplicates()

    for tbl,grouping_keys in TABLES.items():
        logger.info(f'Creating dataset for {tbl}')
        ds = open_hg(tbl)
        ds.rewrite(False)
        full_tbl = all_tables[tbl]
        # full_tbl = full_tbl.dropna()
        if len(grouping_keys):
            for grouping, subset in full_tbl.groupby(grouping_keys):
                tags = dict(zip(grouping_keys,grouping))
                subset = subset.drop(columns=grouping_keys)
                ds.add_table(subset,**tags)
        else:
            ds.add_table(full_tbl,purpose=tbl)
        ds.rewrite(True)

    logger.info('Creating indexes')
    ds = open_hg('indexes')
    ds.add_table(model_parameter_index,role='model-parameter')
    ds.add_table(model_element_index,role='model-element')
    logger.info('Done')
    # return all_tables

def host(dashboard_data_dir:str):
    port = 8765
    host_process = subprocess.Popen(['python','-m','hydrograph._host',str(port)],cwd=dashboard_data_dir)
    sleep(0.5)
    if host_process.poll() is not None:
      host_process = None
      raise Exception('Failed to start host process')
    logger.info('Host process started on port %d',port)
    input('Press enter to exit')
    host_process.terminate()
    host_process.wait()
    logger.info('Hosting stopped')

