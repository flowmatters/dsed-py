import logging
from glob import glob
import os
import shutil
import subprocess
from time import sleep
import pandas as pd
import numpy as np
import geopandas as gpd
import hydrograph as hg
from string import Template
from . import RunDetails
from .const import M_TO_KM, G_TO_KG, CM3_TO_M3, M_TO_MM
from .post import run_regional_contributor, back_calculate, MassBalanceBuilder
from .post.streambank import compute_streambank_parameters
from .util import read_source_csv
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask

logger = logging.getLogger(__name__)


PARAM_FN='ParameterTable.csv'
RAW_FN='RawResults.csv'
CLIMATE_FN='climateTable.csv'
RESULTS_VALUE_COLUMN='Total_Load_in_Kg'
RUN_METADATA_FN='DSScenarioRunInfo.xml'
NEST_DASK_JOBS=False

SEDIMENT_BULK_DENSITY = 1.5 # g/cm^3

AREAS_FN='fuAreasTable.csv'

TABLES={
    'parameters':['CONSTITUENT','PARAMETER'],
    'raw':['Constituent','Process','BudgetElement','Stage'],
    'areas':[],
    'run_metadata':[]
}

def add_per_year(df,row):
    df['kg_per_year'] = df[RESULTS_VALUE_COLUMN]/row['years']
    return df

def compute_generated_loads(raw,params):
    logger.info('Back-calculating generated loads')
    raw['Stage'] = 'Delivered'
    unrelated = raw[raw.Constituent.isin(['Flow'])]
    loads = raw[~raw.Constituent.isin(['Flow'])]
    generated = loads.copy()
    generated['Stage'] = 'Generated'
    generated = generated.groupby('MODEL').apply(lambda x: back_calculate.backcalc_loads(x,params))
    return pd.concat([loads,unrelated,generated])

TRANSFORMS={
    'raw':add_per_year,
    'parameters':compute_streambank_parameters
}

DEFAULT_REPORTING_LEVELS = ['Region','Basin_35','MU_48','WQI_Cal']

def load_reporting_regions(reporting_regions:str,reporting_levels:list):
    if isinstance(reporting_regions,str):
        reporting_regions = gpd.read_file(reporting_regions)
    if reporting_levels is None:
        reporting_levels = DEFAULT_REPORTING_LEVELS
    reporting_regions = reporting_regions[['SUBCAT']+reporting_levels]
    return reporting_regions

def determine_num_years(results_dir:str):
    run_info = RunDetails(results_dir)
    return run_info.yearsOfRecording

def map_run(param_fn:str,base_dir:str)->dict:
    rel_path = os.path.relpath(param_fn,base_dir) # Will ensure consistent separators
    path_parts = rel_path.replace(base_dir,'').split(os.sep)
    name = path_parts[0].upper()
    if len(name)==2:
        name = '_'.join([path_parts[0],path_parts[2]])
        scenario = path_parts[2]
    else:
        if ('BASE' in name) or ('BL' in name):
            scenario = 'baseline'
        elif ('PREDEV' in name) or ('PD' in name):
            scenario = 'predev'
        else:
            logger.info('Could not determine scenario from run name %s',name)
            scenario = 'unknown'
    result = dict(model=rel_path.replace(base_dir,'')[:2],
                  run=name,
                  scenario=scenario,
                  parameters=param_fn,
                  raw=param_fn.replace(PARAM_FN,RAW_FN),
                  areas=param_fn.replace(PARAM_FN,AREAS_FN),
                  years=determine_num_years(os.path.dirname(param_fn)),
                  run_metadata=os.path.join(os.path.dirname(param_fn),RUN_METADATA_FN))
    logger.info('Detected run %s:%s (%d years)',result['model'],result['scenario'],result['years'])
    return result

def map_runs_in_directory(results_dir:str) -> list:
    if not os.path.exists(results_dir):
        logger.error('Results directory %s does not exist',results_dir)
        return []
    pts = glob(os.path.join(results_dir,'**','ParameterTable.csv'),recursive=True)
    count = len(pts)
    if count == 0:
        logger.error('No parameter files found in %s',results_dir)
        return []

    if count>1:
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
    return read_source_csv(cache_fn)

def find_all_runs(source_data_directories:list)->list:
    all_runs = []
    logger.info('Finding all runs in %d directories',len(source_data_directories))
    for source_dir in source_data_directories:
        dir_runs = map_runs_in_directory(source_dir)
        for run in dir_runs:
            if contains_matching_run(all_runs,run):
                logger.warning('Equivalent to run %s already in list',run_label(run))
                continue
            logger.info('Using run %s from %s',run_label(run),source_dir)
            all_runs.append(run)
    return all_runs

def augment_table(tbl,run):
    tbl['REGION'] = run['model']
    tbl['SCENARIO'] = run['scenario']
    tbl.rename(columns=dict(ModelElement='CATCHMENT',FU='ELEMENT',ModelElementType='FEATURE_TYPE'),inplace=True)

def load_tables(runs,data_cache,reporting_regions=None):
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

            augment_table(tbl,mod)
            if table in TRANSFORMS:
                tbl = TRANSFORMS[table](tbl,mod)

            loaded.append(tbl)
        combined = pd.concat(loaded).reset_index().drop(columns='index')
        if 'CATCHMENT' in combined.columns and reporting_regions is not None:
            logger.debug(reporting_regions.head())
            combined = pd.merge(combined,reporting_regions,left_on=['CATCHMENT','REGION'],right_on=list(reporting_regions.columns[:2]),how='left')
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

def classify_results(raw,parameters,model_param_index,nest_dask_jobs=False):
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

    if nest_dask_jobs:
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

def extract_paramter(parameter:pd.DataFrame,param:str,new_name:str=None):
    if new_name is None:
        new_name = param
    param = parameter[parameter.PARAMETER==param][['REGION','SCENARIO','CATCHMENT','VALUE']]
    param = param.rename(columns=dict(VALUE=new_name))
    return param

def process_link_results(link_results,parameters):
    link_length = extract_paramter(parameters,'Link Length','length')
    link_length['length'] = M_TO_KM * link_length['length'].astype('f')

    link_results = pd.merge(link_results,link_length,on=['REGION','SCENARIO','CATCHMENT'],how='left')
    link_results['kg_per_km'] = link_results[RESULTS_VALUE_COLUMN]/link_results['length']
    link_results['kg_per_km_per_year'] = link_results['kg_per_year']/link_results['length']
    link_results = link_results.drop(columns='length')

    link_results_pivot = link_results.pivot(
        index=['REGION','FEATURE_TYPE','SCENARIO','CATCHMENT','Constituent'],
        columns='BudgetElement',
        values='kg_per_year').fillna(0.0).reset_index().rename(columns={'Flood Plain Deposition':'kg_per_year'})
    link_results_pivot['BudgetElement'] = 'Flood Plain Deposition'
    link_results_pivot['ELEMENT'] = 'Link'
    link_results_pivot['Process'] = 'Loss'
    link_results_pivot['PC'] = 100.0 * link_results_pivot['kg_per_year'] / (link_results_pivot['Subcatchment Supply'] + link_results_pivot['Link In Flow'])
    link_results_pivot['PC'] = link_results_pivot['PC'].fillna(0.0)
    link_results_pivot = link_results_pivot[[c for c in link_results_pivot.columns if c not in set(link_results.BudgetElement)]]
    fp_sediment = link_results_pivot[link_results_pivot.Constituent.str.startswith('Sediment')]
    fp_other = link_results_pivot[~link_results_pivot.Constituent.str.startswith('Sediment')]

    floodplain_area = extract_paramter(parameters,'Flood Plain Area','floodplain_area')
    floodplain_area['floodplain_area'] = floodplain_area['floodplain_area'].astype('f')
    bulk_denstiy_kg_m3 = SEDIMENT_BULK_DENSITY * G_TO_KG / CM3_TO_M3
    fp_sediment = pd.merge(fp_sediment,floodplain_area,on=['REGION','SCENARIO','CATCHMENT'],how='left')
    fp_sediment['m^3/year'] = fp_sediment['kg_per_year'] / bulk_denstiy_kg_m3
    fp_sediment['mm/year'] = M_TO_MM * fp_sediment['m^3/year'] / fp_sediment['floodplain_area']
    fp_sediment = fp_sediment.drop(columns='floodplain_area')

    floodplain_rows = link_results[link_results.BudgetElement=='Flood Plain Deposition']
    link_results = link_results[link_results.BudgetElement!='Flood Plain Deposition']
    floodplain_stats = pd.concat([fp_sediment,fp_other])
    floodplain_rows = pd.merge(floodplain_rows,floodplain_stats,
                               on=['REGION','SCENARIO','CATCHMENT','ELEMENT','Constituent','BudgetElement','Process','FEATURE_TYPE'])
    floodplain_rows = floodplain_rows.rename(columns={'kg_per_year_x':'kg_per_year'}).drop(columns='kg_per_year_y')
    link_results = pd.concat([link_results,floodplain_rows])

    return link_results

def process_storage_trapping(raw,storage_list):
    if storage_list is None:
        logger.warning('No storage list provided, calculating trapping efficiency for all nodes')
        storage_results = raw[raw.FEATURE_TYPE=='Node']
    else:
        storage_results = raw[raw.CATCHMENT.isin(storage_list)]
    if not len(storage_results):
        logger.warning('No storage results found')
        return pd.DataFrame()

    pivot_index=['REGION','SCENARIO','CATCHMENT','Constituent','FEATURE_TYPE']
    pivot_index = list(pivot_index)
    stats = pd.pivot_table(storage_results,index=pivot_index,columns='Process',values='Total_Load_in_Kg',aggfunc='sum',fill_value=0.0)
    if any(col not in stats.columns for col in ['Loss','Residual','Yield']):
        logger.warning('Not all of Loss, Residual and Yield present in storage results, skipping trapping efficiency calculation')
        return pd.DataFrame()
    stats['Trapping Efficiency'] = 100.0 * (1 - stats["Yield"]/(stats["Loss"]+stats["Residual"]+stats["Yield"]))
    rows = pd.melt(stats.reset_index(),id_vars=pivot_index,value_name='Total_Load_in_Kg',var_name='BudgetElement')
    result = rows[rows.BudgetElement.isin(['Trapping Efficiency'])]
    result['Process'] = 'Storage Trapping'

    return result

def subcatchment_totals(results):
    grouped = results.groupby(['REGION','SCENARIO','CATCHMENT','Constituent','Process','BudgetElement']).agg(
        {
            RESULTS_VALUE_COLUMN:'sum',
            'kg_per_year':'sum',
            'AREA':'sum'
        }).reset_index()
    grouped['ELEMENT'] = 'Subcatchment'
    grouped['FEATURE_TYPE'] = 'Catchment'
    return grouped

def process_run_data(runs,data_cache,nest_dask_jobs=False,reporting_regions=None,reporting_levels:list=None,storage_list=None):
    if len(runs)==1:
        logger.info('Processing single run %s',run_label(runs[0]))
    else:
        logger.info('Processing %d runs',len(runs))

    if reporting_regions is not None:
        logger.info('Processing reporting regions')
        reporting_regions = load_reporting_regions(reporting_regions,reporting_levels)

    all_tables = load_tables(runs,data_cache,reporting_regions)

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
    parameters_without_model = parameters_without_model.dropna(subset=['VALUE'])
    all_tables['parameters'] = parameters_without_model
    all_tables['parameters_orig'] = parameters

    model_parameter_index = parameters[['MODEL','PARAMETER']].drop_duplicates()

    logger.info('Processing raw results')
    orig_raw = raw = all_tables['raw']
    fu_results, other_results = split_fu_and_stream(raw,fu_names)
    fu_results = clear_rows_for_zero_area_fus(fu_results,fu_areas,[RESULTS_VALUE_COLUMN,'kg_per_year'],keep_area=True)
    sc_totals = subcatchment_totals(fu_results)
    all_tables['sc_totals'] = sc_totals

    fu_results = pd.concat([fu_results,sc_totals])
    fu_results['AREA_HA'] = fu_results['AREA'] * 1e-4
    for col in [RESULTS_VALUE_COLUMN,'kg_per_year']:
        fu_results[f'{col}_per_ha'] = fu_results[col]/fu_results['AREA_HA']
    fu_results = fu_results.drop(columns=['AREA','AREA_HA'])

    sc_inflows_to_link = sc_totals[sc_totals.Process=='Supply'].groupby(
        ['REGION','SCENARIO','CATCHMENT','Constituent']).agg(
        {
            RESULTS_VALUE_COLUMN:'sum',
            'kg_per_year':'sum'
        }).reset_index()
    sc_inflows_to_link['BudgetElement'] = 'Subcatchment Supply'
    sc_inflows_to_link['Process'] = 'Supply'
    sc_inflows_to_link['ELEMENT'] = 'Stream'
    sc_inflows_to_link['FEATURE_TYPE'] = 'Link'
    all_tables['sc_inflows_to_link'] = sc_inflows_to_link
    link_results = other_results[other_results.ELEMENT.isin(['Link','Stream'])]
    link_results = pd.concat([link_results,sc_inflows_to_link])
    other_results = other_results[~other_results.ELEMENT.isin(['Link','Stream'])]

    link_results = process_link_results(link_results,parameters)
    all_tables['link_results'] = link_results

    storage_results = process_storage_trapping(raw,storage_list)

    raw = pd.concat([fu_results,link_results,other_results,storage_results])
    raw = add_key(raw)
    raw = raw.dropna(subset=[RESULTS_VALUE_COLUMN])
    raw = classify_results(raw,parameters,model_parameter_index,nest_dask_jobs)
    raw = compute_generated_loads(raw,parameters)
    raw = raw.reset_index(drop=True)

    all_tables['raw'] = raw

    return all_tables

def concat_all_tables(all_tables):
    result = {}
    for k in all_tables[0].keys():
        result[k] = pd.concat([tbl[k] for tbl in all_tables])
    return result

def store_rsdr_results(runs,dashboard_data_dir:str,*job_results):
    for run, results in zip(runs,job_results):
        for key in ['scenario','model']:
            results[key] = run[key]
    all_results = pd.concat(job_results)
    all_results['rc'] = all_results['model'] + '-' + all_results['ModelElement']

    subcatchment_results = all_results.groupby(
        ['model','rc','Constituent','Rep_Region','scenario']).first().reset_index()
    subcatchment_pivot = subcatchment_results.pivot(
        columns='Constituent',
        index=['model','rc','Rep_Region','scenario'],
        values='RSDR').reset_index()

    ds = hg.open_dataset(os.path.join(dashboard_data_dir,'rsdr'),mode='w')
    for scenario in set(all_results.scenario):
        subset = subcatchment_pivot[subcatchment_pivot.scenario==scenario]
        for col in set(all_results.Constituent):
            logger.info('Storing RSDR results for %s/%s',scenario,col)
            columns = ['model','rc','Rep_Region','scenario',col]
            for keep, rr in [('last','local'),('first','final')]:
                table = subset.sort_values(col,ascending=True).drop_duplicates(
                    subset=['rc'],keep=keep)[columns]
                table = table.rename(columns={col:'rsdr'})
                ds.add_table(table,constituent=col,reporting_region=rr,scenario=scenario)

def build_rsdr_dataset(dashboard_data_dir:str,runs:list,network_data_dir:str,reporting_regions:str,rsdr_reporting:str='MU_48',client:Client=None):
    logger.info('Building RSDR dataset')
    jobs = []
    for run in runs:
        region = run['model']
        logger.info('Preparing regional contributor for %s (%s)',region,run['scenario'])
        network_fn = glob(os.path.join(network_data_dir,f'{region}*network.json'))[0]
        run_dir = os.path.dirname(run['parameters'])
        job = client.submit(run_regional_contributor,region,network_fn,reporting_regions,run_dir,rsdr_reporting)
        jobs.append(job)
    logger.info('Running %d RSDR jobs',len(jobs))
    result = client.submit(store_rsdr_results,runs,dashboard_data_dir,*jobs)
    return result

def prep_massbalance(run,reporting_areas,data_cache,dataset_path,network_data_dir=None):
    ds = hg.open_dataset(dataset_path,mode=hg.MODE_WRITE_NO_INDEX)
    reporting_levels = reporting_areas.columns[1:]
    raw = read_csv(run['raw'],data_cache)
    augment_table(raw,run)
    node_names = []
    if network_data_dir is not None:
        # Attribute node fluxes to nearest catchment in order to report in appropriate regions.
        # Try downstream first, then upstream if no downstream catchment found
        import veneer
        network_fn = glob(os.path.join(network_data_dir,f'{run["model"]}*network.json'))[0]
        network = veneer.load_network(network_fn)
        node_names = set(raw[(raw['FEATURE_TYPE']=='Node')&(raw['BudgetElement']!='Node Yield')]['CATCHMENT'])
        for node_name in node_names:
            try:
                node = network.by_name(node_name)
            except:
                logger.error('Node %s not found in network %s for %s',node_name,network_fn,run['model'])
                raise ValueError(f'Node {node_name} not found in network {network_fn} for {run["model"]}')
            ds_links = network.downstream_links(node)
            if len(ds_links)==0:
                logger.error('No downstream links found for node %s in %s',node_name,run['model'])
                raise ValueError(f'No downstream links found for node {node_name}')
            catchments = [c for c in [network.catchment_for_link(l) for l in ds_links] if c is not None]
            link_names = [l['properties']['name'] for l in ds_links]
            if not len(catchments):
                logger.warning('No catchment found for downstream link %s of node %s in %s',link_names,node_name,run['model'])
                us_links = network.upstream_links(node)
                link_names = [l['properties']['name'] for l in us_links]
                catchments = [c for c in [network.catchment_for_link(l) for l in us_links] if c is not None]
                if not len(catchments):
                    logger.error('No catchment found for upstream links %s of node %s in %s',link_names,node_name,run['model'])
                    raise ValueError(f'No catchment found for upstream or downstream link of node {node_name} in {run["model"]}')
            if len(catchments)>1:
                logger.warning('Multiple catchments found for node %s in %s, using first (%s)',node_name,run['model'],catchments[0]['properties']['name'])
            catchment = catchments[0]['properties']['name']
            raw.loc[(raw['FEATURE_TYPE']=='Node')&(raw['CATCHMENT']==node_name),'CATCHMENT'] = catchment
    
    mb_calc = MassBalanceBuilder(None,None)
    for level in reporting_levels:
        values = reporting_areas[level].dropna().unique()
        for value in values:
            areas = reporting_areas[reporting_areas[level]==value]
            raw_subset = pd.merge(raw,areas,left_on=['REGION','CATCHMENT'],right_on=[reporting_levels[0],'SUBCAT'],how='inner')
            mba_final, mass_balance_percentages, mass_balance_loss_v_supply = mb_calc.build_run_for_raw_results(raw_subset,run['years'])
            mb_clean = mba_final.rename(columns=lambda c: c.split(' (')[0])
            tags = dict(
                scenario=run['scenario'],
                scale=level,
                area=value
            )
            ds.add_table(mba_final,purpose='mass-balance',**tags)
            ds.add_table(mb_clean,purpose='mass-balance-clean',**tags)
            ds.add_table(mass_balance_percentages,purpose='mass-balance-percentages',**tags)
            ds.add_table(mass_balance_loss_v_supply,purpose='mass-balance-loss-v-supply',**tags)
    assert ds.index is not None
    return ds.index


def prep(source_data_directories:list,dashboard_data_dir:str,data_cache:str=None,
         network_data_dir:str=None,reporting_regions:str=None,reporting_levels:list=None,rsdr_reporting:str=None,parallel:int|Client=None,
         model_parameter_index=None,report_card_params=None,storage_list=None):
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
    reporting_levels: list
        List of reporting levels to use for the reporting regions
    rsdr_reporting: str
        Field name in the reporting regions shapefile/GeoJSON to use for RSDR reporting
    parallel: bool or Client
        Whether to run the processing in parallel or not. Default is True. Uses Dask
    report_card_params: dict
        Parameters for building the report card datasets. See build_report_card_datasets() for details
    Notes:
    * Processes RSDRs if network_data_dir and reporting_regions are provided
    '''
    if isinstance(storage_list,str):
        storage_list = pd.read_csv(storage_list,index_col=0).index.tolist()

    if isinstance(parallel,int):
        logger.info('Creating local Dask cluster with %d workers',parallel)
        cluster = LocalCluster(n_workers=parallel,threads_per_worker=1)
        client = Client(cluster)
        parallel = client
    elif parallel is None:
        logger.info('Running with default workers')
        cluster = LocalCluster(threads_per_worker=1,processes=True)
        client = Client(cluster)
        parallel = client
    else:
        logger.info('Using provided Dask client')
        client = parallel

    if data_cache is None:
        data_cache = os.path.abspath('./results-data-cache')

    def open_hg(lbl,mode=hg.MODE_WRITE):
        path = os.path.join(dashboard_data_dir,lbl)
        logger.info(f'Opening dataset for %s at %s',lbl,path)
        return hg.open_dataset(path,mode=mode)

    runs = find_all_runs(source_data_directories)
    logger.info('Got %d runs',len(runs))
    if len(runs)==0:
        logger.error('No runs found, exiting')
        return

    futures = []

    if report_card_params is not None:
        logger.info('Building report card datasets')
        # build_report_card_datasets(source_data_directories[0],
        #                            subcatchment_lut_fn=reporting_regions,
        #                            **report_card_params,
        #                            dashboard_data_dir=dashboard_data_dir)
        futures.append(client.submit(build_report_card_datasets,
                                    source_data_directories[0],
                                    subcatchment_lut_fn=reporting_regions,
                                    **report_card_params,
                                    dashboard_data_dir=dashboard_data_dir
                                    ))

    if network_data_dir and reporting_regions:
        logger.info('Processing RSDRs')
        f = build_rsdr_dataset(dashboard_data_dir,runs,network_data_dir,reporting_regions,rsdr_reporting=rsdr_reporting,client=client)
        futures.append(f)

    main_jobs = []
    for run in runs:
        logger.info('Run %s',run_label(run))
        main_jobs.append(client.submit(process_run_data,[run],data_cache,NEST_DASK_JOBS,reporting_regions,reporting_levels,storage_list))

    logger.info('Combining all tables')
    def create_main_datasets(model_parameter_index,*all_results):
        all_tables = concat_all_tables(all_results)
        parameters = all_tables['parameters_orig']

        if model_parameter_index is None:
            logger.info('Creating model parameter index')
            model_parameter_index = parameters[['MODEL','PARAMETER']].drop_duplicates()
        elif isinstance(model_parameter_index,str):
            logger.info('Loading model parameter index from %s',model_parameter_index)
            model_parameter_index = pd.read_csv(model_parameter_index,index_col=0)
        else:
            logger.info('Using provided model parameter index')

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
                    flag_columns = [c for c in subset.columns if c.startswith('is_')]
                    for col in flag_columns:
                        any_true = bool(subset[col].any())
                        tags[col.replace('is_','contains_')] = any_true
                    subset = subset.drop(columns=grouping_keys)
                    ds.add_table(subset,**tags)
            else:
                ds.add_table(full_tbl,purpose=tbl)
            ds.rewrite(True)

        logger.info('Creating indexes')
        ds = open_hg('indexes')
        ds.add_table(model_parameter_index,role='model-parameter')
        ds.add_table(model_element_index,role='model-element')
        ds.add_table(reporting_regions_df,role='reporting-regions')
        return None

    reporting_regions_df = load_reporting_regions(reporting_regions,reporting_levels)

    logger.info('Building mass balance dataset')
    mb_dataset = open_hg('massbalance')
    mb_dataset.rewrite(False)
    mb_jobs = []
    for run in runs:
        region_subset = reporting_regions_df[reporting_regions_df[reporting_levels[0]]==run['model']]
        mb_jobs.append(client.submit(prep_massbalance,run,region_subset,data_cache,mb_dataset.path,network_data_dir))

    def combine_mb_indexes(*indexes):
        mb_dataset = open_hg('massbalance',hg.MODE_READ_WRITE)
        mb_dataset.rewrite(False)
        logger.info('Combining %d indexes',len(indexes))
        for index in indexes:
            mb_dataset._add_index(index)
        mb_dataset.rewrite(True)
    futures.append(client.submit(combine_mb_indexes,*mb_jobs))

    main_results = client.gather(main_jobs,errors='raise')
    create_main_datasets(model_parameter_index,*main_results)

    futures = [f for f in futures if f is not None]
    if len(futures):
        logger.info('Waiting for %d task(s) to finish',len(futures))
        results = []
        for f in futures:
            results.append(f.result())

    logger.info('Done')

def build_report_card_datasets(source_data,observed_loads_fn,constituents,
                               fus_of_interest,num_years,subcatchment_lut_fn,
                               overall_label,loads_obs_column,dashboard_data_dir):
    logger.info('Building report card datasets')
    from .post import report_card as rc
    rc.progress = rc.nop
    rc.OVERALL_REGION = overall_label
    rc.OBSERVATION_COLUMN=loads_obs_column
    rc.populate_load_comparisons(source_data,observed_loads_fn,constituents,dashboard_data_dir)
    rc.populate_overview_data(source_data,subcatchment_lut_fn,constituents,fus_of_interest,num_years,dashboard_data_dir)

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

