import logging
from glob import glob
import os
import subprocess
from time import sleep
import pandas as pd
import numpy as np
import hydrograph as hg
from string import Template
from . import RunDetails

logger = logging.getLogger(__name__)

PARAM_FN='ParameterTable.csv'
RAW_FN='RawResults.csv'
RESULTS_VALUE_COLUMN='Total_Load_in_Kg'

AREAS_FN='fuAreasTable.csv'
FILES=[
    PARAM_FN,
    RAW_FN,
    AREAS_FN
]
TABLES={
    'parameters':['CONSTITUENT','PARAMETER'],
    'raw':['Constituent','Process','BudgetElement'],
    'areas':[]
}
def add_per_year(df,row):
    df['kg_per_year'] = df[RESULTS_VALUE_COLUMN]/row['years']
    return df

TRANSFORMS={
    'raw':add_per_year
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
                  years=determine_num_years(os.path.dirname(param_fn)))
    name = result['run'].upper()
    if ('BASE' in name) or ('BL' in name):
        result['scenario'] = 'baseline'
    elif ('PREDEV' in name) or ('PD' in name):
        result['scenario'] = 'predev'
    else:
        result['scenario'] = 'unknown'
    logging.info('Detected run %s:%s (%d years)',result['model'],result['scenario'],result['years'])
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
    logging.warning('Reading ragged CSV file %s',fn)
    import csv

    with open(fn,'r') as fp:
      lines=list(csv.reader(fp))
      header = lines[0]
      extra_names = ['a','b','c']
      tbl = pd.read_csv(fn,skiprows=1,names=header+extra_names)
      tbl = tbl.drop(columns=extra_names)
      return tbl

def read_csv(fn):
    try:
        return pd.read_csv(fn)
    except pd.errors.ParserError:
        return read_ragged_csv(fn)

def find_all_runs(source_data_directories:list)->list:
    all_runs = []
    for source_dir in source_data_directories:
        dir_runs = map_runs_in_directory(source_dir)
        for run in dir_runs:
            if contains_matching_run(all_runs,run):
                logger.warning('Equivalent to run %s already in list',run_label(run))
                continue
            logging.info('Using run %s from %s',run_label(run),source_dir)
            all_runs.append(run)
    return all_runs

def load_tables(runs):
    res = {}
    for table in TABLES.keys():
        logger.info(f'Loading {table} for each run.')
        loaded = []
        for mod in runs:
            tbl = read_csv(mod[table])
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
    df['key'] = df['REGION']+':'+df['CATCHMENT']
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
    logger.info('Matching sediment models (Coarse to Fine)')
    raw['MODEL'] = raw.apply(match_sediment_model,axis=1)
    raw['is_emc_dwc'] = raw['MODEL'].apply(lambda m: m in emc_dwc_models)
    raw['is_timeseries'] = raw['MODEL'].apply(lambda m: m in ts_models)
    raw.loc[raw.BudgetElement=='Gully',['is_timeseries','is_emc_dwc']] = False
    # raw.loc[(raw.BudgetElement=='')&(raw.MODEL=='Cropping Sediment (Sheet & Gully) - GBR'),'is_emc_dwc'] = True
    raw.loc[(raw.BudgetElement=='Hillslope sub-surface soil')&(raw.MODEL=='Cropping Sediment (Sheet & Gully) - GBR'),'is_timeseries'] = False
    raw = raw.rename(columns=dict(CONSTITUENT='Constituent'))
    return raw

def prep(source_data_directories:list,dashboard_data_dir:str):
    def open_hg(lbl):
        path = os.path.join(dashboard_data_dir,lbl)
        logger.info(f'Opening dataset for %s at %s',lbl,path)
        return hg.open_dataset(path,'w')

    runs = find_all_runs(source_data_directories)
    logger.info('Got %d runs',len(runs))
    all_tables = load_tables(runs)

    fu_areas = all_tables['areas']
    fu_areas = fu_areas.rename(columns=dict(Catchment='CATCHMENT',FU='ELEMENT',Area='AREA'))
    fu_names=set(fu_areas.ELEMENT)
    all_tables['areas'] = fu_areas

    parameters = all_tables['parameters']
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
    model_element_index = parameters[['MODEL','ELEMENT','SCENARIO']].drop_duplicates()

    raw = all_tables['raw']
    fu_results, other_results = split_fu_and_stream(raw,fu_names)
    fu_results = clear_rows_for_zero_area_fus(fu_results,fu_areas,[RESULTS_VALUE_COLUMN,'kg_per_year'],keep_area=True)
    fu_results['AREA_HA'] = fu_results['AREA'] * 1e-4
    for col in [RESULTS_VALUE_COLUMN,'kg_per_year']:
        fu_results[f'{col}_per_ha'] = fu_results[col]/fu_results['AREA_HA']
    fu_results = fu_results.drop(columns=['AREA','AREA_HA'])
    raw = pd.concat([fu_results,other_results])
    raw = add_key(raw)
    raw = raw.dropna(subset=[RESULTS_VALUE_COLUMN])
    raw = classify_results(raw,parameters,model_parameter_index)
    all_tables['raw'] = raw

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

