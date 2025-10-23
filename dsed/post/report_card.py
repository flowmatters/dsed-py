import os
from pathlib import Path
from . import moriasi
from ..util import read_source_csv
import spotpy
import matplotlib
import dsed.const as c
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import hydrograph as hg
from matplotlib.sankey import Sankey
import logging
logger = logging.getLogger(__name__)

CONSTITUENT_REPORTING_NAMES={
    'Sediment - Fine': 'TSS',
    'N_Particulate': 'PN',
    'P_Particulate': 'PP',
    'P_FRP': 'DIP',
    'N_DON': 'DON',
    'P_DOP': 'DOP',
    'N_DIN': 'DIN'
}

OVERALL_REGION='Overall'
OBSERVATION_COLUMN='Observations'

FU_COLORS = ['yellow','darkgreen','plum','purple','lime','gray','orangered','navy','greenyellow','red',]

REGION_LU_FILE_NAME_SUFFIX = '_Subcat_Regions_LUT.csv'
REGION_LUT_NODE_SC_FILENAME = '_Reg_cat_node_link.csv'

STANDARD_PIE_OPTIONS=dict(autopct='%1.1f%%', labels=None, fontsize=0, figsize=(12,12),pctdistance=0.1, radius=1.1,wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'solid'})
PARAS_COMPARE_LATEST = [
    'Flow',
    'TSS',
    'TN',
    'PN',
    'DIN',
    'DON',
    'TP',
    'PP',
    'DIP',
    'DOP',
]

progress=print
def nop(*args, **kwargs):
    pass

def read_csv(fn, *args, **kwargs):
    directory = os.path.dirname(fn)
    filename_pattern = os.path.basename(fn)
    filename_lower = filename_pattern.lower()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower() == filename_lower:
                file_path = os.path.join(root, file)
                return read_source_csv(file_path, *args, **kwargs)
    logger.warning(f"File matching '{filename_pattern}' not found in '{directory}' (case-insensitive).")
    return None

def save_figure(fn):
    os.makedirs(os.path.dirname(fn),exist_ok=True)
    plt.savefig(fn,dpi=200,bbox_inches="tight")
    plt.clf()

def load_lut(fn):
    if isinstance(fn,gpd.GeoDataFrame):
        lut = fn
    else:
        lut = gpd.read_file(fn)
    lut.rename(columns={'SUBCAT': 'ModelElement'}, inplace=True)
    lut = lut.drop(columns=['geometry'],errors='ignore')
    return lut

# def getRegionLUTPath(main_path,regionIDString):
#     # regLUTPath = main_path[0:main_path.rfind("\\")]
#     regFileIn = os.path.join(main_path,'Regions', regionIDString + REGION_LU_FILE_NAME_SUFFIX)
#     return regFileIn

def rename_constituents(df):
    df = df.copy()
    df['Constituent'] = df['Constituent'].replace(CONSTITUENT_REPORTING_NAMES)
    return df

def get_regions(loads_data):
    return list(loads_data['Region'].unique())

def filter_for_good_quality_observations(df,copy=True):
    result = df[df['RepresentivityRating'].isin(['Excellent','Good','Moderate'])]
    if copy:
        return result.copy()
    return result

def model_results_dir(main_path,region,model_run):
    base = os.path.join(main_path,region,'Model_Outputs',model_run)
    results_sets = os.listdir(base)
    if not len(results_sets):
        return None

    assert len(results_sets) == 1, f"Expected exactly one results set in {base}, found {results_sets}"
    return os.path.join(base,results_sets[0])

def load_model_results(main_path,region,model,result):
    path = model_results_dir(main_path,region,model)
    if path is None:
        return None
    fn = os.path.join(
        path,
        result)
    if not os.path.exists(fn):
        logger.warning('Run (%s/%s) exists but results file does not: %s',region,model,result)
        return None
    result = read_csv(fn,header=0)
    return result

def process_all_dfs(dfs,process):
    '''
    Run the same process (callable) on all DataFrames referenced from a set of nested
    dictionaries.
    '''
    result = {}
    for k,v in dfs.items():
        if isinstance(v,dict):
            result[k] = process_all_dfs(v,process)
        else:
            result[k] = process(v)
    return result

def to_percentage_of_whole(dfs,axis=0):
    '''
    Convert all DataFrames in a nested set of dictionaries to contain values that are a percentage
    of either the entire column (default) or row (axis=1)
    '''
    def process(df):
        if isinstance(df,float):
            return df
        return df.div(df.sum(axis=axis), axis='index' if axis==1 else 'columns') * 100
    return process_all_dfs(dfs,process)

def scale_all_dfs(dfs,scale):
    '''
    Scale all DataFrames in a nested set of dictionaries by a constant factor
    '''
    def process(df):
        return df * scale
    return process_all_dfs(dfs,process)

def concat_data_frames_at_level(dfs,level,drop=True):
    if level > 0:
        return {k: concat_data_frames_at_level(v,level-1,drop) for k,v in dfs.items()}
    
    entries = list(dfs.items())
    if not len(entries):
        return pd.DataFrame()

    if isinstance(entries[0][1],pd.DataFrame):
        result = pd.concat([v for k,v in entries],axis=1,keys=[k for k,v in entries])
        if drop:
            result = result.T.droplevel(1).T
        return result
    
    inverted = {}
    for k0,v0 in entries:
        for k1,v1 in v0.items():
            if k1 not in inverted:
                inverted[k1] = {}
            inverted[k1][k0] = v1
    return concat_data_frames_at_level(inverted,level+1,drop)

def sum_data_frames_at_level(dfs,level,sum_axis):
    if level > 0:
        return {k: sum_data_frames_at_level(v,level-1,sum_axis) for k,v in dfs.items()}

    entries = list(dfs.items())
    if not len(entries):
        return pd.DataFrame()

    # if isinstance(entries[0][1],pd.DataFrame):
    if any(isinstance(entry[1],pd.DataFrame) and (len(entry[1])>0) for entry in entries):
        dfs = [v for _,v in entries]
        if not all(isinstance(v,pd.DataFrame) for v in dfs):
            logger.warning('Not all entries at level %d are DataFrames: %s',level,[(k,type(v)) for k,v in entries])
            non_dfs = [(k,v) for k,v in entries if not isinstance(v,pd.DataFrame)]
            for k,v in non_dfs:
                logger.warning('Non-DataFrame entry: %s %s',k,str(v))
            raise ValueError('Inconsistent types at level %d'%level)
        sums = [v.sum(axis=sum_axis) for v in dfs]
        result = pd.concat(sums,axis=1,keys=[k for k,v in entries])#.T.droplevel(1).T
        return result

    inverted = {}
    for k0,v0 in entries:
        for k1,v1 in v0.items():
            if k1 not in inverted:
                inverted[k1] = {}
            inverted[k1][k0] = v1
    return sum_data_frames_at_level(inverted,level+1,sum_axis)

def create_output_directories(base,regions):
    reportcardOutputsPrefix = Path(base)

    paths = []
    for region in regions:
        logger.info("Setting up directories: %s",region)

        #Region
        outDirReg = reportcardOutputsPrefix / region
        paths.append(outDirReg)
        ## Process
        #Region
        paths.append(outDirReg / 'processBasedRegionSupply')
        paths.append(outDirReg / 'processBasedRegionExports')
        paths.append(outDirReg / 'processBasedBasinSupply')
        paths.append(outDirReg / 'processBasedBasinExports')
        paths.append(outDirReg / 'processBasedExports')

        ## Load Reduction
        outDirReduc = outDirReg / 'percentReductions'
        paths.append(outDirReduc)
        #Region
        paths.append(outDirReduc / 'processBasedRegionSupply')
        paths.append(outDirReduc / 'processBasedRegionExports')

        #Landuse
        paths.append(outDirReg / 'landuseBasedSupply')
        outDirExpLU = outDirReg / 'landuseBasedExports'
        paths.append(outDirExpLU)
        outDirSupLUReduc = outDirReg / 'percentReductions' / 'processBasedLanduseSupply'
        paths.append(outDirSupLUReduc)
        outDirExpLUReduc = outDirReg / 'percentReductions' / 'processBasedLanduseExports'
        paths.append(outDirExpLUReduc)

        ## Predev / Anthro
        outDirSupAnthro = outDirReg / 'predevAnthropogenicExports'
        paths.append(outDirSupAnthro)
        outDirExpAnthro = outDirReg / 'predevAnthropogenicExports'
        paths.append(outDirExpAnthro)

        ## Modelled vs Measured
        outDirMvM = outDirReg / 'modelledVSmeasured'
        outDirMvM_quality = outDirMvM / 'qualityData'
        outDirMvM_all = outDirMvM / 'allData'
        paths.append(outDirMvM_quality / 'ratios')
        paths.append(outDirMvM_all / 'ratios')
        paths.append(outDirMvM_quality / 'averageAnnuals' / 'bySites')
        paths.append(outDirMvM_all / 'averageAnnuals' / 'bySites')
        paths.append(outDirMvM_quality / 'averageAnnuals' / 'byConstituents')
        paths.append(outDirMvM_all / 'averageAnnuals' / 'byConstituents')
        paths.append(outDirMvM_quality / 'Moriasi')
        paths.append(outDirMvM_all / 'Moriasi')

        paths.append(outDirMvM_quality / 'annuals')
        paths.append(outDirMvM_all / 'annuals')

        ## Sankey
        paths.append(outDirReg / 'budgetExports_sankeyDiagrams')

        ## Stream
        paths.append(outDirReg / 'Stream_Length' / 'landUSeAreas_streamLengths')
        ## summary table
        paths.append(outDirReg / 'summaryTables')
        paths.append(outDirReg / 'summaryTables' / 'landUSeAreas_streamLengths')
        paths.append(outDirReg / 'summaryTables' / 'totalLoads')
        paths.append(outDirReg / 'summaryTables' / 'arealLoads')
        paths.append(outDirReg / 'variousTables')

    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
    return paths

def ensure_paths_have_files(paths):
    empty_count = 0
    for p in paths:
        if not os.path.exists(p):
            logger.warning('Previously created path does not exist: %s',p)
            continue
        if os.path.isdir(p) and not os.listdir(p):
            logger.warning('Previously created path is empty: %s',p)
            empty_count += 1
    assert empty_count == 0, f"{empty_count} paths are empty"

# def for_each_region_model()

def identify_regions_and_models(main_path):
    regions = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path,d))]
    runs = set()
    assert len(regions), f'No regions found in {main_path}'
    assert all(len(r)==2 for r in regions), f'Expected all region names to be 2 characters, found {regions}'
    for region in regions:
        model_path = os.path.join(main_path,region,'Model_Outputs')
        if not os.path.exists(model_path):
            logger.warning('No model outputs for region %s',region)
            continue
        runs.update(os.listdir(model_path))
    runs = list(runs)
    assert len(runs), f'No model runs found in {main_path}'
    scenarios = [m.split('_')[0] for m in runs]
    report_cards = set([r.split('_')[-1] for r in runs])
    assert len(report_cards) == 1, f"Expected exactly one report card type, found {report_cards}"
    return regions,runs,scenarios,list(report_cards)[0]

def read_regional_contributor(main_path,sc_region_lut):
    regions, runs, scenarios, rc = identify_regions_and_models(main_path)
    result = {}
    sc_region_lut = load_lut(sc_region_lut)

    for region in regions:
        progress (region)

        regLUT = sc_region_lut[sc_region_lut['Region'] == region]
        result[region] = {} #model: pd.DataFrame() for model in MODELs}

        for model_full, model in zip(runs,scenarios):
            progress('  ',model)

            # region_txt = regions[regions.index(region)], # ????
            RegContributorDataGrid_fil_df = load_model_results(main_path,region,model_full,'RegContributorDataGrid.csv')
            #Join the reg Cont table and the regions df
            if RegContributorDataGrid_fil_df is None:
                logger.info('No model results for %s %s',region,model)
                result[region][model] = None
                continue
            RegContributorDataGrid_fil_df = pd.merge(RegContributorDataGrid_fil_df, regLUT, on='ModelElement')
            RegContributorDataGrid_fil_df = rename_constituents(RegContributorDataGrid_fil_df)
            result[region][model] = RegContributorDataGrid_fil_df
    return result

FILL_FUS=['Bananas','Dairy','Sugarcane']
def add_columns_for_missing_fus(df):
    for fu in FILL_FUS:
        if fu not in df:
            df[fu] = np.nan

def load_basin_export_tables(regional_contributor,constituents,fus_of_interest,filter=None):
    if filter is None:
        filter = {}
    result = {}

    for region, region_data in regional_contributor.items():
        progress(region)

        region_results = result[region] = {}

        for model, data in region_data.items():
            progress('  ',model)
            model_results = region_results[model] = {}

            if data is None:
                logger.info('No regional contributor for %s %s',region,model)
                continue

            for k,v in filter.items():
                data = data[data[k]==v]
            data_summary = data.reset_index().groupby(['MU_48','Constituent','FU']).sum(numeric_only=True)
            basins = data_summary.index.levels[0]


            for basin in basins:
                progress('    ',basin)

                basin_results = model_results[basin] = {}
                data_summary2 = data_summary.loc[basin].T[constituents]
                data_summary3 = data_summary2.T

                for con in constituents:
                    progress ('      ',con)

                    basin_results[con] = data_summary3.loc[con]['LoadToRegExport (kg)']

                basin_results = model_results[basin] = pd.DataFrame(basin_results).T
                # BasinLoadToRegExport_FU[region][model][basin]['Grazing'] = BasinLoadToRegExport_FU[region][model][basin]['Grazing Forested'] + \
                #                                                            BasinLoadToRegExport_FU[region][model][basin]['Grazing Open']
                basin_results['Cropping'] = safe_sum_columns(basin_results,['Dryland Cropping','Irrigated Cropping'])
                basin_results['Urban + Other'] = safe_sum_columns(basin_results,['Urban','Other'])
                add_columns_for_missing_fus(basin_results)
                available_fus = [fu for fu in fus_of_interest if fu in basin_results.columns]
                basin_results = model_results[basin] = basin_results[available_fus].T
    return result

def safe_sum_columns(df,cols):
    cols = [col for col in cols if col in df.columns]
    if not len(cols):
        return pd.Series(0,index=df.index)
    return df[cols].sum()

def build_region_export_tables(basin_load_to_reg_export):
    result = {}

    for region, region_data in basin_load_to_reg_export.items():
        progress(region)

        result[region] = {}

        for model, base in region_data.items():
            base = basin_load_to_reg_export[region][model]
            basins = base.keys()
            progress('  ',model,basins)
            region_sum_fu = 0

            for basin in basins:
                region_sum_fu = region_sum_fu + basin_load_to_reg_export[region][model][basin].fillna(0)
                region_sum_fu

            result[region][model] = region_sum_fu
    return result

def build_overall_export_tables(regions,models,region_load_to_reg_export):
    result = {}

    for model in models:
        progress(model)
        overall_sum_FU = 0

        for region in regions:
            progress('  ',region)
            region_export = region_load_to_reg_export[region][model]
            if hasattr(region_export,'fillna'):
                region_export = region_export.fillna(0)
            overall_sum_FU = overall_sum_FU + region_export
        result[model] = overall_sum_FU
    return result

def process_based_basin_export_tables(constituents,regional_contributor,**filters):
    result = {}

    for region, region_data in regional_contributor.items():
        progress(region)

        result[region] = {}
        for model, data in region_data.items():
            progress('  ',model)

            result[region][model] = {}

            if data is None:
                logger.info('No regional contributor for %s %s',region,model)
                continue
            for k,v in filters.items():
                data = data[data[k]==v]
            # data_summary_Process = data.reset_index().groupby(['Rep_Region','Constituent','Process']).sum()
            data_summary_Process = data.reset_index().groupby(['MU_48','Constituent','Process']).sum()
            basins = data_summary_Process.index.levels[0]


            for basin in basins:
                progress ('    ',basin)

                result[region][model][basin] = {}
                data_summary2_Process = data_summary_Process.loc[basin].T[constituents]
                data_summary3_Process = data_summary2_Process.T

                for con in constituents:
                    progress ('      ',con)
                    constituent_data = pd.DataFrame(data_summary3_Process.loc[con]['LoadToRegExport (kg)'].astype('f')).T

                    process_mapping = {}
                    if con in ['TSS','PN','PP']:
                        if con =='TSS':
                            process_mapping['Hillslope'] = ['Hillslope surface soil','Undefined']
                        elif con in ['PN','PP']:
                            process_mapping['Hillslope'] = ['Hillslope no source distinction','Undefined']

                        process_mapping['Streambank'] = ['Streambank']
                        process_mapping['Gully'] = ['Gully']
                        process_mapping['ChannelRemobilisation'] = ['Channel Remobilisation']

                        process_order = ['Hillslope','Streambank','Gully','ChannelRemobilisation']

                    # elif con=='DIN':
                    else:
                        process_mapping['SurfaceRunoff'] = ['Undefined','Diffuse Dissolved','Hillslope no source distinction']

                        if 'Seepage' in constituent_data.columns:
                            process_mapping['Seepage'] = ['Seepage']
                        else:
                            process_mapping['Seepage'] = []

                        process_mapping['PointSource'] = ['Point Source']

                        process_order = ['SurfaceRunoff','Seepage','PointSource']

                    for new_process, old_processes in process_mapping.items():
                        if not len(old_processes):
                            constituent_data[new_process] = 0.0
                            continue
                        if not any(op in constituent_data.columns for op in old_processes):
                            constituent_data[new_process] = 0.0
                            continue
                        constituent_data[new_process] = constituent_data[old_processes].sum(axis=1)
                        if np.isnan(constituent_data[new_process].iloc[0]):
                            print(f"NaN found when mapping {old_processes} to {new_process} for {region} {model} {basin} {con}")
                            print(constituent_data)
                            assert False

                    result[region][model][basin][con] = constituent_data[process_order].T
    return result

def process_based_region_export_tables(regions,constituents,basin_export_by_process):
    result = {}

    for region in regions:
        progress(region)

        result[region] = {}

        for con in constituents:
            progress('  ',con)

            base = basin_export_by_process[region]['BASE']
            basins = base.keys()

            region_sum_process = 0

            for basin in basins:
                region_sum_process = region_sum_process + basin_export_by_process[region]['BASE'][basin][con].fillna(0)
                region_sum_process

            result[region][con] = region_sum_process
    return result

def process_based_overall_export(regions,constituents,region_export_by_process):
    result = {}

    for con in constituents:
        progress(con)
        overall_sum_process = 0

        for region in regions:
            progress('  ',region)
            region_data = region_export_by_process[region][con]
            if hasattr(region_data,'fillna'):
                region_data = region_data.fillna(0)
            overall_sum_process = overall_sum_process + region_data

        result[con] = overall_sum_process
    return result

def load_regional_source_sink(main_path,regions,models,rc,sc_region_lut):
    result = {}

    sc_region_lut = load_lut(sc_region_lut)
    for region in regions:
        progress (region)

        result[region] = {}
        # regLUT = sc_region_lut[sc_region_lut['Region'] == region]

        for model in models:
            progress('  ',model)
            result[region][model] = load_model_results(main_path,region,model+'_'+rc,'RegionalSourceSinkSummaryTable.csv')
    return result

def getRegion_cat_node_link_Path(main_path,regionIDString):
    regLUTPath = main_path[0:main_path.rfind("\\")]
    regFileIn = os.path.join(regLUTPath, 'Regions', regionIDString + REGION_LUT_NODE_SC_FILENAME)
    return regFileIn

def region_source_sink_summary(main_path,regions,models,rc,subcatchments):
    result = {}

    full_lut = load_lut(subcatchments)
    for region in regions:
        progress (region)

        result[region] = {}

        # regLUT = read_csv(getRegion_cat_node_link_Path(main_path,region))
        regLUT = full_lut[full_lut['Region'] == region]
        #regLUT.rename(columns={'SUBCAT': 'ModelElement'}, inplace=True)

        #REGCONTRIBUTIONDATAGRIDS[region] = {model: pd.DataFrame() for model in models}

        for model in models:
            progress(model)

            RawResult_fil_df = load_model_results(main_path,region,model + '_' + rc,'RawResults.csv')
            if RawResult_fil_df is None:
                logger.info('No model results for %s %s',region,model)
                result[region][model] = None
                continue
            #Rename Hillsope variations to just 'Hillsope' - this will make it easier when calulating Anthro loads
            RawResult_fil_df.loc[(RawResult_fil_df['BudgetElement'] == 'Hillslope surface soil') | (RawResult_fil_df['BudgetElement'] == 'Hillslope sub-surface soil')| (RawResult_fil_df['BudgetElement'] == 'Hillslope no source distinction'), 'BudgetElement'] = 'Hillslope'

            #Join the reg Cont table and the regions df
            RawResult_fil_df = pd.merge(RawResult_fil_df, regLUT, on='ModelElement')

            Regional_source_sink = pd.DataFrame(RawResult_fil_df.groupby(['MU_48','Constituent','Process','BudgetElement']).agg({'Total_Load_in_Kg':'sum'})).reset_index()

            Regional_source_sink = rename_constituents(Regional_source_sink)
            result[region][model] = Regional_source_sink

    return result


def source_sink_by_basin(regions, models, constituents,core_constituents,regional_source_sinks):
    result = {}
    for region in regions:
        progress(region)

        result[region] = {}

        for model in models:
            progress(' ',model)

            data = regional_source_sinks[region][model]
            if data is None:
                logger.info('No regional source sink for %s %s',region,model)
                result[region][model] = None
                continue
        #data_summary_Budget = data.groupby(['SummaryRegion','Constituent','BudgetElement']).sum(numeric_only = True)
            data_summary_Budget = data.groupby(['MU_48','Constituent','BudgetElement']).sum(numeric_only = True)

            basins = data_summary_Budget.index.levels[0]

            result[region][model] = {}

            for basin in basins:
                progress ('  ',basin)

                result[region][model][basin] = {}
                data_summary2_Budget = data_summary_Budget.loc[basin].T[constituents]
                data_summary3_Budget = data_summary2_Budget.T


                for con in core_constituents:
                    progress (con)

                    df_temp = data_summary3_Budget.loc[con]

                    required_elements = ['Hillslope', 'Gully', 'Undefined', 'Diffuse Dissolved', 'Seepage',
                       'Leached', 'TimeSeries Contributed Seepage', 'Extraction','Node Loss', 'Leached','Residual Node Storage',
                       'DWC Contributed Seepage', 'Link Yield', 'Link In Flow',
                       'Residual Link Storage', 'Link Initial Load', 'Streambank',
                       'Channel Remobilisation', 'Stream Deposition',
                       'Reservoir Decay','Reservoir Deposition',
                       'Flood Plain Deposition', 'Point Source', 'Stream Decay']


                # Ensure all required elements are present in the DataFrame index
                    for element in required_elements:
                        if element not in df_temp.index:
                        # Add the missing element with a Total_Load_in_Kg of 0
                            df_temp.loc[element] = 0

                    result[region][model][basin][con] = df_temp


                    if con in ['TSS','PN','PP']:
                    ### Supply
                        result[region][model][basin][con] = pd.DataFrame(result[region][model][basin][con]).T

                        if con =='TSS':
                            if region in ['CY', 'BU', 'FI', 'BM']:
                                result[region][model][basin][con]['Hillslope'] = result[region][model][basin][con]['Hillslope']
                            else:
                                result[region][model][basin][con]['Hillslope'] = result[region][model][basin][con]['Hillslope'] + \
                                                                                         result[region][model][basin][con]['Undefined']


                        elif con in ['PN','PP']:
                            result[region][model][basin][con]['Hillslope'] = result[region][model][basin][con]['Hillslope'] + \
                                                                                         result[region][model][basin][con]['Undefined']

                        result[region][model][basin][con]['Streambank'] = result[region][model][basin][con]['Streambank']
                        result[region][model][basin][con]['Gully'] = result[region][model][basin][con]['Gully']
                        result[region][model][basin][con]['ChannelRemobilisation'] = result[region][model][basin][con]['Channel Remobilisation']

                    ### loss
                        result[region][model][basin][con]['Extraction + Other Minor Losses'] = result[region][model][basin][con]['Extraction'] + \
                                                                                                           result[region][model][basin][con]['Node Loss'] + \
                                                                                                           result[region][model][basin][con]['Residual Link Storage'] + \
                                                                                                           result[region][model][basin][con]['Residual Node Storage']

                        result[region][model][basin][con]['FloodPlainDeposition'] = result[region][model][basin][con]['Flood Plain Deposition']
                        result[region][model][basin][con]['ReservoirDeposition'] = result[region][model][basin][con]['Reservoir Deposition']
                        result[region][model][basin][con]['StreamDeposition'] = result[region][model][basin][con]['Stream Deposition']

                        processes_of_interest = ['Hillslope','Streambank','Gully','ChannelRemobilisation',
                                          'Extraction + Other Minor Losses','FloodPlainDeposition','ReservoirDeposition','StreamDeposition']

                    elif con=='DIN':
                    ### Supply
                        result[region][model][basin][con] = pd.DataFrame(result[region][model][basin][con]).T

                        if region in ['CY', 'BU', 'BM']:
                            result[region][model][basin][con]['SurfaceRunoff'] = result[region][model][basin][con]['Diffuse Dissolved'] + \
                                                                                             result[region][model][basin][con]['Hillslope']
                        elif region == 'FI':
                            result[region][model][basin][con]['SurfaceRunoff'] = result[region][model][basin][con]['Diffuse Dissolved']

                        else:
                            result[region][model][basin][con]['SurfaceRunoff'] = result[region][model][basin][con]['Undefined'] + \
                                                                                             result[region][model][basin][con]['Diffuse Dissolved'] + \
                                                                                             result[region][model][basin][con]['Hillslope']


                        result[region][model][basin][con]['PointSource'] = result[region][model][basin][con]['Point Source']

                        if 'Seepage' in result[region][model][basin][con].columns:
                            result[region][model][basin][con]['Seepage'] = result[region][model][basin][con]['Seepage']
                        else:
                            result[region][model][basin][con]['Seepage'] = 0


                    ### loss
                        result[region][model][basin][con]['Extraction + Other Minor Losses'] = result[region][model][basin][con]['Extraction'] + \
                                                                                                           result[region][model][basin][con]['Node Loss'] + \
                                                                                                           result[region][model][basin][con]['Residual Link Storage'] + \
                                                                                                           result[region][model][basin][con]['Residual Node Storage']


                        processes_of_interest = ['SurfaceRunoff','Seepage','PointSource',
                                          'Extraction + Other Minor Losses']

                    result[region][model][basin][con] = result[region][model][basin][con][processes_of_interest].T
    return result

def source_sink_by_region(core_constituents,basin_source_sink):
    result = {}
    regions = basin_source_sink.keys()
    for region in regions:
        progress(region)
        result[region] = {}
        for con in core_constituents:
            progress(' ',con)
            base = basin_source_sink[region]['BASE']
            basins = base.keys()

            Region_sum = 0

            for basin in basins:
                progress('  ',basin)
                Region_sum = Region_sum + basin_source_sink[region]['BASE'][basin][con].fillna(0)
                Region_sum

                result[region][con] = Region_sum
    return result

def overall_source_sink_budget(regions,core_constituents,regional_source_sink):
    result = {}

    for con in core_constituents:
        progress(con)
        overall_sum_process = 0

        for region in regions:
            # progress(region)
            overall_sum_process = overall_sum_process + regional_source_sink[region][con].fillna(0)
            result[con] = overall_sum_process

    return result

def lu_area_by_region(main_path,fus):
    regions,runs,scenarios,rc = identify_regions_and_models(main_path)
    base_scenario = [m for m in runs if m.startswith('BASE')][0]
    result = {}

    for region in regions:
        progress(region)

        fuAreaTable = load_model_results(main_path,region,base_scenario,'fuAreasTable.csv')

        fuArea_summary = fuAreaTable.groupby(['FU']).sum(numeric_only= True)
        fuArea = fuArea_summary.T
        #fuArea['Grazing'] = fuArea['Grazing Forested'] + fuArea['Grazing Open']
        fuArea['Cropping'] = fuArea['Dryland Cropping'] + fuArea['Irrigated Cropping']
        fuArea['Urban + Other'] = fuArea['Other'] + fuArea['Urban']
        for fu in FILL_FUS:
            if fu not in fuArea:
                fuArea[fu] = np.nan

        fuArea = fuArea
        fuArea = fuArea[fus]
        fuArea = fuArea.T

        result[region] = fuArea

    return result

def stream_lengths(main_path,regions,region_names,rc):
    result = []

    for region in regions:
        progress(region)
        parameterTable = load_model_results(main_path,region,'BASE_'+rc,'ParameterTable.csv')
        # paramterTable_fil = modelResultsPrefix + REGIONs[REGIONs.index(region)] + '//Model_Outputs//'  + 'BASE' + '_' + RC + '//ParameterTable.csv'
        # paramterTable = read_csv(paramterTable_fil,header=0,usecols=[0,1,2,3,4,5,6])

        lengths = pd.DataFrame(parameterTable[parameterTable['PARAMETER']=='Link Length']['VALUE'])
        lengths['VALUE'] = pd.to_numeric(lengths.VALUE)
        totalLength = lengths.sum()*c.M_TO_KM

        result.append(totalLength)

    result = pd.DataFrame(result)
    result = result.T
    result[OVERALL_REGION] = result.T.sum()
    result = result.T
    result.index = region_names
    result = result.fillna(0).astype(int)
    result.columns = ['Stream']
    return result

def fu_load_summary(output_path,regions,region_names,core_constituents,region_export_by_fu,years):
    result = {}

    for con in core_constituents[0:4]:
        progress(con)

        con_results = pd.DataFrame([])
        valid_regions = []
        for ix,region in enumerate(regions):
            progress(' ',region)
            region_data = region_export_by_fu[region]['BASE']
            if region_data is None or isinstance(region_data,int):
                logger.info('No regional export by FU for %s',region)
                continue
            valid_regions.append(region_names[ix])
            con_data = region_data[con]
            if con == 'TSS':
                scale = c.KG_TO_KTONS
            else:
                scale = c.KG_TO_TONS
            FU_Load_summary_con_temp = pd.DataFrame(con_data).T*scale/years

            con_results = pd.concat([con_results,pd.DataFrame(FU_Load_summary_con_temp)])

        con_results = con_results.T
        con_results[OVERALL_REGION] = con_results.T.sum()
        con_results = con_results.T
        con_results.index = valid_regions + [region_names[-1]]
        con_results = con_results.fillna(0).round(2)

        if con == 'TSS':
            unit = 'kt/year'
        else:
            unit = 't/year'
        con_results = con_results.rename(columns = lambda c: f'{c} ({unit})')
        con_results.to_csv(os.path.join(output_path, OVERALL_REGION, 'summaryTables', 'totalLoads', f'totalLoadperYear_{con}.csv'))

        result[con] = con_results
    return result

def fu_areal_load_summary(output_path,fu_load,fu_areas):
    result = {}

    for constituent,df in fu_load.items():
        progress(constituent)

        # FU_Load_summary_con = FU_Load_summary[constituent]
        if constituent == 'TSS':
            scale = c.KTONS_TO_TONS
        else:
            scale = c.TONS_TO_KG

        valid_fu_areas = fu_areas[fu_areas.index.isin(df.index)]

        FU_arealLoad_summary_con = df.values*scale/valid_fu_areas.values
        FU_arealLoad_summary_con = pd.DataFrame(FU_arealLoad_summary_con)
        FU_arealLoad_summary_con.index = df.index
        FU_arealLoad_summary_con = FU_arealLoad_summary_con.fillna(0).round(3)

        if constituent == 'TSS':
            units = 't'
        else:
            units = 'kg'
        # if df.columns[0].startswith('Banana'):
        #     progress('Before rename to include units/ha/yr')
        #     progress(df.columns)
        # new_columns = [f'{col} ({units}/{base_unit(col)}/year)' for col in df.columns]
        # TODO Must be a nicer way to handle this!
        FU_arealLoad_summary_con.columns = [f'Banana ({units}/ha/year)', f'Conservation ({units}/ha/year)',f'Cropping ({units}/ha/year)',
                                f'Dairy ({units}/ha/year)',f'Forestry ({units}/ha/year)',f'Grazing ({units}/ha/year)',f'Horticulture ({units}/ha/year)',
                                f'Stream ({units}/km/year)',f'Sugarcane ({units}/ha/year)', f'Urban + Other ({units}/ha/year)']


        FU_arealLoad_summary_con.to_csv(os.path.join(output_path, OVERALL_REGION, 'summaryTables', 'arealLoads', f'arealLoadperYear_{constituent}.csv'))

        result[constituent] = FU_arealLoad_summary_con
    return result

def plot_export_contribution_by_region(output_path,fu_load_summary,constituents):
    for con in constituents:
        plt.clf()
        summaryExport_overall = fu_load_summary[con].T.sum()[:-1]

        ax = summaryExport_overall.plot.pie(**STANDARD_PIE_OPTIONS)

        percent = 100.*summaryExport_overall.values/summaryExport_overall.values.sum()
        labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(summaryExport_overall.index, percent)]

        ax.legend(labels, bbox_to_anchor=(0.95,0.61),ncol=1,fontsize=14)
        ax.set(ylabel='')
        dest = Path(output_path) / OVERALL_REGION / 'variousExports'
        dest.mkdir(parents=True, exist_ok=True)
        save_figure(dest / ('exportContributionByRegion_' + con + '.png'))

def plot_tss_tonnages(output_path,fu_load_summary,fu_area_summary,constituent='TSS'):
    PROs = ['Grazing','Stream']

    for p in PROs:
        progress(p)

        plt.clf()

        if p == 'Grazing':
            pro_load = 'Grazing (kt/year)'
            pro_area = 'Grazing (ha)'
        elif p == 'Stream':
            pro_load = 'Stream (kt/year)'
            pro_area = 'Stream (km)'

        if constituent == 'TSS':
            scale = c.KTONS_TO_TONS
        else:
            scale = c.TONS_TO_KG
        ArealRegionLoad = fu_load_summary[constituent][pro_load].T*scale/fu_area_summary[pro_area]

        #progress (ArealRegionLoad)

        ax1 = fu_load_summary[constituent][pro_load].T.plot(kind='bar',color='m',width=0.4,grid="off",edgecolor='black')
        ax1.set_xlabel('')
        ax1.grid(False)


        ax2 = ax1.twinx()
        ax2 = ArealRegionLoad.plot(kind='bar',color='g',width=0.15,grid="off",edgecolor='black')
        ax2.set_xlabel('')
        ax2.grid(False)

        if constituent == 'Flow':
            ax1.set_ylabel(constituent + ' (GL/yr OR ML/yr ???)',size=8)
            ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
            ax2.set_ylabel(constituent + ' (GL/yr OR ML/yr ???)',size=8)
            ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
        elif constituent == 'TSS':
            ax1.set_ylabel(constituent + ' (kt/yr)',size=8)
            ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
            if p == 'Grazing':
                ax2.set_ylabel(constituent + ' (t/ha/yr)',size=8)
                ax2.legend(['t/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            else:
                ax2.set_ylabel(constituent + ' (t/km/yr)',size=8)
                ax2.legend(['t/km/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
        elif constituent in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            ax1.set_ylabel(constituent + ' (t/yr)',size=8)
            ax1.legend(['t/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
            ax2.set_ylabel(constituent + ' (kg/ha/yr)',size=8)
            ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
        else:
            ax1.set_ylabel(constituent + ' (kg/yr)',size=8)
            ax1.legend(['kg/yr'],loc='lower left',bbox_to_anchor=(0.05,1.0),fontsize=8)
            ax2.set_ylabel(constituent + ' (kg/ha/yr)',size=8)
            ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)

        save_figure(os.path.join(output_path,OVERALL_REGION,'variousExports', constituent + '_exportTonnageArealsbyRegion_' + p + '.png'))

def plot_sugarcane_tonnages(output_path,fu_load_summary,fu_area_summary,constituent='DIN'):
    PROs = ['Sugarcane']

    for p in PROs:
        progress(p)

        progress(constituent)

        plt.clf()

        if p == 'Sugarcane':
            pro_load = 'Sugarcane (t/year)'
            pro_area = 'Sugarcane (ha)'
        else:
            pro_load = None
            pro_area = None

        if constituent == 'DIN':
            scale = c.KTONS_TO_TONS
        else:
            scale = c.TONS_TO_KG
        ArealRegionLoad = fu_load_summary[constituent][pro_load].T*scale/fu_area_summary[pro_area]

        #progress (ArealRegionLoad)

        ax1 = fu_load_summary[constituent][pro_load].T.plot(kind='bar',color='m',width=0.4,grid="off",edgecolor='black')
        ax1.set_xlabel('')
        ax1.grid(False)

        ax2 = ax1.twinx()
        ax2 = ArealRegionLoad.plot(kind='bar',color='g',width=0.15,grid="off",edgecolor='black')
        ax2.set_xlabel('')
        ax2.grid(False)

        if constituent == 'Flow':
            ax1.set_ylabel(constituent  + ' (GL/yr OR ML/yr ???)',size=8)
            ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
            ax2.set_ylabel(constituent + ' (GL/yr OR ML/yr ???)',size=8)
            ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
        elif constituent == 'TSS':
            ax1.set_ylabel(constituent + ' (kt/yr)',size=8)
            ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
            if p == 'Grazing':
                ax2.set_ylabel(constituent + ' (t/ha/yr)',size=8)
                ax2.legend(['t/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            else:
                ax2.set_ylabel(constituent + ' (t/km/yr)',size=8)
                ax2.legend(['t/km/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
        elif constituent in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            ax1.set_ylabel(constituent + ' (t/yr)',size=8)
            ax1.legend(['t/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
            ax2.set_ylabel(constituent + ' (kg/ha/yr)',size=8)
            ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
        else:
            ax1.set_ylabel(constituent + ' (kg/yr)',size=8)
            ax1.legend(['kg/yr'],loc='lower left',bbox_to_anchor=(0.05,1.0),fontsize=8)
            ax2.set_ylabel(constituent + ' (kg/ha/yr)',size=8)
            ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)

        save_figure(os.path.join(output_path,OVERALL_REGION,'variousExports', constituent + '_exportTonnageArealsbyRegion_' + p + '.png'))



def plot_process_contributions(output_path,regions,contituents,basin_export_by_process,years,show_plot=True,save_plot=True, basin_renames=None):
    if basin_renames is None:
        basin_renames = {}
    ### Process based contribution plotting - both Tonnage and % CONTRIBUTION for TSS, PN, PP, & DIN
    ### ONLY the Base model is considered
    # was_interactive = plt.isinteractive()
    # if show_plot and not was_interactive:
    #     plt.ion()
    # elif not show_plot and was_interactive:
    #     plt.ioff()

    for con in contituents:
        progress(con)

        basins_list = pd.DataFrame([])
        basins_process_summary = pd.DataFrame([])

        def deal_with_plot(region,plot_prefix,csv_prefix=None,csv_data=None):
            if show_plot:
                plt.show()
            if save_plot:
                save_figure(os.path.join(output_path, region, 'processBasedExports', plot_prefix+'_' + con + '.png'))   
                if csv_prefix is not None and csv_data is not None:
                    csv_data.to_csv(os.path.join(output_path, region, 'variousTables', csv_prefix + '_' + con + '.csv'))

        missing_regions = set()
        for region in regions:
            progress('  ',region)

            base = basin_export_by_process[region]['BASE']

            basins = base.keys()

            basin_process_summary = []
            basin_list = []

            if not len(basins):
                logger.info('No data for %s',region)
                missing_regions.add(region)
                continue

            for basin in basins:
                progress('    ',basin)

                basin_list.append(basin)
                basin_process_summary.append(base[basin][con]['LoadToRegExport (kg)']/years)

            if con == 'TSS':
                scale = c.KG_TO_KTONS
            elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
                scale = c.KG_TO_TONS
            else:
                scale = 1.0
            basin_process_summary = pd.DataFrame(basin_process_summary)*scale

            basins_list = pd.concat([basins_list,pd.DataFrame(basin_list)])
            basins_process_summary = pd.concat([basins_process_summary,basin_process_summary])

            basin_list = pd.DataFrame(basin_list)
            #basin_list.columns = ['Region Basin']
            basin_list.columns = ['Management Units']
            #progress(basin_list)

            processSummary_Region = basin_process_summary.copy()

            basin_list['Management Units'] = basin_list['Management Units'].replace(basin_renames)
            if region == 'WT':
                processSummary_Region['Management Units'] = basin_list['Management Units'].values
            # #     # Set 'Management Units' as the index
                processSummary_Region = processSummary_Region.reset_index(drop=True)  # Reset the index (remove 'Process')
                processSummary_Region = processSummary_Region.set_index('Management Units')
                #processSummary_Region = processSummary_Region.reindex(['Daintree', 'Mossman', 'Barron', 'Mulgrave-Russell', 'Johnstone', 'Tully', 'Murray', 'Lower Herbert', 'Upper Herbert'])
            else:
                processSummary_Region['Management Units'] = basin_list['Management Units'].values
            # #     # Set 'Management Units' as the index
                processSummary_Region = processSummary_Region.reset_index(drop=True)  # Reset the index (remove 'Process')
                processSummary_Region = processSummary_Region.rename_axis(None, axis=1)
                processSummary_Region = processSummary_Region.set_index('Management Units')

            # b = processSummary_Region.T

            # processSummary_Region = b.T
            processSummary_Region = processSummary_Region[::-1]

            processSummary_Region['total'] = processSummary_Region.sum(axis=1)
            percent_region = processSummary_Region.div(processSummary_Region.total, axis='index') * 100
            percent_region

            #########################################
            ### process based tonnage contribution ##
            #########################################

            axx = processSummary_Region.T[0:-1].T.plot(kind='barh',fontsize = 14, stacked=True,color=['green','blue','pink','cyan'],edgecolor='black')

            if con == 'Flow':
                axx.set_xlabel(con + ' (GL/yr OR ML/yr ???)',size=12)
            elif con == 'TSS':
                axx.set_xlabel(con + ' Load (kt/yr)',size=12)
            elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
                axx.set_xlabel(con + ' Load (t/yr)',size=12)
            else:
                axx.set_xlabel(con + ' Load (kg/yr)',size=12)

            axx.set_ylabel('', size=14)

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.1,left=0.06,top=2.945,right=0.99)
            #axx = fig.add_subplot(111)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),fancybox=True, shadow=True, ncol=4, fontsize = 12)
            deal_with_plot(region,'processTonnage')

            ########################################
            ###### process based % contribution #####
            ########################################

            axx = percent_region.T[0:-1].T.plot(kind='barh',fontsize = 14, stacked=True,color=['green','blue','pink','cyan'],edgecolor='black')
            axx.set_xlim([0,100])

            axx.set_xlabel(con + ' Load Contribution (%)',size=12)

            axx.set_ylabel('', size=14)

            axx.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.1,left=0.06,top=2.945,right=0.99)
            # axx = fig.add_subplot(111)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),fancybox=True, shadow=True, ncol=4, fontsize = 12)

            deal_with_plot(region,'processPercent')

        if len(missing_regions):
            logger.warning('No data for %s. Cannot produce overall chart',', '.join(missing_regions))
            continue
            
        basins_list.columns = [f'{OVERALL_REGION} Management Units']
        basins_list[f'{OVERALL_REGION} Management Units'] = basins_list[f'{OVERALL_REGION} Management Units'].replace(basin_renames)

        processSummary = basins_process_summary.copy()

        processSummary[f'{OVERALL_REGION} Management Units'] = basins_list[f'{OVERALL_REGION} Management Units'].values
        #
        processSummary = processSummary.reset_index(drop=True)  # Reset the index (remove 'Process')
        processSummary = processSummary.rename_axis(None, axis=1)
        processSummary = processSummary.set_index(f'{OVERALL_REGION} Management Units')

        a = processSummary.T
        # a.insert(loc = 7 , column = '', value = 0)
        # a.insert(loc = 16 , column = ' ', value = 0)
        # a.insert(loc = 22 , column = '  ', value = 0)
        # a.insert(loc = 27 , column = '   ', value = 0)
        # a.insert(loc = 34 , column = '    ', value = 0)


        a.insert(loc = 7 , column = '', value = 0)
        a.insert(loc = 17 , column = ' ', value = 0)
        a.insert(loc = 29 , column = '  ', value = 0)
        a.insert(loc = 34 , column = '   ', value = 0)
        a.insert(loc = 47 , column = '    ', value = 0)

        processSummary = a.T
        processSummary = processSummary[::-1]

        if con == 'TSS':
            processSummary['export(kt)/year'] = processSummary.sum(axis=1)
        elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            processSummary['export(t)/year'] = processSummary.sum(axis=1)
        else:
            processSummary['export(kg)/year'] = processSummary.sum(axis=1)


        ### estimate % contribution
        if con == 'TSS':
            percent = processSummary.div(processSummary['export(kt)/year'], axis='index') * 100
        elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            percent = processSummary.div(processSummary['export(t)/year'], axis='index') * 100
        else:
            percent = processSummary.div(processSummary['export(kg)/year'], axis='index') * 100

        #########################################
        ### process based tonnage contribution ##
        #########################################
        axx = processSummary.T[0:-1].T.plot(kind='barh',fontsize = 10, stacked=True,color=['green','blue','pink','cyan'],edgecolor='black')

        if con == 'Flow':
            axx.set_xlabel(con + ' (GL/yr OR ML/yr ???)',size=12)
        elif con == 'TSS':
            axx.set_xlabel(con + ' Load (kt/yr)',size=12)
        elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            axx.set_xlabel(con + ' Load (t/yr)',size=12)
        else:
            axx.set_xlabel(con + ' Load (kg/yr)',size=12)

        def setup_axis(ax):
            ax.set_ylabel('', size=14)

            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            ax.text(ax.get_xlim()[1]*-0.27, 46.5, "Cape York", size=12,rotation=90, weight="bold")
            ax.text(ax.get_xlim()[1]*-0.27, 37.75, "Wet Tropics", size=12,rotation=90, weight="bold")
            ax.text(ax.get_xlim()[1]*-0.27, 27.75, "Burdekin", size=12,rotation=90, weight="bold")
            ax.text(ax.get_xlim()[1]*-0.27, 18.5, '    Mackay \n Whitsunday', size=12,rotation=90, weight="bold")
            ax.text(ax.get_xlim()[1]*-0.27, 12.25, "Fitzroy", size=12,rotation=90, weight="bold")
            ax.text(ax.get_xlim()[1]*-0.27, 0.25, "Burnett Mary", size=12,rotation=90, weight="bold")

            ax.axhline(y=45, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
            ax.axhline(y=35, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
            ax.axhline(y=23, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
            ax.axhline(y=18, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
            ax.axhline(y=5, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
        setup_axis(axx)

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.1,left=0.06,top=2.945,right=0.99)
        #axx = fig.add_subplot(111)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),fancybox=True, shadow=True, ncol=4, fontsize = 10)

        deal_with_plot(OVERALL_REGION,'processTonnage','exportTonnage',pd.DataFrame(processSummary[::-1].iloc[:,-1]))


    # #     clf()

    #     #########################################
    #     ###### process based % contribution #####
    #     #########################################

        axx = percent.T[0:-1].T.plot(kind='barh',fontsize = 10, stacked=True,color=['green','blue','pink','cyan'],edgecolor='black')
        axx.set_xlim([0,100])

        axx.set_xlabel(con + ' Load Contribution (%)',size=12)

        setup_axis(axx)

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.1,left=0.06,top=2.945,right=0.99)
        #axx = fig.add_subplot(111)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),fancybox=True, shadow=True, ncol=4, fontsize = 10)

        deal_with_plot(OVERALL_REGION,'processPercent')


def plot_land_use_exports(output_path,regions,constituents,region_export_by_fu,region_fu_area,overall_export_by_fu,years):
    for region in regions:
        progress(region)

        RegionLoad = region_export_by_fu[region]
        RegionLoad = RegionLoad['BASE']
        if RegionLoad is None or isinstance(RegionLoad,int):
            logger.info('No regional export by FU for %s',region)
            continue
        RegionLoad = RegionLoad[constituents]
        unit_conversions = [c.KG_TO_KTONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS][:len(constituents)]
        RegionLoad_export = RegionLoad*unit_conversions/years #unit conversion
        RegionFUArea = region_fu_area[region]
        RegionFUArea = RegionFUArea*c.M2_TO_HA  #unit conversion

        for con in constituents:
            progress(con)

            plt.clf()

            if con == 'TSS':
                ArealRegionLoad = RegionLoad[con]*c.KG_TO_TONS/RegionFUArea['Area']/years
            else:
                ArealRegionLoad = RegionLoad[con]/RegionFUArea['Area']/years

            #progress (ArealRegionLoad)

            ax1 = RegionLoad_export[con].plot(kind='bar',color='m',width=0.4,grid="off",edgecolor='black')
            ax1.set_xlabel('')
            ax1.grid(False)

            ax2 = ax1.twinx()
            ax2 = ArealRegionLoad.plot(kind='bar',color='g',width=0.15,grid="off",edgecolor='black')
            ax2.set_xlabel('')
            ax2.grid(False)


            if con == 'Flow':
                ax1.set_ylabel(con + ' (GL/yr OR ML/yr ???)',size=8)
                ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
                ax2.set_ylabel(con + ' (GL/yr OR ML/yr ???)',size=8)
                ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            elif con == 'TSS':
                ax1.set_ylabel(con + ' (kt/yr)',size=8)
                ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
                ax2.set_ylabel(con + ' (t/ha/yr)',size=8)
                ax2.legend(['t/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
                ax1.set_ylabel(con + ' (t/yr)',size=8)
                ax1.legend(['t/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
                ax2.set_ylabel(con + ' (kg/ha/yr)',size=8)
                ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            else:
                ax1.set_ylabel(con + ' (kg/yr)',size=8)
                ax1.legend(['kg/yr'],loc='lower left',bbox_to_anchor=(0.05,1.0),fontsize=8)
                ax2.set_ylabel(con + ' (kg/ha/yr)',size=8)
                ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)

            save_figure(os.path.join(output_path,region,'landuseBasedExports',con + '_exportTonnageArealsbyLanduse.png'))


            ### Estimating and Plotting TSS, PN, PP, DIN Contribution by Landuse

            contribution = RegionLoad[constituents]
            contribution = contribution/contribution.sum()*100

            progress(contribution)

            ax3 = contribution.T.plot(kind='bar',width=0.6,grid="off",color=FU_COLORS)

            ax3.legend(bbox_to_anchor=(1,1),fontsize=6.4,ncol=5)
            ax3.set_ylim(0,100)
            ax3.set_ylabel("Percent Contribution",size=8)
            ax3.set_xticklabels(['TSS','PN','PP','DIN'],rotation='horizontal')
            ax3.set_xlabel('')


            progress(contribution)

            save_figure(os.path.join(output_path,region,'landuseBasedExports','exportPercentByLanduse.png'))

    for con in constituents:
        plt.clf()
        colors = FU_COLORS

        ax = overall_export_by_fu['BASE'][con].plot.pie(colors=colors,**STANDARD_PIE_OPTIONS)

        percent = 100.*overall_export_by_fu['BASE'][con].values/overall_export_by_fu['BASE'][con].values.sum()
        labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(overall_export_by_fu['BASE'].index, percent)]


        ax.legend(labels, bbox_to_anchor=(0.95,0.67),ncol=1,fontsize=14)  #(1,1.1)
        ax.set(ylabel='')

        save_figure(os.path.join(output_path,OVERALL_REGION,'variousExports','exportContributionByLanduse_' + con + '.png'))















# migrated by copilot
def collate_measured_annual_regions(loads_data):
    region_list = get_regions(loads_data)
    measured_annual_regions = {region: {} for region in region_list}
    for region in region_list:
        sites_of_interest = loads_data[loads_data['Region'] == region][['Site Code', 'Node']]
        measured_sites = sites_of_interest
        measured_annual_sites = {}
        for _,row in measured_sites.iterrows():
            node_name = row['Node']
            site = row['Site Code']

            columns = ['RepresentivityRating', 'Flow', 'TSS', 'TN', 'PN', 'NOX', 'NH4', 'DIN', 'DON', 'TP', 'PP', 'DIP', 'DOP']
            annual_load_temp = loads_data[loads_data['Node'] == node_name]
            annual_load_temp = annual_load_temp.set_index(['Year'])
            annual_load_temp = annual_load_temp[columns]
            measured_annual_sites[site] = pd.DataFrame(annual_load_temp)
        measured_annual_regions[region] = measured_annual_sites
    return measured_annual_regions

# migrated by copilot
def build_annual_comparison_by_quality(region_list, measured_annual_regions, site_list, paras_compare, constituent_names):
    logger.warning('build_annual_comparison_by_quality may not work as intended')
    quality = ['yes', 'no']
    summary_annual_regions_quality = {q: {} for q in quality}
    for q in quality:
        result_df_annual = pd.DataFrame()
        summary_annual_regions = {region: pd.DataFrame() for region in region_list}
        for region in region_list:
            measured_sites = site_list[site_list['Region'] == region]['Site Code'].tolist()
            modelled_sites = site_list[site_list['Region'] == region]['Node'].tolist()
            for idx, con in enumerate(paras_compare):
                for idx_site, site in enumerate(measured_sites):
                    Node_in_source = modelled_sites[idx_site]
                    site_monitoring_data = measured_annual_regions[region][site].reset_index()
                    site_monitoring_data['Node'] = site
                    if con == 'Flows':
                        site_monitoring_data_TS = site_monitoring_data[['Node', 'Year', 'RepresentivityRating', 'Flow']]
                    else:
                        site_monitoring_data_TS = site_monitoring_data[['Node', 'Year', 'RepresentivityRating', constituent_names[idx]]]
                    if q == 'yes':
                        site_monitoring_data_TS = filter_for_good_quality_observations(site_monitoring_data_TS, copy=True)
                        if site_monitoring_data_TS['Year'].count() < 3:
                            continue
                    site_monitoring_data_TS = site_monitoring_data_TS.copy()
                    column_mapping = {
                        'Node': 'Site',
                        'Year': 'Year',
                        'RepresentivityRating': 'RepresentivityRating',
                        site_monitoring_data_TS.columns[3]: 'Monitored'
                    }
                    site_monitoring_data_TS.rename(columns=column_mapping, inplace=True)
                    site_monitoring_data_TS['Monitored'] = round(site_monitoring_data_TS['Monitored'], 1)
                    # Modelled data loading and processing would go here (if needed for comparison)
                    # Append to result
                    result_df_annual = pd.concat([result_df_annual, site_monitoring_data_TS], ignore_index=True)
            summary_annual_regions[region] = result_df_annual
        summary_annual_regions_quality[q] = summary_annual_regions
    return summary_annual_regions_quality

# migrated by copilot
def collate_modelled_annual_regions(main_path,site_list,comparison_run):
    region_list = get_regions(site_list)
    paras_compare = ['Sediment - Fine','N_Particulate','N_DIN','N_DON','P_Particulate','P_FRP','P_DOP','Flows']
    paras_compare_suffixes = ['_cubicmetrespersecond','_kilograms']
    modelled_daily_regions = {region: {} for region in region_list}
    modelled_annual_regions = {region: {} for region in region_list}
    def water_year(date_string):
        y, m, d = [int(c) for c in date_string.split('-')]
        if m <= 6:
            return f'{y-1}-{y}'
        return f'{y}-{y+1}'
    for region in region_list:
        region_data = site_list[site_list['Region'] == region]
        measured_sites = region_data['Site Code'].tolist()
        modelled_sites = region_data['Node'].tolist()
        modelled_daily_sites = {site: pd.DataFrame() for site in measured_sites}
        modelled_annual_sites = {site: pd.DataFrame() for site in measured_sites}
        for site, site_code in zip(modelled_sites, measured_sites):
            if site is None:
                logger.error('Site is None in region %s. site code: %s', region, site_code)
                continue
                # raise ValueError('Site is None')
            # Build file paths
            def read_constituent(constituent):
                if constituent == 'Flows':
                    suffix = paras_compare_suffixes[0]
                    c_fn = 'Flow'
                else:
                    suffix = paras_compare_suffixes[1]
                    c_fn = constituent
                fn = os.path.join(model_results_dir(main_path, region, comparison_run), 'TimeSeries', constituent, f"{c_fn}_{site}{suffix}.csv")
                return read_csv(fn, header=0)

            TSS = read_constituent(paras_compare[0])
            PN = read_constituent(paras_compare[1])
            DIN = read_constituent(paras_compare[2])
            DON = read_constituent(paras_compare[3])
            PP = read_constituent(paras_compare[4])
            DIP = read_constituent(paras_compare[5])
            DOP = read_constituent(paras_compare[6])
            Flow = read_constituent(paras_compare[7])
            if any(df is None for df in [TSS, PN, DIN, DON, PP, DIP, DOP, Flow]):
                logger.error('One of the constituent files is missing for site %s in region %s', site, region)
                continue
            # Assign variables
            Date = np.array(Flow['Date'])
            Flow_arr = np.array(Flow[Flow.columns[1]]) * c.CUMECS_TO_ML_DAY
            TSS_arr = np.array(TSS[TSS.columns[1]]) * c.KG_TO_TONS
            PN_arr = np.array(PN[PN.columns[1]]) * c.KG_TO_TONS
            PP_arr = np.array(PP[PP.columns[1]]) * c.KG_TO_TONS
            DIN_arr = np.array(DIN[DIN.columns[1]]) * c.KG_TO_TONS
            DON_arr = np.array(DON[DON.columns[1]]) * c.KG_TO_TONS
            DIP_arr = np.array(DIP[DIP.columns[1]]) * c.KG_TO_TONS
            DOP_arr = np.array(DOP[DOP.columns[1]]) * c.KG_TO_TONS
            # Combine all daily parameters
            modelled_daily = np.array([Date, Flow_arr, TSS_arr, PN_arr, DIN_arr, DON_arr, PP_arr, DIP_arr, DOP_arr]).T
            modelled_daily = pd.DataFrame(modelled_daily, columns=['Date', 'Flow', 'TSS', 'PN', 'DIN', 'DON', 'PP', 'DIP', 'DOP'])
            modelled_daily = modelled_daily.set_index(['Date'])
            # Insert TN & TP
            modelled_daily.insert(loc=2, column='TN', value=modelled_daily['PN'] + modelled_daily['DIN'] + modelled_daily['DON'])
            modelled_daily.insert(loc=6, column='TP', value=modelled_daily['PP'] + modelled_daily['DIP'] + modelled_daily['DOP'])
            # Water years
            water_years = modelled_daily.index.map(water_year)
            modelled_annual = modelled_daily.groupby(water_years).sum()
            measured_site_name = measured_sites[list(modelled_sites).index(site)]
            modelled_daily_sites[measured_site_name] = modelled_daily
            modelled_annual_sites[measured_site_name] = modelled_annual
        modelled_daily_regions[region] = modelled_daily_sites
        modelled_annual_regions[region] = modelled_annual_sites
    return modelled_daily_regions, modelled_annual_regions

def compute_predev_vs_anthropogenic(basin_load_to_reg_export_fu, region, constituent,years,development_scenario='BASE'):
    all_exports = basin_load_to_reg_export_fu[region]
    if 'PREDEV' not in all_exports:
        logger.info('No pre development data for %s',region)
        return None, None

    predev = all_exports['PREDEV']
    if len(predev)==0:
        logger.info('No pre development data for %s',region)
        return None, None

    base = all_exports[development_scenario]
    if len(base)==0:
        logger.info('No %s data for %s', development_scenario, region)
        return None, None

    basins = base.keys()
    basin_list = []
    basin_predev_summary = []
    basin_base_summary = []
    basin_anthro_summary = []
    for basin in basins:
        basin_list.append(basin)
        basin_predev_summary.append(predev[basin][constituent].sum()/years)
        basin_base_summary.append(base[basin][constituent].sum()/years)
        basin_anthro_summary.append(base[basin][constituent].sum()/years - predev[basin][constituent].sum()/years)
    basin_list = pd.DataFrame(basin_list, columns=['basins'])

    # Unit conversion
    basin_predev_summary = pd.DataFrame(basin_predev_summary)
    basin_base_summary = pd.DataFrame(basin_base_summary)
    basin_anthro_summary = pd.DataFrame(basin_anthro_summary)

    anthro_vs_predev = pd.concat([basin_predev_summary, basin_anthro_summary], axis=1)
    anthro_vs_predev.columns = ['Predevelopment', 'Anthropogenic']
    anthro_vs_predev.index = basin_list['basins']
    predev_vs_base_vs_change = pd.concat([basin_predev_summary, basin_base_summary], axis=1)
    predev_vs_base_vs_change.columns = ['Pre-development', 'Base']
    predev_vs_base_vs_change.index = basin_list['basins']
    total = pd.DataFrame(predev_vs_base_vs_change.sum())#.T
    # progress(total)
    total.columns = ['Total']
    predev_vs_base_vs_change = pd.concat([predev_vs_base_vs_change, total.T], ignore_index=False)
    if (anthro_vs_predev.sum().sum() == 0) and (predev_vs_base_vs_change.sum().sum() == 0):
        return None, None

    return anthro_vs_predev, predev_vs_base_vs_change

def compute_anthropogenic_summary(basin_load_to_reg_export_fu,constituents,years):
    regions = list(basin_load_to_reg_export_fu.keys())
    results = {}
    for con in constituents:
        progress(con)
        results[con] = {}
        for region in regions:
            progress('  ',region)
            contributions, totals = compute_predev_vs_anthropogenic(basin_load_to_reg_export_fu, region, con,years,'BASE')
            results[con][region] = {
                'contributions': contributions,
                'totals': totals
            }
    return results

# migrated by copilot
def plot_predev_vs_anthropogenic(
    output_path,
    regions,
    constituents,
    basin_load_to_reg_export_fu,
    years,
    region_labels
):
    results = {}
    for con in constituents:
        scale = 1.0
        if con == 'Flow':
            scale = c.M3_TO_ML
        elif con == 'TSS':
            scale = c.KG_TO_KTONS
        elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            scale = c.KG_TO_TONS

        anthros_vs_predevs = pd.DataFrame([])
        predev_vs_base_vs_change_summary = pd.DataFrame([])
        missing_regions = set()
        for region in regions:
            contributions, totals = compute_predev_vs_anthropogenic(basin_load_to_reg_export_fu, region, con,years)
            if contributions is None:
                logger.info('Skipping %s for %s due to missing data', region, con)
                missing_regions.add(region)
                continue
            contributions = contributions * scale
            totals = totals * scale
            # Reindex for region
            # ...region-specific reindexing logic can be added here if needed...
            ax = contributions.plot(kind='bar', fontsize=10, stacked=True, color=['dodgerblue', 'darkorange'], edgecolor='black')
            if con == 'Flow':
                ax.set_ylabel(con + ' (GL/yr OR ML/yr ???)', size=12)
            elif con == 'TSS':
                ax.set_ylabel(con + ' Load (kt/yr)', size=12)
            elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
                ax.set_ylabel(con + ' Load (t/yr)', size=12)
            else:
                ax.set_ylabel(con + ' Load (kg/yr)', size=12)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.125), fancybox=True, ncol=2, fontsize=11)
            ax.set_xlabel('')
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.1, left=0.06, top=2.945, right=0.99)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04), fancybox=True, shadow=True, ncol=2, fontsize=11)
            save_figure(os.path.join(output_path, region, "predevAnthropogenicExports", f"predevAnthropogenicExports_{con}.png"))
            anthros_vs_predevs = pd.concat([anthros_vs_predevs, contributions])
            predev_vs_base_vs_change_summary = pd.concat([predev_vs_base_vs_change_summary, totals])
        
        if len(missing_regions):
            logger.warning('No data for %s. Cannot produce overall chart', ', '.join(missing_regions))
            continue
        # Add summary columns
        predev_vs_base_vs_change_summary.insert(loc=2, column='Anthropogenic', value=predev_vs_base_vs_change_summary['Base'] - predev_vs_base_vs_change_summary['Pre-development'])
        predev_vs_base_vs_change_summary.insert(loc=3, column='Increase Factor', value=predev_vs_base_vs_change_summary['Anthropogenic'] / predev_vs_base_vs_change_summary['Pre-development'])
        predev_vs_base_vs_change_summary.round({"Pre-development": 0, "Base": 0, "Anthropogenic": 0, "Increase Factor": 2})
        # Save summary CSV
        predev_vs_base_vs_change_summary.to_csv(os.path.join(output_path, region_labels[-1], "variousTables", f"predevVSbaseVSchange_summary_{con}.csv"))
        results[con] = {
            'anthros_vs_predevs': anthros_vs_predevs,
            'predev_vs_base_vs_change_summary': predev_vs_base_vs_change_summary
        }

        a = anthros_vs_predevs.T
        a.insert(loc = 7 , column = '', value = 0)
        a.insert(loc = 17 , column = ' ', value = 0)
        a.insert(loc = 29 , column = '  ', value = 0)
        a.insert(loc = 34 , column = '   ', value = 0)
        a.insert(loc = 47 , column = '    ', value = 0)
        anthros_vs_predevs = a.T
        anthros_vs_predevs = anthros_vs_predevs[::-1]

        axx = anthros_vs_predevs.plot(kind='barh',fontsize = 10, stacked=True,color=['dodgerblue','darkorange'],edgecolor='black')

        if con == 'Flow':            
            axx.set_xlabel(con + ' (GL/yr OR ML/yr ???)',size=12)
        elif con == 'TSS':
            axx.set_xlabel(con + ' Load (kt/yr)',size=12)
        elif con in ['DIN', 'DON', 'PN', 'DOP', 'DIP', 'PP']:
            axx.set_xlabel(con + ' Load (t/yr)',size=12)
        else:
            axx.set_xlabel(con + ' Load (kg/yr)',size=12)
            
        axx.set_ylabel('', size=14)
                    
        axx.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        axx.text(axx.get_xlim()[1]*0.67, 46.5,  region_labels[0], size=12)
        axx.text(axx.get_xlim()[1]*0.67, 37.75, region_labels[1], size=12)
        axx.text(axx.get_xlim()[1]*0.67, 27.75, region_labels[2], size=12)
        axx.text(axx.get_xlim()[1]*0.67, 18.5,  region_labels[3], size=12)
        axx.text(axx.get_xlim()[1]*0.67, 12.25, region_labels[4], size=12)
        axx.text(axx.get_xlim()[1]*0.67, 0.25,  region_labels[5], size=12)

        axx.axhline(y=45, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
        axx.axhline(y=35, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
        axx.axhline(y=23, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
        axx.axhline(y=18, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
        axx.axhline(y=5, xmin=-1, xmax=100000, color='black', linestyle='--', lw=1)
            
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.1,left=0.06,top=2.945,right=0.99)
        #axx = fig.add_subplot(1,1,1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),fancybox=True, shadow=True, ncol=2, fontsize = 11)

        save_figure(os.path.join(output_path, region_labels[-1], 'predevAnthropogenicExports','predevAnthropogenicExports_' + con + '.png'))

    return results


def summarise_annual_observations(site_list,modelled_annual_by_region,measured_annual_by_region,only_good_quality=None):
    if only_good_quality is None:
        quality = ['yes','no']
        result = {}

        for q in quality:
            progress(q)
            result[q] = summarise_annual_observations(site_list,modelled_annual_by_region,measured_annual_by_region,only_good_quality=(q == 'yes'))
        return result

    regions = get_regions(site_list)
    summaryAnnualRegions = {}

    for region in regions:
        progress (region)
        measuredSites = site_list[site_list['Region']==region]['Site Code']
        modelled_sites = site_list[site_list['Region']==region]['Node']
        summaryAnnualSites = {site: pd.DataFrame() for site in measuredSites}
        region_modelled = modelled_annual_by_region[region]
        region_measured = measured_annual_by_region[region]
        #summaryAnnualSites = {site: pd.DataFrame() for site in parasCompare_latest}

        for site, modelled_site_name in zip(measuredSites, modelled_sites):
            progress (site)
            if modelled_site_name is None:
                logger.warning('No model node name corresponding to site %s in region %s', site, region) 
                continue
                # raise ValueError('Site is None')
            site_modelled = region_modelled[site]
            if site_modelled is None or site_modelled.empty:
                logger.info('No modelled data for %s/%s', region, site)
                continue

            site_measured = region_measured[site]
            summaryAnnualParas = {para: pd.DataFrame() for para in PARAS_COMPARE_LATEST}

            for para in PARAS_COMPARE_LATEST:
                progress (para)
                if not para in site_modelled:
                    logger.info('No modelled data for %s/%s/%s', region, site, para)
                    # continue
                modelled = site_modelled[para]

                ###Use the below command if the measured is not filtering by representivity rating
                #measured = measuredAnnualRegions[region][site][para]
                #progress(measured.count())

                #### ONLY to filter Excellent, Good & Moderate Representative ratings from observed data (neglect Indicative)

                if only_good_quality:
                    measured = filter_for_good_quality_observations(site_measured)[para]
                    if measured.count() < 3:
                        logger.info('Insufficient "good" quality observations. Skipping %s/%s/%s', region, site, para)
                        continue
                else:
                    measured = site_measured[para]

                # Ensure each year only appears once (take mean if duplicates exist)
                measured = measured.groupby(level=0).mean()
                modelled = modelled.groupby(level=0).mean()

                common_years = measured.index.intersection(modelled.index)

                measured = measured.loc[common_years]
                modelled = modelled.loc[common_years]


                compare = pd.concat([measured,modelled], axis=1)
                compare.columns = [OBSERVATION_COLUMN,'Modelled']

                if para == 'Flow':
                    compare = compare/1000 # TODO WHY?
                elif para == 'TSS':
                    compare = compare/1000 # TODO WHY?
                else:
                    compare = compare

                years_sorted = sorted(compare.index, key=lambda x: int(x.split('-')[0]))

                compare.index = pd.Categorical(compare.index, categories=years_sorted, ordered=True)

                compare = compare.sort_index()
                summaryAnnualParas[para] = compare

            summaryAnnualSites[site] = summaryAnnualParas

        summaryAnnualRegions[region] = summaryAnnualSites

    return summaryAnnualRegions

def compute_model_observed_ratios(site_list,annual_by_quality,only_good_quality=None):
    if only_good_quality is None:
        result = {}
        for q in ['yes','no']:
            progress(q)
            filter_by_quality = q == 'yes'
            # filter_label = 'good quality only' if filter_by_quality else 'all data'
            result[q] = compute_model_observed_ratios(site_list, annual_by_quality,only_good_quality=filter_by_quality)
        return result

    regions = get_regions(site_list)
    result = {}
    for region in regions:
        progress(region)
        result[region] = {}
        q = 'yes' if only_good_quality else 'no'
        measuredSites = site_list[site_list['Region']==region]['Site Code']
        region_data = annual_by_quality[q][region]

        for site in measuredSites:
            averageAnnualsBySites = []
            progress(site)
            site_data = region_data[site]
            if site_data is None or not len(site_data):
                logger.info('No data for %s/%s', region, site)
                result[region][site] = None
                continue

            for para in PARAS_COMPARE_LATEST:
                progress(para)

                # summaryAnnual_temp = annual_by_quality[q][region]
                annuals = site_data[para]
                nonNaNs = annuals.dropna()
                averageAnnual = nonNaNs.mean()
                numYears = nonNaNs.count()
                averageAnnual['num'] = numYears.sum()/2

                averageAnnualsBySites.append(averageAnnual)

            averageAnnualSites = pd.DataFrame(averageAnnualsBySites)
            averageAnnualSites = averageAnnualSites.fillna(0)
            averageAnnualSites = averageAnnualSites.T
            averageAnnualSites.columns = ['', '','','','','','','','','']  #,''

            columns = ['Flow (GL/yr)', 'TSS (kt/yr)','TN (t/yr)','PN (t/yr)','DIN (t/yr)','DON (t/yr)','TP (t/yr)','PP (t/yr)','DIP (t/yr)','DOP (t/yr)']  #, 'PSII TE (kg/yr)'

            averageAnnualSites.columns = columns

            ratios_names = ['PN:TSS \n(t/kt)','PN:TN \n(t/t)','DIN:TN \n(t/t)','DON:TN \n(t/t)','PP:TSS \n(t/kt)','PP:TP \n(t/t)','DIP:TP \n(t/t)','DOP:TP \n(t/t)']

            if averageAnnualSites.sum().sum() == 0:
                result[region][site] = None
                continue

            measured = pd.DataFrame(averageAnnualSites.T[OBSERVATION_COLUMN]).T
            modelled = pd.DataFrame(averageAnnualSites.T['Modelled']).T

            ratios_measured = pd.DataFrame([
                                measured['PN (t/yr)']/measured['TSS (kt/yr)'],
                                measured['PN (t/yr)']/measured['TN (t/yr)'],
                                measured['DIN (t/yr)']/measured['TN (t/yr)'],
                                measured['DON (t/yr)']/measured['TN (t/yr)'],
                                measured['PP (t/yr)']/measured['TSS (kt/yr)'],
                                measured['PP (t/yr)']/measured['TP (t/yr)'],
                                measured['DIP (t/yr)']/measured['TP (t/yr)'],
                                measured['DOP (t/yr)']/measured['TP (t/yr)']
                                ])

            ratios_modelled = pd.DataFrame([
                                modelled['PN (t/yr)']/modelled['TSS (kt/yr)'],
                                modelled['PN (t/yr)']/modelled['TN (t/yr)'],
                                modelled['DIN (t/yr)']/modelled['TN (t/yr)'],
                                modelled['DON (t/yr)']/modelled['TN (t/yr)'],
                                modelled['PP (t/yr)']/modelled['TSS (kt/yr)'],
                                modelled['PP (t/yr)']/modelled['TP (t/yr)'],
                                modelled['DIP (t/yr)']/modelled['TP (t/yr)'],
                                modelled['DOP (t/yr)']/modelled['TP (t/yr)']
                                ])

            ratios_summary = pd.concat([ratios_measured,ratios_modelled],axis=1)
            ratios_summary.index = ratios_names
            result[region][site] = ratios_summary
    return result

def model_observed_ratios_by_site(output_path,regions,site_list,annual_by_quality,only_good_quality=True):
    ratio_tables = compute_model_observed_ratios(site_list,annual_by_quality,only_good_quality)
    for region in regions:
        region_tables = ratio_tables[region]
        progress(region)

        for site, ratios_summary in region_tables.items():
            if ratios_summary is None:
                continue
            axx = pd.DataFrame(ratios_summary).plot(kind='bar',fontsize=8,color=['dodgerblue','tomato'],edgecolor='black')

            axx.set_ylabel("Ratios", size = 8)
            axx.legend(['Monitored','Modelled'],ncol=2,fontsize=8)
            #plt.text(-0.5,-1,'PN:TSS and PP:TSS ratios are in t/kt, while others ratios are in t/t',fontsize = 10)


            #axx.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.1,left=0.0,top=1.0,right=1.3)
            # axx = fig.add_subplot(111)

            if only_good_quality:
                sub_dir = 'qualityData'
            else:
                sub_dir = 'allData'
            save_figure(os.path.join(output_path,region,'modelledVSmeasured',sub_dir,'ratios',site + '.png'))

def average_annual_comparisons(site_list,annual_by_quality):
    regions = get_regions(site_list)
    result = {}
    for filter_data in [True,False]:
        q = 'yes' if filter_data else 'no'
        progress(q)
        result[q] = {}
        for region in regions:
            progress(region)
            result[q][region] = {}
            measuredSites = site_list[site_list['Region'] == region]['Site Code'].tolist()
            region_data = annual_by_quality[q][region]

            for site in measuredSites:
                averageAnnualsBySites = []
                progress(site)
                site_data = region_data[site]

                if site_data is None or not len(site_data):
                    logger.info('No data for %s/%s', region, site)
                    result[q][region][site] = None
                    continue

                for para in PARAS_COMPARE_LATEST:
                    progress(para)


                    annuals = site_data[para]
                    nonNaNs = annuals.dropna()
                    averageAnnual = nonNaNs.mean()
                    numYears = nonNaNs.count()

                    averageAnnual['num'] = numYears.sum()/2

                    averageAnnualsBySites.append(averageAnnual)

                averageAnnualSites = pd.DataFrame(averageAnnualsBySites)
                averageAnnualSites = averageAnnualSites.fillna(0)
                averageAnnualSites = averageAnnualSites.T
                averageAnnualSites.columns = ['Flow (GL/yr)', 'TSS (kt/yr)','TN (t/yr)','PN (t/yr)','DIN (t/yr)','DON (t/yr)','TP (t/yr)','PP (t/yr)','DIP (t/yr)','DOP (t/yr)']  #, 'PSII TE (kg/yr)'

                if averageAnnualSites.sum().sum() == 0:
                    result[q][region][site] = None
                    continue
                result[q][region][site] = averageAnnualSites

    return result

def plot_average_annual_comparisons(output_path,regions,site_list,annual_by_quality):
    comparisons = average_annual_comparisons(regions,site_list,annual_by_quality)
    for filter_data in [True,False]:
        q = 'yes' if filter_data else 'no'
        progress(q)

        for region, region_data in comparisons[q].items():
            progress(region)

            for site, site_data in region_data.items():
                progress(site)

                averageAnnualSites = site_data
                if averageAnnualSites is None:
                    continue
                columns = averageAnnualSites.columns[:]
                averageAnnualSites.columns = ['', '','','','','','','','','']  #,''
                axx = pd.DataFrame(averageAnnualSites[0:2].T).plot(kind='bar',fontsize=8,color=['dodgerblue','tomato'],edgecolor='black')

                axx.set_ylabel("Flow and Loads", size = 8)
                axx.legend(['Monitored','Modelled'],ncol=2,fontsize=8)

                axx.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

                columns = ['Flow (GL/yr)', 'TSS (kt/yr)','TN (t/yr)','PN (t/yr)','DIN (t/yr)','DON (t/yr)','TP (t/yr)','PP (t/yr)','DIP (t/yr)','DOP (t/yr)']  #, 'PSII TE (kg/yr)'

                averageAnnualSites = averageAnnualSites.values.astype(int)
                averageAnnualSites = pd.DataFrame(averageAnnualSites)
                averageAnnualSites = averageAnnualSites.map('{:,}'.format)
                averageAnnualSites = averageAnnualSites.values

                # Add a table at the bottom of the axes
                the_table = plt.table(cellText=averageAnnualSites,
                            rowLabels=['Monitored','Modelled','Years of Data'],
                            colLabels=columns,
                            loc='bottom')
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(7)


                fig = plt.gcf()
                fig.subplots_adjust(bottom=0.1,left=0.0,top=1.0,right=1.3)
                #axx = fig.add_subplot(111)

                if q == 'yes':
                    sub_dir = 'qualityData'
                else:
                    sub_dir = 'allData'
                save_figure(os.path.join(output_path,region,'modelledVSmeasured',sub_dir,'averageAnnuals','bySites','averageAnnuals_' + site + '.png'))

def average_annual_comparison_at_regional_sites(site_list,annual_by_quality):
    regions = get_regions(site_list)
    result = {}
    for q in ['yes','no']:#[0:1]:
        progress(q)
        result[q] = {}
        for para in PARAS_COMPARE_LATEST:
            progress(para)
            result[q][para] = {}

            for region in regions:
                progress(region)
                region_data = annual_by_quality[q][region]

                sitesOfInterest = pd.DataFrame(site_list[site_list['Region'] == region]['Site Code'].tolist())
                sitesOfInterest.columns = ['Sname']
                sitesOfInterest = sitesOfInterest.sort_values('Sname')
                sitesOfInterest = sitesOfInterest['Sname'].tolist()
                measuredSites = sitesOfInterest

                progress(measuredSites)

                averageAnnualsByParas = []
                valid_sites = []

                for site in measuredSites:
                    progress(site)
                    site_data = region_data[site]
                    if site_data is None or not len(site_data):
                        logger.info('No data for %s/%s', region, site)
                        continue
                    valid_sites.append(site)
                    annuals = site_data[para]
                    averageAnnual = annuals.dropna().mean()

                    averageAnnualsByParas.append(averageAnnual)

                averageAnnualParas = pd.DataFrame(averageAnnualsByParas)
                averageAnnualParas.index = valid_sites
                num = regions.index(region)
                #axx = fig.add_subplot(3,2,num+1)
                averageAnnualParas = averageAnnualParas.dropna()

                if averageAnnualParas.empty:
                    result[q][para][region] = None
                    continue

                result[q][para][region] = averageAnnualParas

    return result

def plot_average_annual_comparison_at_regional_sites(output_path,site_list,annual_by_quality):
    ### Plots average annuals by Constituents for all regional sites
    comparisons = average_annual_comparison_at_regional_sites(site_list,annual_by_quality)
    for q in ['yes','no'][0:1]:
        progress(q)
        para_comparisons = comparisons[q]
        for para, para_data in para_comparisons.items():
            progress(para)
            missing_regions = set()
            for region, region_data in para_data.items():
                progress(region)
                averageAnnualParas = region_data
                if averageAnnualParas is None:
                    logger.info('Skipping %s for %s due to missing data', region, para)
                    missing_regions.add(region)
                    continue
                measuredSites = averageAnnualParas.index.tolist()
                progress(measuredSites)

                ax = pd.DataFrame(averageAnnualParas).plot(kind='bar',color=['dodgerblue','tomato'],edgecolor='black')

                progress(averageAnnualParas)

                ax.set_xlabel('')

                if para == 'Flow':
                    ax.set_ylabel(para + ' (GL)',size=10)
                elif para == 'TSS':
                    ax.set_ylabel(para + ' (kt)',size=10)
                elif para == 'PSII TE':
                    ax.set_ylabel(para + ' (kg)',size=10)
                else:
                    ax.set_ylabel(para + ' (t)',size=10)

                ax.legend(['Monitored','Modelled'],ncol=2,fontsize=9, loc='lower center', bbox_to_anchor=(0.5,1))

                # xticks = xticks_by_region[region]

                # xticks = list(xticks[i] for i in pd.DataFrame(measuredSites).index[np.isin(measuredSites, averageAnnualParas.index)==True])#pd.DataFrame(measuredSites)[np.isin(measuredSites, averageAnnualParas.index)==True])
                xticks = [site_list.loc[site_list['Site Code'] == site, 'Display Name'].values[0] for site in measuredSites if site in averageAnnualParas.index]
                ax.set_xticklabels(xticks, rotation = 90)

                if q == 'yes':
                    sub_dir = 'qualityData'
                else:
                    sub_dir = 'allData'
                save_figure(os.path.join(output_path, region, 'modelledVSmeasured', sub_dir, 'averageAnnuals', 'byConstituents', 'averageAnnuals_' + para + '.png'))

def average_annuals_by_constituent_regional(output_path,regions,site_list,annual_by_quality):
    for q in ['yes', 'no']:
        for para in PARAS_COMPARE_LATEST:
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
            fig.subplots_adjust(hspace=.7, wspace=.2)

            for region_idx, region in enumerate(regions):
                measuredSites = site_list.loc[
                    site_list['Region'] == region, 'Site Code'
                ].sort_values().tolist()

                averageAnnualsByParas = []
                valid_sites = []
                for site in measuredSites:
                    site_data = annual_by_quality[q][region][site]
                    if para not in site_data:
                        logger.info('No data for %s/%s/%s', region, site, para)
                        continue
                    valid_sites.append(site)
                    annuals = site_data[para]
                    averageAnnual = annuals.dropna().mean()
                    averageAnnualsByParas.append(averageAnnual)

                averageAnnualParas = pd.DataFrame(
                    averageAnnualsByParas, index=valid_sites
                ).dropna()

                progress(f"Region: {region}, Parameter: {para}, Quality: {q}")
                progress(averageAnnualParas)  # <-- dump actual data values here

                axx = axs.flatten()[region_idx]

                if averageAnnualParas.empty:
                    progress(f"[DEBUG] No data to plot for region '{region}'")
                    axx.set_visible(False)
                else:
                    averageAnnualParas.plot(
                        kind='bar',
                        color=['dodgerblue', 'tomato'],
                        edgecolor='black',
                        ax=axx
                    )
                    axx.set_title(region)

            # Save the figure after all subplots are ready
            if q == 'yes':
                sub_dir = 'qualityData'
            else:
                sub_dir = 'allData'
            save_figure(os.path.join(output_path,OVERALL_REGION,'modelledVSmeasured',sub_dir,'averageAnnuals',f'averageAnnuals_{para}.png'))

            plt.close(fig)

def calc_moriasi_stats(constituents,site_list,annual_by_quality):
    regions = get_regions(site_list)
    MORIASI_COLUMNS=[
        'Region','Site','Constituent',
        'RSR','RSR Rating',
        'NSE','NSE Rating',
        'R2','R2 Rating',
        'PBias','PBias Rating'
    ]
    result = {}
    for q in ['yes','no']:
        result[q] = {}

        progress(q)

        for region in regions:
            progress(region)
            result[q][region] = {}

            sitesOfInterest = site_list[site_list['Region'] == region]['Site Code'].tolist()
            measuredSites = sitesOfInterest

            sitesMoriasi = []

            summaryAnnual_temp_list = annual_by_quality[q][region]

            dfs = []

            for site_id, param_dict in summaryAnnual_temp_list.items():
                for param, df in param_dict.items():
                    # Reset index to get year as a column
                    df_reset = df.reset_index()
                    # Rename the index column to 'Year'
                    df_reset = df_reset.rename(columns={'index': 'Year'})
                    # Add columns for site and parameter
                    df_reset['Site'] = site_id
                    df_reset['Constituent'] = param
                    dfs.append(df_reset)

            # Concatenate all into one DataFrame
            summaryAnnual_temp = pd.concat(dfs, ignore_index=True)

            # Optional: reorder columns for clarity
            summaryAnnual_temp = summaryAnnual_temp[['Site', 'Constituent', 'Year', OBSERVATION_COLUMN, 'Modelled']]

            summaryAnnual_temp = summaryAnnual_temp.rename(columns={OBSERVATION_COLUMN: 'Monitored'})

            # annuals = summaryAnnual_temp[(summaryAnnual_temp['Site']==site ) & (summaryAnnual_temp['Constituent']==para)]

            for site in sitesOfInterest:
                if summaryAnnual_temp.loc[(summaryAnnual_temp['Site'] == site) & (summaryAnnual_temp['Constituent'] == 'Flow')].empty == False:
                    if summaryAnnual_temp.loc[(summaryAnnual_temp['Site'] == site) & (summaryAnnual_temp['Constituent'] == 'Flow')]['Monitored'].count()>2:
                        sitesMoriasi.append(site)
                    else:
                        continue
                else:
                    continue

            for site in sitesMoriasi:

                progress(site)
                ### 1986 to 2023

                Moriasi_stat_site_86_23 =  pd.DataFrame(columns=MORIASI_COLUMNS)

                Annual_site_df = summaryAnnual_temp.loc[summaryAnnual_temp['Site'] == site]

                for idx, constituent in enumerate(constituents):
                    progress(constituent)
                    Annual_site_df_constituent = Annual_site_df.loc[(Annual_site_df['Constituent'] == constituent)]

                    # Remove rows with NaN in 'Modelled' and 'Monitored' columns
                    Annual_site_df_cleaned = Annual_site_df_constituent.dropna(subset=['Modelled', 'Monitored'])

                    # Calculate the average values for Load_Modelled_T and Load_monitored_T
                    load_modelled = Annual_site_df_cleaned['Modelled']
                    load_monitored = Annual_site_df_cleaned['Monitored']

                    pbias = round(spotpy.objectivefunctions.pbias(load_monitored, load_modelled ),2)
                    nse = round(spotpy.objectivefunctions.nashsutcliffe(load_monitored, load_modelled),2)

                    RSR = (1-nse)**(1/2)

                    # Calculate R-squared
                    SS_res = ((load_monitored - load_modelled) ** 2).sum()
                    SS_tot = ((load_monitored - load_modelled.mean()) ** 2).sum()
                    r2 = round((1 - (SS_res / SS_tot)),2)

                    # get the ratings :
                    if constituent in ['TN', 'PN', 'DIN', 'DON']:
                        moriasi_category='TN'
                    elif constituent in ['TP', 'PP', 'DIP', 'DOP']:
                        moriasi_category='TP'
                    else:
                        moriasi_category=constituent

                    rsr_rating = moriasi.rsr_rating(RSR,moriasi_category)
                    pbias_rating = moriasi.pbias_rating(pbias,moriasi_category)
                    nse_rating = moriasi.nse_rating(nse,moriasi_category)
                    r2_rating = moriasi.r2_rating(r2,moriasi_category)

                    # Append the values to the moriasi_stat

                    Moriasi_stat_site_86_23.loc[idx] = [region,site, constituent,RSR, rsr_rating, nse, nse_rating, r2, r2_rating, pbias, pbias_rating]
                result[q][region][site] = Moriasi_stat_site_86_23

    return result

def report_calc_moriasi_stats(output_path,regions,constituents,site_list,annual_by_quality):
    all_stats = calc_moriasi_stats(constituents,site_list,annual_by_quality)
    for q in ['yes','no']:

        filename_suffix='_quality_only' if q=='yes' else '_alldata'
        site_directory='qualityData' if q=='yes' else 'allData'

        progress(q)

        Moriasi_stat_region_86_23 =  pd.DataFrame()

        for region, region_data in all_stats[q].items():
            progress(region)

            sitesOfInterest = site_list[site_list['Region'] == region]['Site Code'].tolist()
            measuredSites = sitesOfInterest

            for site, site_data in region_data.items():

                progress(site)
                ### 1986 to 2023

                Moriasi_stat_site_86_23 =  site_data
                Moriasi_stat_site_86_23.to_csv(os.path.join(output_path,region,'modelledVSmeasured',site_directory,'Moriasi',f'{site}_Moriasi_stats_86_14{filename_suffix}.csv'),index=False)

                Moriasi_stat_region_86_23 = pd.concat([Moriasi_stat_region_86_23, Moriasi_stat_site_86_23])

        Moriasi_stat_region_86_23.to_csv(os.path.join(output_path,f'Moriasi_stats_reporting_period_86_23{filename_suffix}.csv'),index=False)

def compute_annual_comparisons_all_sites(site_list,annual_by_quality):
    result = {}
    regions = get_regions(site_list)
    for q in ['yes','no']:
        progress(q)
        result[q] = {}
        for region in regions:
            progress (region)
            result[q][region] = {}
            measuredSites = site_list[site_list['Region'] == region]['Site Code'].tolist()
            region_data = annual_by_quality[q][region]
            sitesAnnual = []
            for site in measuredSites:
                # progress(site)
                site_data = region_data[site]
                if 'Flow' in site_data and not site_data['Flow'].empty:
                    sitesAnnual.append(site)

            for site in sitesAnnual:
                progress (site)

                    #summaryAnnualParas = {para: pd.DataFrame() for para in parasCompare_latest}
                result[q][region][site] = {}

                for para in PARAS_COMPARE_LATEST[:-1]:  #[0:1]
                    progress (para)

                    annuals = annual_by_quality[q][region][site][para]
                    compare = annuals.dropna().copy()

                    if compare[OBSERVATION_COLUMN].isnull().all().all():
                        compare[OBSERVATION_COLUMN] = compare[OBSERVATION_COLUMN].replace('NaN', 0)
                    # else:
                    #     compare[OBSERVATION_COLUMN] = compare[OBSERVATION_COLUMN]

                    compare = compare.T
                    compare['Avg. Annual'] = compare.T.dropna().mean()
                    compare = compare.T
                    result[q][region][site][para] = compare
    return result

def plot_annual_all_sites(output_path,regions,site_list,annual_by_quality):
    comparisons = compute_annual_comparisons_all_sites(site_list,annual_by_quality)
    for q in ['yes','no']:
        progress(q)
        quality_data = comparisons[q]
        for region, region_data in quality_data.items():
            progress (region)

            for site, site_data in region_data.items():
                progress (site)

                    #summaryAnnualParas = {para: pd.DataFrame() for para in parasCompare_latest}

                for para in PARAS_COMPARE_LATEST[:-1]:  #[0:1]
                    progress (para)
                    compare = site_data[para]

                    #axx = compare.dropna().plot(kind='bar',title= para + ' at ' + site)
                    axx = compare.dropna().plot(kind='bar')  #,title= para + ' at ' + site

    #                 compare.to_csv(reportcardOutputsPrefix + '//' + region + '//modelledVSmeasured//qualityData//' + 'annuals//' + site + para + 'annualComparions.csv')

                    if para == 'Flow':
                        axx.set_ylabel(para + ' (GL)',size=8)
                    elif para == 'TSS':
                        axx.set_ylabel(para + ' (kt)',size=8)
                    elif para == 'PSII TE':
                        axx.set_ylabel(para + ' (kg)',size=8)
                    else:
                        axx.set_ylabel(para + ' (t)',size=8)

                        #axx.set_xlabel('Water Year',size=8)

                    if q == 'yes':
                        sub_dir = 'qualityData'
                    else:
                        sub_dir = 'allData'
                    save_path = os.path.join(output_path,region,'modelledVSmeasured',sub_dir,'annuals',site)
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    save_figure(os.path.join(save_path,para + '.png'))


### Temporarily stored REGSOURCESINKSUMMARY are used below to do number crunching to produce various plots
### LoadToStream are estimated below for all BASINs by FUs of interest and CONSTITUENTs of interest
### NO UNIT CONVERSION IS DONE HERE - all units are similar to that of REGSOURCESINKSUMMARY
def land_use_supply_by_basin(constituents,regional_contributions,fus_of_interest):
    BasinLoadToStream_FU = {}

    for region, region_data in regional_contributions.items():
        progress(region)

        BasinLoadToStream_FU[region] = {}

        for model_full, data in region_data.items():
            model = model_full.split('_')[0]
            progress(model)

            BasinLoadToStream_FU[region][model] = {}#{basin: pd.DataFrame() for basin in BASINs}

            if data is None:
                continue
            # data_summary = data.reset_index().groupby(['Rep_Region','Constituent','FU']).sum()
            data_summary = data.reset_index().groupby(['MU_48','Constituent','FU']).sum(numeric_only = True)
            BASINs = data_summary.index.levels[0]

            for basin in BASINs:
                progress(basin)

                BasinLoadToStream_FU[region][model][basin] = {}
                data_summary2 = data_summary.loc[basin].T[constituents]
                data_summary3 = data_summary2.T

                for con in constituents:
                    progress (con)

                    #if region == 'BU' and model == 'BASE':
                    #    BasinLoadToStream_FU[region][model][basin][con] = data_summary3.loc[con]['LoadToStream']
                    #elif region == 'BU' and model == 'CHANGE':
                    #    BasinLoadToStream_FU[region][model][basin][con] = data_summary3.loc[con]['LoadToStream']
                    #else:
                    BasinLoadToStream_FU[region][model][basin][con] = data_summary3.loc[con]['LoadToStream (kg)']

                BasinLoadToStream_FU[region][model][basin] = pd.DataFrame(BasinLoadToStream_FU[region][model][basin]).T
                # BasinLoadToStream_FU[region][model][basin]['Grazing'] = BasinLoadToStream_FU[region][model][basin]['Grazing Forested'] + \
                #                                                            BasinLoadToStream_FU[region][model][basin]['Grazing Open']
                BasinLoadToStream_FU[region][model][basin]['Cropping'] = BasinLoadToStream_FU[region][model][basin]['Dryland Cropping'] + \
                                                                            BasinLoadToStream_FU[region][model][basin]['Irrigated Cropping']
                BasinLoadToStream_FU[region][model][basin]['Urban + Other'] = BasinLoadToStream_FU[region][model][basin]['Urban'] + \
                                                                                BasinLoadToStream_FU[region][model][basin]['Other']
                for fu in FILL_FUS:
                    if fu not in BasinLoadToStream_FU[region][model][basin]:
                        BasinLoadToStream_FU[region][model][basin][fu] = pd.NA

                BasinLoadToStream_FU[region][model][basin] = BasinLoadToStream_FU[region][model][basin][fus_of_interest].T
    return BasinLoadToStream_FU

### Region by Region Land Use based Exported load Contributions are estimated using above Basin by Basin estimates
def land_use_supply_by_region(basin_supply,scenario='BASE'):
    result = {}

    for region, base in basin_supply.items():
        progress(region)

        base = basin_supply[region][scenario]
        basins = base.keys()

        Region_sum = 0

        for basin, df in basin_supply[region][scenario].items():
            df_clean = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            Region_sum = Region_sum + df_clean

        result[region] = Region_sum
    return result

def land_use_supply_totals(region_supply):
    result = 0

    for region,data in region_supply.items():
        progress(region)
        if hasattr(data, 'fillna'):
            data = data.fillna(0)
        result = result + data

    return result

def plot_land_use_based_supply(output_path,regions,constituents,region_supply,region_areas,years):
    for region in regions:
        progress(region)

        RegionLoad = region_supply[region]
        if RegionLoad is None or isinstance(RegionLoad, int):
            logger.info('Skipping %s due to missing data', region)
            continue
        RegionLoad = RegionLoad[constituents]
        if 'Flow' in RegionLoad.columns:
            RegionLoad = RegionLoad.drop(columns=['Flow'])
        RegionLoad_export = RegionLoad*[c.KG_TO_KTONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS,c.KG_TO_TONS]/years #unit conversion
        RegionFUArea = region_areas[region]
        RegionFUArea = RegionFUArea*c.M2_TO_HA  #unit conversion

        for con in constituents[0:4]:
            progress(con)

            plt.clf()

            if con == 'TSS':
                ArealRegionLoad = RegionLoad[con]*c.KG_TO_TONS/RegionFUArea['Area']/years
            else:
                ArealRegionLoad = RegionLoad[con]/RegionFUArea['Area']/years

            #progress (ArealRegionLoad)

            ax1 = RegionLoad_export[con].plot(kind='bar',color='m',width=0.4,grid="off",edgecolor='black')
            ax1.set_xlabel('')
            ax1.grid(False)

            ax2 = ax1.twinx()
            ax2 = ArealRegionLoad.plot(kind='bar',color='g',width=0.15,grid="off",edgecolor='black')
            ax2.set_xlabel('')
            ax2.grid(False)

            if con == 'Flow':
                ax1.set_ylabel(con + ' (GL/yr OR ML/yr ???)',size=8)
                ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
                ax2.set_ylabel(con + ' (GL/yr OR ML/yr ???)',size=8)
                ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            elif con == 'TSS':
                ax1.set_ylabel(con + ' (kt/yr)',size=8)
                ax1.legend(['kt/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
                ax2.set_ylabel(con + ' (t/ha/yr)',size=8)
                ax2.legend(['t/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            elif con in ['DIN','DON','PN','DOP','DIP','PP']:
                ax1.set_ylabel(con + ' (t/yr)',size=8)
                ax1.legend(['t/yr'],loc='lower left',bbox_to_anchor=(0.05,1),fontsize=8)
                ax2.set_ylabel(con + ' (kg/ha/yr)',size=8)
                ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)
            else:
                ax1.set_ylabel(con + ' (kg/yr)',size=8)
                ax1.legend(['kg/yr'],loc='lower left',bbox_to_anchor=(0.05,1.0),fontsize=8)
                ax2.set_ylabel(con + ' (kg/ha/yr)',size=8)
                ax2.legend(['kg/ha/yr'],loc='lower left', bbox_to_anchor=(0.2,1.0),fontsize=8)

            save_figure(os.path.join(output_path,region,'landuseBasedSupply', con + '_supplyTonnageArealsbyLanduse.png'))

            ### Estimating and Plotting TSS, PN, PP, DIN Contribution by Landuse

            contribution = RegionLoad[constituents[:4]]
            contribution = contribution/contribution.sum()*100


            ax3 = contribution.T.plot(kind='bar',width=0.6,grid="off",color=FU_COLORS)

            ax3.legend(bbox_to_anchor=(1,1),fontsize=6.4,ncol=5)
            ax3.set_ylim(0,100)
            ax3.set_ylabel("Percent Contribution",size=8)
            ax3.set_xticklabels(['TSS','PN','PP','DIN'],rotation='horizontal')
            ax3.set_xlabel('')

            save_figure(os.path.join(output_path,region,'landuseBasedSupply','supplyPercentByLanduse.png'))

def plot_sankey_diagrams(output_path,sankey_regions,sankey_basins,source_sink_budgets,export_by_process,years):
    # Make these settings explicit....
    scenario='BASE'
    constituent='TSS'
    scale=c.KG_TO_KTONS
    units='kt/Year'

    for region in sankey_regions:
        progress(region)
        ss_region = source_sink_budgets[region][scenario]
        export_region = export_by_process[region][scenario]

        basins = sankey_basins[sankey_regions.index(region)]

        for basin in basins:
            progress(basin)
            if basin not in ss_region:
                logger.info('Skipping %s/%s due to missing source sink data', region, basin)
                continue
            if basin not in export_region:
                logger.info('Skipping %s/%s due to missing export data', region, basin)
                continue
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.1,left=0.06,top=1.945,right=2.2)
            ax = fig.add_subplot(1,1,1,xticks=[], yticks=[])
            plt.rc('font', size=12)
            ax.axis('off')


            SourceSink = ss_region[basin][constituent]*scale/years
            SourceSink.columns = [units]
            SourceSink

            Export = export_region[basin][constituent]*scale/years
            Export.columns = [units]
            Export


            SourceSink_new = pd.DataFrame([])
            Export_new = pd.DataFrame([])

            SourceSink_new["Streambank_supply"] = SourceSink.T["Streambank"]
            SourceSink_new["Hillslope_supply"] = SourceSink.T["Hillslope"]
            SourceSink_new["Gully_supply"] = SourceSink.T["Gully"]
            SourceSink_new["ChannelRemobilisation_supply"] = SourceSink.T["ChannelRemobilisation"]

            SourceSink_new["Extraction_Other_loss"] = SourceSink.T["Extraction + Other Minor Losses"]*-1
            SourceSink_new["FloodPlainDeposition_loss"] = SourceSink.T["FloodPlainDeposition"]*-1
            SourceSink_new["ReservoirDeposition_loss"] = SourceSink.T["ReservoirDeposition"]*-1
            SourceSink_new["StreamDeposition_loss"] = SourceSink.T["StreamDeposition"]*-1

            Export_new["ChannelRemobilisation_export"] = Export.T["ChannelRemobilisation"]*-1
            Export_new["Gully_export"] = Export.T["Gully"]*-1
            Export_new["Hillslope_export"] = Export.T["Hillslope"]*-1
            Export_new["Streambank_export"] = Export.T["Streambank"]*-1

            summary = pd.concat([SourceSink_new.T,Export_new.T])

            Total_supply = SourceSink.T["Streambank"]+SourceSink.T["Hillslope"]+SourceSink.T["Gully"]+SourceSink.T["ChannelRemobilisation"]
            Total_loss = SourceSink.T["Extraction + Other Minor Losses"]+SourceSink.T["FloodPlainDeposition"]+SourceSink.T["ReservoirDeposition"]+SourceSink.T["StreamDeposition"]
            Total_export = Export.T["Gully"]+Export.T["Hillslope"]+Export.T["Streambank"]+Export.T["ChannelRemobilisation"]

            summary = summary[summary[units] != 0 ]

            numbers = summary[units].tolist()

            progress(summary)
            progress(summary[units].sum())
            progress(numbers)
            #######################
            ##### plotting ########
            #######################
            TEXT_OPT={
                'fontsize':20,
                'color': 'black'
            }

            SANKEY_OPT={
                'unit':'kt/y',
                'offset':0.2,
                'head_angle':100,
                'alpha':0.75,
                'lw':1,
                'facecolor':'brown'
            }

            #
            #
            # PRIMARY DIFFERENCE IS WHETHER OR NOT THEY HAVE RESERVOIR DEPOSITION
            #
            #
            sankey_specifics={
                'format':'%.2f',
                'scale':0.001
            }
            have_reservoir=False
            if region == 'CY':
                lbl_supply_loc = -1.32,0
                lbl_export_loc = 1.3,0.4
                lbl_loss_loc = 0.45, -1.1
                lbl_char = '~'

            elif region == 'WT':
                lbl_supply_loc = -1.5,0
                lbl_export_loc = 1.3, 0.4
                lbl_loss_loc = 0.45,-1.2
                lbl_char = ' '
                sankey_specifics['format']='%.0f'
                if basin not in ['Johnstone','Upper Herbert','Tully']:
                    have_reservoir=True

            elif region == 'BU':
                have_reservoir=True
                lbl_supply_loc = -1.5,0
                lbl_export_loc = 1.2, 0.3
                lbl_loss_loc = 0.7,-1.1
                lbl_char = ' '
                sankey_specifics['format']='%.0f'
                sankey_specifics['scale']=0.00007

            elif region == 'MW':
                lbl_supply_loc = -1.2,0.2
                lbl_export_loc = 1.2, 0.3
                lbl_loss_loc = 0.7,-1.1
                lbl_char = '~'
                sankey_specifics['scale']=0.0003
                if basin == 'Pioneer' :
                    have_reservoir=True

            elif region == 'FI':
                lbl_supply_loc = -1.25,0.0
                lbl_export_loc = 1.1, 0.45
                lbl_loss_loc = 0.7,-1.2
                lbl_char = ' '
                sankey_specifics['format']='%.0f'
                sankey_specifics['scale']=0.0001

            elif region == 'BM':
                have_reservoir=True

            if have_reservoir:
                labels=['Streambank\nSupply','Hillslope\nSupply','Gully\nSupply',
                        'Extraction and \nOther \nMinor Lossees', 'Floodplain\ndeposition','Reservoir\nDeposition',
                        'Gully\nExport', 'Hillslope\nExport','Streambank\nExport',]  #'Channel\nRemobilisation\nExport',
                orientations=[0, -1, 1, #1,
                                -1, -1, -1, #-1,
                                1, 1, 0] #1,
            else:
                labels=['Streambank\nSupply','Hillslope\nSupply','Gully\nSupply',
                        'Extraction and \nOther \nMinor Losses', 'Floodplain\ndeposition',
                        'Gully\nExport', 'Hillslope\nExport','Streambank\nExport',]
                orientations=[0, -1, 1,
                                -1, -1,
                                1, 1, 0]

            Sankey(flows=numbers,
                    labels=labels,
                    orientations=orientations,
                    ax=ax,**sankey_specifics,**SANKEY_OPT).finish()

            lbl_total_supply = str(round(Total_supply[units],0).astype(int))
            lbl_total_export = str(int(round(Total_export[units],0)))
            lbl_total_loss = str(round(Total_loss[units],0).astype(int))

            plt.text(*lbl_supply_loc,'TOTAL SUPPLY\n  ' + lbl_char + ' ' + lbl_total_supply + ' kt/yr', **TEXT_OPT, rotation=90)
            plt.text(*lbl_export_loc,'TOTAL EXPORT\n  ' + lbl_char + ' ' + lbl_total_export + ' kt/yr', **TEXT_OPT, rotation=0)
            plt.text(*lbl_loss_loc,'TOTAL LOSS\n  ' + lbl_char + ' ' + lbl_total_loss + ' kt/yr', **TEXT_OPT, rotation=0)

            save_figure(os.path.join(output_path,region,'budgetExports_sankeyDiagrams',basin + '_budget_TSS.png'))

def load_all_tables(ds,dfs,tag_names,**kwargs):
    tag_name = tag_names[0]
    tag_names = tag_names[1:]
    for tag_value,collection in dfs.items():
        tags = kwargs.copy()
        tags[tag_name] = tag_value
        if collection is None:
            continue
        elif isinstance(collection,pd.DataFrame):
            collection.index = collection.index.map(lambda s: str(s).replace('\n',' '))
            ds.add_table(collection,**tags)
        elif isinstance(collection,float):
            logger.info(f"Skipping float for {tag_name}={tag_value}")
        else:
            try:
                assert isinstance(collection,dict)
            except:
                logger.error(f"Unexpected type {type(collection)} for {tags}")
                raise
            load_all_tables(ds,collection,tag_names,**tags)

def process_two_sets_of_dfs(numerators,denominators,process):
    '''
    Process pairs of DataFrames in two nested sets of dictionaries using a function that takes two DataFrames and returns a DataFrame
    '''
    if isinstance(numerators,pd.DataFrame) or isinstance(numerators, float):
        return process(numerators,denominators)
    else:
        keys = set(numerators.keys()).intersection(set(denominators.keys()))
        return {k: process_two_sets_of_dfs(numerators[k],denominators[k],process) for k in keys}

def per_area(load,area):
    if isinstance(load,pd.DataFrame):
        return (load.T/area['Area']).T
    return 0.0 # load / area['Area']

def divide_dfs(numerators,denominators,fill_value=0.0):
    '''
    Divide all DataFrames in a nested set of dictionaries by corresponding DataFrames in another nested set of dictionaries
    '''
    def process(df_num,df_denom):
        if df_denom.columns.size == 1:
            # Broadcasting division across columns
            return df_num.div(df_denom.loc[:,df_denom.columns[0]])
        return df_num.div(df_denom,fill_value=fill_value)
    return process_two_sets_of_dfs(numerators,denominators,process)

def relabel_quality(dfs):
    relabelled = {}
    for q, df in dfs.items():
        label = 'good quality only' if q == 'yes' else 'all data'
        relabelled[label] = df
    return relabelled


def build_site_list(loads_data):
    site_list = loads_data.groupby(['Site Code']).first().reset_index()
    return site_list[['Region', 'Site Code', 'Node', 'Display Name']]

def load_observed_loads(loads_data_fn):
    loads = read_csv(loads_data_fn,header=0)
    replacements = {}
    for _, group in loads.groupby('Node'):
        codes = list(set(group['Site Code']))
        if len(codes)>1:
            for c in codes[1:]:
                replacements[c] = codes[0]
    loads['Site Code'] = loads['Site Code'].replace(replacements)
    return loads

#### Dashboard Datasets
def populate_load_comparisons(source_data,loads_data_fn,constituents,dest_data):
    regions,runs,scenarios,report_card = identify_regions_and_models(source_data)

    compare_ds = hg.open_dataset(os.path.join(dest_data,'load-comparisons'),mode='w')
    compare_ds.rewrite(False)
    loads = load_observed_loads(loads_data_fn)
    measured_annual_regions = collate_measured_annual_regions(loads)
    site_list = build_site_list(loads)
    # Collate modelled annual regions using refactored function
    modelled_daily_regions, modelled_annual_regions = collate_modelled_annual_regions(source_data, site_list, f'BASE_{report_card}')
    summaryAnnualRegions_quality = summarise_annual_observations(site_list,modelled_annual_regions,measured_annual_regions)

    # Ratios
    ratios = compute_model_observed_ratios(site_list, summaryAnnualRegions_quality)
    ratios = relabel_quality(ratios)
    load_all_tables(compare_ds, ratios, ['filter','region','site'],comparison='ratio')

    # Annual timeseries
    annual_comparisons = compute_annual_comparisons_all_sites(site_list,summaryAnnualRegions_quality)
    annual_comparisons = relabel_quality(annual_comparisons)
    load_all_tables(compare_ds, annual_comparisons, ['filter','region','site','parameter'],comparison='annual')

    # Mean annual by constituent
    mean_annual_by_constituent = average_annual_comparison_at_regional_sites(site_list,summaryAnnualRegions_quality)
    mean_annual_by_constituent = relabel_quality(mean_annual_by_constituent)
    load_all_tables(compare_ds, mean_annual_by_constituent, ['filter','parameter','region'],comparison='mean-annual-by-constituent')

    # Mean annual by site
    mean_annual_by_site = average_annual_comparisons(site_list,summaryAnnualRegions_quality)
    mean_annual_by_site = relabel_quality(mean_annual_by_site)
    load_all_tables(compare_ds, mean_annual_by_site, ['filter','region','site'],comparison='mean-annual-by-site')

    # Stats
    moriasi_stats = calc_moriasi_stats(constituents,site_list,summaryAnnualRegions_quality)
    moriasi_stats = relabel_quality(moriasi_stats)
    load_all_tables(compare_ds,moriasi_stats, ['filter','region','site'],comparison='moriasi-stats')

    # Site list
    site_list_db = site_list.rename(columns=lambda c:c.replace(' ','_').lower())
    compare_ds.add_table(site_list_db,contextual='site-list')
    compare_ds.rewrite(True)

def populate_overview_data(source_data,subcatchment_lut,constituents,fus_of_interest,num_years,dest_data):
    per_year = 1/num_years
    regions,runs,scenarios,report_card = identify_regions_and_models(source_data)
    overview_ds = hg.open_dataset(os.path.join(dest_data,'overview'),mode='w')
    overview_ds.rewrite(False)
    REGCONTRIBUTIONDATAGRIDS = read_regional_contributor(source_data,subcatchment_lut)
    BasinLoadToRegExport_Process = {}
    for fu in fus_of_interest+[None]:
        if fu is None:
            BasinLoadToRegExport_Process['all'] = process_based_basin_export_tables(constituents,REGCONTRIBUTIONDATAGRIDS)
        else:
            BasinLoadToRegExport_Process[fu] = process_based_basin_export_tables(constituents,REGCONTRIBUTIONDATAGRIDS,FU=fu)
    BasinLoadToRegExport_Process_PC = to_percentage_of_whole(BasinLoadToRegExport_Process)
    process_based_export = {
        '%': BasinLoadToRegExport_Process_PC,
        'kg/yr': scale_all_dfs(BasinLoadToRegExport_Process,per_year)
    }
    
    process_export_aggregated = concat_data_frames_at_level(process_based_export,4)
    load_all_tables(overview_ds,process_export_aggregated,['units','fu','region','scenario','constituent'],aggregation='process')
    process_export_by_region = sum_data_frames_at_level(process_export_aggregated,2,1)
    process_export_by_region['%'] = to_percentage_of_whole(process_export_by_region['kg/yr'])
    load_all_tables(overview_ds,process_export_by_region,['units','fu','scenario','constituent'],aggregation='process',region=OVERALL_REGION)

    BasinLoadToStream_FU = land_use_supply_by_basin(constituents,REGCONTRIBUTIONDATAGRIDS,fus_of_interest)
    # ### FU area estimation for all 6 REGIONs

    real_fus_of_interest = [fu for fu in fus_of_interest if fu != 'Stream']
    RegionFU_Area_ha = scale_all_dfs(lu_area_by_region(source_data,real_fus_of_interest),c.M2_TO_HA)
    # streamLengths = stream_lengths(MAIN_PATH,REGIONS,REGION_NAMES,RC)
    # fu_area_summary.insert(loc=FUS_OF_INTEREST.index('Stream'), column='Stream (km)', value=streamLengths['Stream'])

    landuse_based_supply = {}
    for model in scenarios:
        print(f'Processing model {model}')
        landuse_based_supply[model] = {}
        RegionLoadToStream_FU_per_year = scale_all_dfs(land_use_supply_by_region(BasinLoadToStream_FU,model),per_year)
        landuse_based_supply[model]['kg/yr'] = RegionLoadToStream_FU_per_year
        landuse_based_supply[model]['kg/ha/yr'] = process_two_sets_of_dfs(RegionLoadToStream_FU_per_year,RegionFU_Area_ha,per_area)
        landuse_based_supply[model]['%'] = to_percentage_of_whole(RegionLoadToStream_FU_per_year)

    BasinLoadToRegExport_FU = load_basin_export_tables(REGCONTRIBUTIONDATAGRIDS,constituents,fus_of_interest)
    RegionLoadToRegExport_FU = build_region_export_tables(BasinLoadToRegExport_FU)

    RegionLoadToRegExport_FU_by_model = {}
    for r in regions:
        for m in scenarios:
            if not m in RegionLoadToRegExport_FU_by_model:
                RegionLoadToRegExport_FU_by_model[m] = {}
            RegionLoadToRegExport_FU_by_model[m][r] = RegionLoadToRegExport_FU[r][m]

    landuse_based_export = {}
    RegionLoadToRegExport_FU_by_model_per_year = scale_all_dfs(RegionLoadToRegExport_FU_by_model,per_year)
    for model in scenarios:
        print(f'Processing model {model}')
        landuse_based_export[model] = {}
        data = RegionLoadToRegExport_FU_by_model_per_year[model]
        landuse_based_export[model]['kg/yr'] = data
        landuse_based_export[model]['kg/ha/yr'] = process_two_sets_of_dfs(data,RegionFU_Area_ha,per_area)
        landuse_based_export[model]['%'] = to_percentage_of_whole(data)\
        
    landuse_loads = {
        'supply': landuse_based_supply,
        'export': landuse_based_export
    }

    load_all_tables(overview_ds,landuse_loads,['load_type','scenario','units','region'],aggregation='landuse')

    for process in ['All','Gully', 'Streambank', 'Hillslope surface soil', 'Hillslope subsurface soil', 'Hillslope no source distinction', 'Undefined',
                    'Channel Remobilisation', 'Diffuse Dissolved']:
        filter = {}
        if process != 'All':
            filter['Process'] = process
        BasinLoadToRegExport_FU = load_basin_export_tables(REGCONTRIBUTIONDATAGRIDS,constituents,fus_of_interest,filter=filter)

        antrhopogenic_exports = compute_anthropogenic_summary(BasinLoadToRegExport_FU,constituents,num_years)
        load_all_tables(overview_ds,antrhopogenic_exports,['constituent','region','summary'],aggregation='anthropogenic',process=process)
    overview_ds.rewrite(True)

