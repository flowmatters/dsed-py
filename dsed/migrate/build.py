import os
import json

from functools import reduce

import pandas as pd
import geopandas as gpd
from dsed.ow import DynamicSednetCatchment, FINE_SEDIMENT, COARSE_SEDIMENT

import openwater.nodes as node_types
from openwater.examples import from_source
from openwater import debugging
from openwater.config import Parameteriser, ParameterTableAssignment, DataframeInputs, DefaultParameteriser, NestedParameteriser
import openwater.template as templating
from openwater.template import OWTemplate, TAG_MODEL,TAG_PROCESS

from veneer.general import _extend_network

SQKM_TO_SQM = 1000*1000
M2_TO_HA = 1e-4
PER_SECOND_TO_PER_DAY=86400
PER_DAY_TO_PER_SECOND=1/PER_SECOND_TO_PER_DAY
G_TO_KG=1e-3
TONS_TO_KG=1e3
PERCENT_TO_FRACTION=1e-2

SOURCE_EMC_MODEL='RiverSystem.Catchments.Models.ContaminantGenerationModels.EmcDwcCGModel'
DS_SEDIMENT_MODEL='Dynamic_SedNet.Models.SedNet_Sediment_Generation'
DS_EMC_GULLY_MODEL='Dynamic_SedNet.Models.SedNet_EMC_And_Gully_Model'
DS_CROP_SED_MODEL='GBR_DynSed_Extension.Models.GBR_CropSed_Wrap_Model'
EMC_DWC_MODELS=[SOURCE_EMC_MODEL,DS_EMC_GULLY_MODEL]
GULLY_MODELS=[DS_EMC_GULLY_MODEL,DS_SEDIMENT_MODEL,DS_CROP_SED_MODEL]

def levels_required(table ,column ,criteria):
    if len(set(table[column]) )==1:
        return 0

    subsets = [table[table[criteria[0] ]==v] for v in set(table[criteria[0]])]

    return max([ 1 +levels_required(subset ,column ,criteria[1:]) for subset in subsets])

def simplify(table ,column ,criteria=['Constituent']):
    levels = levels_required(table ,column ,criteria)
    new_columns = criteria[:levels] + [column]
    return table.drop_duplicates(new_columns)[new_columns]


def compute_ts_sediment_scaling(df):
    fine_sed_scaling = df['LoadConversionFactor'] * df['HillslopeFineSDR'] / 100.0
    coarse_sed_scaling = df['HillslopeCoarseSDR'] / 100.0

    fine_df = df[['catchment', 'cgu']]
    fine_df = fine_df.copy()
    fine_df['scale'] = fine_sed_scaling
    fine_df['constituent'] = FINE_SEDIMENT

    coarse_df = df[['catchment', 'cgu']]
    coarse_df = coarse_df.copy()
    coarse_df['scale'] = coarse_sed_scaling
    coarse_df['constituent'] = COARSE_SEDIMENT

    return pd.concat([fine_df, coarse_df])


# What about SDR and Load_Conversion_Factor --> apply to node_types.Scale
# ts_sediment_scaling = dataframe(catchment,cgu,constituent,scaling_factor)
# where scaling_factor is HillslopeCoarseSDR/100 for coarse, and
#                         HillslopeFineSDR/100 * Load_Conversion_Factor for fine

def _rename_tag_columns(dataframe):
    return dataframe.rename(columns={'Catchment': 'catchment', 'Functional Unit': 'cgu', 'Constituent': 'constituent'})


def _rename_link_tag_columns(dataframe, link_renames, link_col='NetworkElement'):
    dataframe = dataframe.replace(link_renames)
    dataframe = dataframe.rename(columns={'Constituent': 'constituent'})
    dataframe['catchment'] = dataframe[link_col].str.slice(19)

    return dataframe

def build_ow_model(data_path, start='1986/07/01', end='2014/06/30',
                   link_renames={},
                   progress=print):
    builder = SourceOpenwaterDynamicSednetMigrator(data_path)
    return builder.build_ow_model(start,end,link_renames,progress)

class SourceOpenwaterDynamicSednetMigrator(object):
    def __init__(self,data_path):
        self.data_path = data_path
        global RR,ROUTING
        RR = node_types.Sacramento
        ROUTING = node_types.StorageRouting

    def _load_json(self,f):
        return json.load(open(os.path.join(self.data_path, f + '.json')))

    def _load_csv(self,f):
        fn = os.path.join(self.data_path, f + '.csv')
        if not os.path.exists(fn):
            fn = fn + '.gz'
            if not os.path.exists(fn):
                return None

        return pd.read_csv(fn, index_col=0, parse_dates=True)

    def _load_param_csv(self,f):
        df = self._load_csv(f)
        if df is None:
            return None
        return _rename_tag_columns(df)

    def _get_combined_cropping(self):
        dwcs = self._load_param_csv('cg-GBR_DynSed_Extension.Models.GBR_Pest_TSLoad_Model')
        if dwcs is None:
            return None,[]

        dwcs['particulate_scale'] = PERCENT_TO_FRACTION * dwcs['DeliveryRatio'] * PERCENT_TO_FRACTION * dwcs['Fine_Percent']
        dwcs['dissolved_scale'] = PERCENT_TO_FRACTION * dwcs['DeliveryRatioDissolved']

        dwcs['final_scale'] = dwcs['Load_Conversion_Factor']

        scaled_cropping_ts = {}
        for col in cropping.columns:
            constituent, variable, catchment, cgu = col.split('$')
            if not variable in ['Dissolved_Load_g_per_Ha', 'Particulate_Load_g_per_Ha']:
                continue

            row = dwcs[dwcs.catchment == catchment]
            row = row[row.constituent == constituent]
            row = row[row.cgu == cgu]
            assert len(row) == 1
            row = row.iloc[0]

            scale = row['%s_scale' % (variable.split('_')[0].lower())] * row['final_scale']

            scaled_cropping_ts[col] = scale * cropping[col]
        scaled_cropping = pd.DataFrame(scaled_cropping_ts)

        combined_cropping_ts = {}
        cropping_cgus = []
        for col in scaled_cropping.columns:
            constituent, variable, catchment, cgu = col.split('$')
            cropping_cgus.append(cgu)
            if variable != 'Dissolved_Load_g_per_Ha':
                continue
            #  and variable != 'Particulate_Load_g_per_Ha':
            dis_col = col
            part_col = '%s$Particulate_Load_g_per_Ha$%s$%s' % (constituent, catchment, cgu)
            comb_col = '%s$Combined_Load_kg_per_m2_per_s$%s$%s' % (constituent, catchment, cgu)
            combined_cropping_ts[comb_col] = scaled_cropping[dis_col] + scaled_cropping[part_col]
        combined_cropping = pd.DataFrame(combined_cropping_ts)
        combined_cropping *= M2_TO_HA * PER_DAY_TO_PER_SECOND * G_TO_KG
        cropping_cgus = list(set(cropping_cgus))
        # print(combined_cropping.describe().transpose().describe())
        # Particulate_Load_g_per_Ha or Dissolved_Load_g_per_Ha
        return combined_copping,cropping_cgus

    def get_cropping_input_timeseries(self,cropping_df):
        cropping = cropping_df
        cropping_inputs = DataframeInputs()

        combined_copping,cropping_cgus = self._get_combined_cropping()
        if combined_copping is not None:
            cropping_inputs.inputter(combined_cropping, 'inputLoad',
                                    '${constituent}$$Combined_Load_kg_per_m2_per_s$$${catchment}$$${cgu}')

        def extract_cropping_columns(df, column_marker, conversion):
            columns = [c for c in df.columns if ('$%s$' % column_marker) in c]
            subset = df[columns].copy()
            subset *= conversion
            renames = {c: c.replace(column_marker, 'rescaled') for c in columns}
            subset = subset.rename(columns=renames)
            return subset, 'inputLoad', '${constituent}$$rescaled$$${catchment}$$${cgu}'

        cropping_inputs.inputter(
            *extract_cropping_columns(cropping, 'Constituent_Load_T_per_Ha', TONS_TO_KG * M2_TO_HA * PER_DAY_TO_PER_SECOND))
        cropping_inputs.inputter(
            *extract_cropping_columns(cropping, 'Surface_DIN_Load_g_per_Ha', G_TO_KG * M2_TO_HA * PER_DAY_TO_PER_SECOND))
        cropping_inputs.inputter(
            *extract_cropping_columns(cropping, 'Leached_DIN_Load_g_per_Ha', G_TO_KG * M2_TO_HA * PER_DAY_TO_PER_SECOND))

        crop_sed = cropping[[c for c in cropping.columns if 'Soil_Load_T_per_Ha' in c]]
        crop_sed *= M2_TO_HA * PER_DAY_TO_PER_SECOND * TONS_TO_KG
        # print(crop_sed.describe().transpose().describe())
        cropping_inputs.inputter(crop_sed, 'inputLoad', '${constituent}$$Soil_Load_T_per_Ha$$${catchment}$$${cgu}')

        return cropping_inputs,cropping_cgus

    def build_catchment_template(self,meta):
        catchment_template = DynamicSednetCatchment(dissolved_nutrients=meta['dissolved_nutrients'],
                                                    particulate_nutrients=meta['particulate_nutrients'],
                                                    pesticides=meta['pesticides'])  # pesticides)

        fus = meta['fus']
        catchment_template.hrus = fus
        catchment_template.cgus = fus
        catchment_template.cgu_hrus = {fu: fu for fu in fus}
        catchment_template.pesticide_cgus = meta['pesticide_cgus']
        catchment_template.hillslope_cgus = meta['usle_cgus']
        catchment_template.gully_cgus = meta['gully_cgus']
        catchment_template.sediment_fallback_cgu = meta['emc_cgus']

        catchment_template.rr = RR

        # def generation_model_for_constituent_and_fu(constituent, cgu=None):
        #     # progress(constituent,cgu)
        #     return default_cg[constituent]

        catchment_template.routing = ROUTING
        return catchment_template

    def assess_meta_structure(self,start,end,cropping):
        meta = {}
        meta['start'] = start
        meta['end'] = end

        fus = self._load_json('fus')
        meta['fus'] = fus

        constituents = self._load_json('constituents')
        meta['constituents'] = constituents

        # TODO: These are really 'cropping' CGUs. Need a better way to identify. OR not require them?
        pesticide_cgus = set([c.split('$')[-1] for c in cropping.columns if ('Dissolved_Load_g_per_Ha' in c) or ('Surface_DIN_Load_g_per_Ha' in c)])
        #pesticide_cgus = pesticide_cgus.union(set(fus).intersection({'Irrigated Cropping','Dryland Cropping'}))

        dissolved_nutrients = [c for c in constituents if '_D' in c or '_F' in c]
        meta['dissolved_nutrients'] = dissolved_nutrients

        particulate_nutrients = [c for c in constituents if '_Particulate' in c]
        meta['particulate_nutrients'] = particulate_nutrients

        sediments = [c for c in constituents if c.startswith('Sediment - ')]
        meta['sediments'] = sediments

        pesticides = [c for c in constituents if not c in dissolved_nutrients + particulate_nutrients + sediments]
        meta['pesticides'] = pesticides
        meta['pesticide_cgus'] = list(pesticide_cgus)

        print('pesticide_cgus',pesticide_cgus)
        cg_models = self._load_csv('cgmodels')
        cg_models = simplify(cg_models, 'model', ['Constituent', 'Functional Unit', 'Catchment'])

        def cgus_using_model(constituent,model):
            all_fus = cg_models[cg_models.Constituent==constituent]
            fus_using_model = list(set(all_fus[all_fus.model==model]['Functional Unit']))

            if not len(fus_using_model):
                return []

            all_models_on_fus = set(all_fus[all_fus['Functional Unit'].isin(fus_using_model)].model)
            is_homogenous = len(all_models_on_fus)==1
            if not is_homogenous:
                print('Looking for FUs using %s for constituent %s'%(model,constituent))
                print('Expected one model for fus(%s)/constituent(%s)'%(','.join(fus_using_model),constituent))
                print('Got: %s'%','.join(all_models_on_fus))
                print('FU/CGU instances: %d'%len(all_fus))
                print('FU/CGU instances using model: %d'%len(fus_using_model))

            assert is_homogenous
            return fus_using_model

        def cgus_using_one_of_models(constituent,models):
            return list(set(sum([cgus_using_model(constituent,m) for m in models],[])))

        # fine_sed_cg = cg_models[cg_models.Constituent == 'Sediment - Fine']
        # fine_sed_cg = dict(zip(fine_sed_cg['Functional Unit'], fine_sed_cg.model))
        # erosion_cgus = [fu for fu, model in fine_sed_cg.items() if
        #                 model == 'Dynamic_SedNet.Models.SedNet_Sediment_Generation']
        meta['emc_cgus'] = cgus_using_model(FINE_SEDIMENT,SOURCE_EMC_MODEL)
        meta['usle_cgus'] = cgus_using_model(FINE_SEDIMENT,DS_SEDIMENT_MODEL)
        meta['gully_cgus'] = cgus_using_one_of_models(FINE_SEDIMENT,GULLY_MODELS)
        meta['hillslope_emc_cgus'] = cgus_using_one_of_models(FINE_SEDIMENT,EMC_DWC_MODELS)

        return meta

    def _date_parameteriser(self,meta):
        date_components = [int(c) for c in meta['start'].split('/')]
        return DefaultParameteriser(node_types.DateGenerator, startYear=date_components[0],
                                    startMonth=date_components[1], startDate=date_components[2])

    def _climate_parameteriser(self,time_period):
        orig_climate = self._load_csv('climate')
        climate_ts = orig_climate.reindex(time_period)
        i = DataframeInputs()
        i.inputter(climate_ts, 'rainfall', 'rainfall for ${catchment}')
        i.inputter(climate_ts, 'pet', 'pet for ${catchment}')
        return i

    def _fu_areas_parameteriser(self):
        res = NestedParameteriser()
        fu_areas = self._load_csv('fu_areas')
        area_models = [
            ('DepthToRate','area'),
            ('PassLoadIfFlow','scalingFactor'),
            (node_types.USLEFineSedimentGeneration,'area'),
            (node_types.DynamicSednetGully,'Area'),
            (node_types.DynamicSednetGullyAlt,'Area'),
            (node_types.SednetParticulateNutrientGeneration,'area')
        ]
        for model,prop in area_models:
            res.nested.append(ParameterTableAssignment(fu_areas,model,prop,'cgu','catchment'))

        return res

    def _runoff_parameteriser(self):
        sacramento_parameters = self._load_csv('runoff_params')
        sacramento_parameters['hru'] = sacramento_parameters['Functional Unit']
        sacramento_parameters = sacramento_parameters.rename(columns={c: c.lower() for c in sacramento_parameters.columns})
        return ParameterTableAssignment(sacramento_parameters, RR, dim_columns=['catchment', 'hru'])

    def _routing_parameteriser(self,link_renames):
        routing_params = _rename_link_tag_columns(self._load_csv('routing_params'), link_renames)
        return ParameterTableAssignment(routing_params, ROUTING, dim_columns=['catchment']),routing_params

    def _constituent_generation_parameteriser(self,meta,cropping,time_period):
        res = NestedParameteriser()

        def apply_dataframe(df,model,complete=True):
            parameteriser = ParameterTableAssignment(df,
                                                     model,
                                                     dim_columns=['catchment', 'cgu', 'constituent'],
                                                     complete=complete)
            res.nested.append(parameteriser)

        cropping_inputs,cropping_cgus = self.get_cropping_input_timeseries(cropping)
        meta['cropping_cgus']=cropping_cgus
        # TODO NEED TO SCALE BY AREA!
        res.nested.append(cropping_inputs)

        usle_timeseries = self._load_csv('usle_timeseries').reindex(time_period)
        usle_timeseries = usle_timeseries.fillna(method='ffill')
        fine_sediment_params = self._load_param_csv('cg-Dynamic_SedNet.Models.SedNet_Sediment_Generation')
        assert set(fine_sediment_params.useAvModel) == {False}
        assert len(set(fine_sediment_params.constituent)) == 1
        fine_sediment_params = fine_sediment_params.rename(columns={
            'Max_Conc': 'maxConc',
            'USLE_HSDR_Fine': 'usleHSDRFine',
            'USLE_HSDR_Coarse': 'usleHSDRCoarse'
        })

        fine_sediment_params = fine_sediment_params.rename(
            columns={c: c.replace('_', '') for c in fine_sediment_params.columns})
        usle_parameters = ParameterTableAssignment(fine_sediment_params, node_types.USLEFineSedimentGeneration,
                                                   dim_columns=['catchment', 'cgu'])
        res.nested.append(usle_parameters)

        usle_timeseries_inputs = DataframeInputs()
        usle_timeseries_inputs.inputter(usle_timeseries, 'KLSC', 'KLSC_Total For ${catchment} ${cgu}')
        usle_timeseries_inputs.inputter(usle_timeseries, 'KLSC_Fine', 'KLSC_Fines For ${catchment} ${cgu}')
        usle_timeseries_inputs.inputter(usle_timeseries, 'CovOrCFact', 'C-Factor For ${catchment} ${cgu}')
        res.nested.append(usle_timeseries_inputs)

        gbr_crop_sed_params = self._load_param_csv('cg-GBR_DynSed_Extension.Models.GBR_CropSed_Wrap_Model')
        gbr_crop_sed_params = gbr_crop_sed_params.rename(
            columns={c: c.replace('_', '') for c in gbr_crop_sed_params.columns})

        gbr_emc_gully_params = self._load_param_csv('cg-Dynamic_SedNet.Models.SedNet_EMC_And_Gully_Model')
        for sed in ['Fine','Coarse']:
            emc_dwc_params = gbr_emc_gully_params.rename({
                sed.lower()+'EMC':'EMC',
                sed.lower()+'DWC':'DWC'
            }).copy()
            emc_dwc_params['constituent'] = 'Sediment - '+sed
            apply_dataframe(emc_dwc_params,node_types.EmcDwc,complete=False)

        gbr_emc_gully_params = gbr_emc_gully_params.rename(
            columns={c: c.replace('_', '') for c in gbr_emc_gully_params.columns})

        fine_sediment_params = pd.concat([fine_sediment_params, gbr_crop_sed_params,gbr_emc_gully_params], sort=False)

        fine_sediment_params = fine_sediment_params.rename(columns={
            'GullyYearDisturb': 'YearDisturbance',
            'AverageGullyActivityFactor': 'averageGullyActivityFactor',
            'GullyManagementPracticeFactor': 'managementPracticeFactor',
            'GullySDRFine': 'sdrFine',
            'GullySDRCoarse': 'sdrCoarse'
        })
        # Area, AnnualRunoff, GullyAnnualAverageSedimentSupply, annualLoad, longtermRunoffFactor
        # dailyRunoffPowerFactor
        gully_params = fine_sediment_params.fillna({
            'sdrFine': 100.0,
            'sdrCoarse': 100.0,
            'managementPracticeFactor': 1.0
        }).fillna(0.0)
        gully_parameters = ParameterTableAssignment(gully_params, node_types.DynamicSednetGullyAlt,
                                                    dim_columns=['catchment', 'cgu'])
        res.nested.append(gully_parameters)
        gully_parameters = ParameterTableAssignment(gully_params, node_types.DynamicSednetGully,
                                                    dim_columns=['catchment', 'cgu'])
        res.nested.append(gully_parameters)

        gully_timeseries = self._load_csv('gully_timeseries').reindex(time_period, method='ffill')
        gully_inputs = DataframeInputs()
        gully_ts_columns = ['Annual Load', 'Annual Runoff']
        gully_ts_destinations = ['annualLoad', 'AnnualRunoff']
        for col, input_name in zip(gully_ts_columns, gully_ts_destinations):
            gully_inputs.inputter(gully_timeseries, input_name, '%s For ${catchment} ${cgu}' % col)

        res.nested.append(gully_inputs)

        ts_load_hillslope_fine = gully_params.pivot('catchment', 'cgu', 'HillSlopeFinePerc') / 100.0
        hillslope_fine = ParameterTableAssignment(ts_load_hillslope_fine, node_types.FixedPartition,
                                                  parameter='fraction', column_dim='cgu', row_dim='catchment')
        res.nested.append(hillslope_fine)

        ts_sediment_scaling = compute_ts_sediment_scaling(gully_params)
        apply_dataframe(ts_sediment_scaling,node_types.ApplyScalingFactor,complete=False)


        emc_dwc = self._load_param_csv('cg-' + SOURCE_EMC_MODEL)
        if emc_dwc is not None:
            emc_dwc = emc_dwc.rename(columns={
                'eventMeanConcentration': 'EMC',
                'dryMeanConcentration': 'DWC'
            })
            apply_dataframe(emc_dwc,node_types.EmcDwc,complete=False)

        gbr_crop_sed_params = gbr_crop_sed_params.rename(columns={
            'Catchment': 'catchment',
            'Functional Unit': 'cgu',
        }) # TODO! Surely not necessary!

        crop_sed_fine_dwcs = gbr_crop_sed_params[['catchment', 'cgu', 'HillslopeFineDWC', 'HillslopeFineSDR']].copy()
        crop_sed_fine_dwcs['DWC'] = crop_sed_fine_dwcs['HillslopeFineDWC']  # * PERCENT_TO_FRACTION * crop_sed_fine_dwcs['HillslopeFineSDR']
        crop_sed_fine_dwcs['constituent'] = FINE_SEDIMENT
        apply_dataframe(crop_sed_fine_dwcs,node_types.EmcDwc,complete=False)

        crop_sed_coarse_dwcs = gbr_crop_sed_params[['catchment', 'cgu', 'HillslopeCoarseDWC', 'HillslopeCoarseSDR']].copy()
        crop_sed_coarse_dwcs['DWC'] = crop_sed_coarse_dwcs[
            'HillslopeCoarseDWC']  # * PERCENT_TO_FRACTION * crop_sed_coarse_dwcs['HillslopeCoarseSDR']
        crop_sed_coarse_dwcs['constituent'] = COARSE_SEDIMENT
        apply_dataframe(crop_sed_coarse_dwcs,node_types.EmcDwc,complete=False)

        dissolved_nutrient_params = self._load_param_csv('cg-Dynamic_SedNet.Models.SedNet_Nutrient_Generation_Dissolved')
        apply_dataframe(dissolved_nutrient_params,node_types.SednetDissolvedNutrientGeneration,complete=False)

        part_nutrient_params = self._load_param_csv('cg-Dynamic_SedNet.Models.SedNet_Nutrient_Generation_Particulate')
        apply_dataframe(part_nutrient_params,node_types.SednetParticulateNutrientGeneration,complete=False)

        sugarcane_din_params = self._load_param_csv('cg-GBR_DynSed_Extension.Models.GBR_DIN_TSLoadModel')
        apply_dataframe(sugarcane_din_params, node_types.EmcDwc,complete=False)

        sugarcane_din_params['scale'] = sugarcane_din_params['Load_Conversion_Factor'] * sugarcane_din_params[
            'DeliveryRatioSurface'] * PERCENT_TO_FRACTION
        apply_dataframe(sugarcane_din_params, node_types.ApplyScalingFactor,complete=False)

        sugarcane_leached_params = self._load_param_csv('cg-GBR_DynSed_Extension.Models.GBR_DIN_TSLoadModel')
        sugarcane_leached_params['constituent'] = 'NLeached'
        sugarcane_leached_params['scale'] = sugarcane_leached_params['Load_Conversion_Factor'] * sugarcane_din_params[
            'DeliveryRatioSeepage'] * PERCENT_TO_FRACTION
        apply_dataframe(sugarcane_leached_params, node_types.ApplyScalingFactor,complete=False)

        sugarcane_p_params = self._load_param_csv('cg-GBR_DynSed_Extension.Models.GBR_DissP_Gen_Model')
        if sugarcane_p_params is not None:
            sugarcane_p_params['PconcentrationMgPerL'] = 1e-3 * sugarcane_p_params['phos_saturation_index'].apply(
                lambda psi: (7.5 * psi) if psi < 10.0 else (-200.0 * 27.5 * psi))
            sugarcane_p_params['EMC'] = sugarcane_p_params['ProportionOfTotalP'] * sugarcane_p_params[
                'Load_Conversion_Factor'] * sugarcane_p_params['DeliveryRatioAsPercent'] * PERCENT_TO_FRACTION
            apply_dataframe(sugarcane_p_params, node_types.EmcDwc,complete=False)

        # OLD
        # if (phos_saturation_index < 10)
        # {
        #     PconcentrationMgPerL = 7.5 * phos_saturation_index / 1000.0;
        # }
        # else
        # {
        #     PconcentrationMgPerL = (-200 + 27.5 * phos_saturation_index) / 1000.0;
        # }

        # quickflowConstituent = quickflow * UnitConversion.CUBIC_METRES_TO_MEGA_LITRES * MEGA_LITRES_TO_LITRES * PconcentrationMgPerL * (1 / KG_TO_MILLIGRAM) * ProportionOfTotalP * Load_Conversion_Factor * DeliveryRatioAsPercent * ConversionConst.Percentage_to_Proportion;

        #   lag_parameters = ParameterTableAssignment(lag_outlet_links(network_veneer),node_types.Lag,dim_columns=['catchment'],complete=False)

        particulate_nut_gen = self._load_param_csv('cg-Dynamic_SedNet.Models.SedNet_Nutrient_Generation_Particulate')
        apply_dataframe(particulate_nut_gen, node_types.SednetParticulateNutrientGeneration,complete=False)

        ts_load_params = self._load_param_csv('cg-Dynamic_SedNet.Models.SedNet_TimeSeries_Load_Model')
        # Particulate_P - time series should load (already converted to kg/m2/s)
        # fu areas should load
        #
        ts_load_params['scale'] = ts_load_params['Load_Conversion_Factor'] * ts_load_params[
            'DeliveryRatio'] * PERCENT_TO_FRACTION
        apply_dataframe(ts_load_params, node_types.EmcDwc,complete=False)
        apply_dataframe(ts_load_params, node_types.ApplyScalingFactor,complete=False)
        # res.nested.append(
        #     ParameterTableAssignment(ts_load_params, node_types.EmcDwc, dim_columns=['catchment', 'cgu', 'constituent'],
        #                              complete=False))
        # res.nested.append(ParameterTableAssignment(ts_load_params, node_types.ApplyScalingFactor,
        #                                            dim_columns=['catchment', 'cgu', 'constituent'], complete=False))

        return res

    def _constituent_transport_parameteriser(self,link_renames,routing_params):
        res = NestedParameteriser()

        lag_parameters = ParameterTableAssignment(lag_non_routing_links(routing_params), node_types.Lag,
                                                  dim_columns=['catchment'], complete=False)
        res.nested.append(lag_parameters)

        instream_fine_sediment_params = _rename_link_tag_columns(
            self._load_csv('cr-Dynamic_SedNet.Models.SedNet_InStream_Fine_Sediment_Model'), link_renames, 'Link')
        link_attributes = instream_fine_sediment_params[
            ['catchment', 'LinkLength_M', 'LinkHeight_M', 'LinkWidth_M']].set_index('catchment')
        link_attributes = link_attributes.rename(
            columns={'LinkLength_M': 'linkLength', 'LinkHeight_M': 'linkHeight', 'LinkWidth_M': 'linkWidth'})

        instream_nutrient_params = _rename_link_tag_columns(
            self._load_csv('cr-Dynamic_SedNet.Models.SedNet_InStream_DissolvedNut_Model'), link_renames, 'Link')
        instream_nutrient_params['uptakeVelocity'] = instream_nutrient_params['UptakeVelocity']
        instream_nutrient_params = instream_nutrient_params.set_index('catchment')
        instream_nutrient_params = instream_nutrient_params.join(link_attributes, how='inner').reset_index()
        #   print(instream_nutrient_params)
        instream_nutrient_parameteriser = ParameterTableAssignment(instream_nutrient_params,
                                                                   'InstreamDissolvedNutrientDecay',
                                                                   dim_columns=['catchment'],
                                                                   complete=False)
        res.nested.append(instream_nutrient_parameteriser)

        instream_fine_sediment_params = instream_fine_sediment_params.rename(columns={
            'BankFullFlow': 'bankFullFlow',
            'BankHeight_M': 'bankHeight',
            'FloodPlainArea_M2': 'floodPlainArea',
            #   'LinkHeight_M':'bankHeight',
            'LinkLength_M': 'linkLength',
            'LinkWidth_M': 'linkWidth',
            'Link_Slope': 'linkSlope',
            'LongTermAvDailyFlow': 'longTermAvDailyFlow',
            'ManningsN': 'manningsN',
            'RiparianVegPercent': 'riparianVegPercent',
            'SoilErodibility': 'soilErodibility',
            'SoilPercentFine': 'soilPercentFine',
            #   'annualReturnInterval':'',
            #   'contribArea_Km':'',
            'initFineChannelStorProp': 'channelStoreFine'
        })
        instream_fine_sediment_params['channelStoreFine'] = - instream_fine_sediment_params['channelStoreFine']

        instream_fine_sed_parameteriser = ParameterTableAssignment(instream_fine_sediment_params, 'InstreamFineSediment',
                                                                   dim_columns=['catchment'],complete=False)
        res.nested.append(instream_fine_sed_parameteriser)
        bank_erosion_parameteriser = ParameterTableAssignment(instream_fine_sediment_params, 'BankErosion',
                                                              dim_columns=['catchment'],complete=False)
        res.nested.append(bank_erosion_parameteriser)

        return res

    def build_ow_model(self,start='1986/07/01', end='2014/06/30',
                       link_renames={},
                       progress=print):

        network = gpd.read_file(os.path.join(self.data_path, 'network.json'))
        network_veneer = _extend_network(self._load_json('network'))

        time_period = pd.date_range(start, end)

        cropping = self._load_csv('cropping')
        cropping = cropping.reindex(time_period)

        meta = self.assess_meta_structure(start,end,cropping)
        print(meta)
        # return meta
        # cr_models = self._load_csv('transportmodels')
        # cr_models = simplify(cr_models, 'model', ['Constituent'])

        # routing_models = self._load_csv('routing_models')
        # routing_models.replace(link_renames, inplace=True)

        # default_cg = catchment_template.cg.copy()

        # tpl_nested = catchment_template.get_template()
        # tpl = tpl_nested.flatten()

        # template_image = debugging.graph_template(tpl)
        # template_image.render('cgu_template', format='png')

        def setup_dates(g):
            date_tpl = OWTemplate()
            date_tags = {
                'calendar': 1
            }
            date_tags[TAG_PROCESS] = 'date_gen'

            date_tpl.add_node(node_types.DateGenerator, **date_tags)
            g = templating.template_to_graph(g, date_tpl)
            date_node = [n for n in g.nodes if g.nodes[n][TAG_MODEL] == 'DateGenerator'][0]
            usle_nodes = [n for n in g.nodes if g.nodes[n][TAG_MODEL] == 'USLEFineSedimentGeneration']
            # progress('USLE Nodes:', len(usle_nodes))
            for usle in usle_nodes:
                g.add_edge(date_node, usle, src=['dayOfYear'], dest=['dayOfYear'])

            return g

        catchment_template = self.build_catchment_template(meta)
        model = from_source.build_catchment_graph(catchment_template, network, progress=nop, custom_processing=setup_dates)
        progress('Model graph built')

        p = Parameteriser()
        p._parameterisers.append(DefaultParameteriser())
        p._parameterisers.append(self._date_parameteriser(meta))
        p._parameterisers.append(self._climate_parameteriser(time_period))
        p._parameterisers.append(self._fu_areas_parameteriser())
        p._parameterisers.append(self._runoff_parameteriser())
        p._parameterisers.append(self._constituent_generation_parameteriser(meta,cropping,time_period))

        rp,routing_params = self._routing_parameteriser(link_renames)
        p._parameterisers.append(rp)
        p._parameterisers.append(self._constituent_transport_parameteriser(link_renames,routing_params))

        model._parameteriser = p
        progress('Model parameterisation established')

        return model, meta, network

def nop(*args,**kwargs):
    pass


def lag_outlet_links(network):
    outlets = [n['properties']['id'] for n in network.outlet_nodes()]
    links_to_outlets = reduce(lambda x,y: list(x) + list(y),
                              [network['features'].find_by_to_node(n) for n in outlets])
    single_link_networks = [l for l in links_to_outlets if len(network.upstream_links(l))==0]
    outlet_links = [l['properties']['name'] for l in single_link_networks]
    print('Outlet links',len(outlet_links),outlet_links)
    return lag_links(outlet_links)

def lag_links(links):
    return pd.DataFrame([{'catchment':l.replace('link for catchment ',''),'timeLag':1} for l in links])

def lag_headwater_links(network):
    links = network['features'].find_by_feature_type('link')
    headwater_links = [l for l in links if len(network.upstream_links(l))==0]
    headwater_link_names = [l['properties']['name'] for l in headwater_links]
    print('Headwater links',len(headwater_link_names),headwater_link_names)
    return lag_links(headwater_link_names)

def lag_non_routing_links(params):
    RC_THRESHOLD=1e-5
    links_to_lag = list(params[params.RoutingConstant<RC_THRESHOLD].catchment)
    #print('Links to lag',len(links_to_lag),links_to_lag)
    return lag_links(links_to_lag)

