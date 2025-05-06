import pandas as pd
import logging
from ..const import PERCENT_TO_FRACTION

logger = logging.getLogger(__name__)

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

