from .. import const as c
import logging
import pandas as pd
logger = logging.getLogger(__name__)


UNITS={
    'Flow':('ML/d',c.CUMECS_TO_ML_DAY,None),
    'Sediment - Fine':('kt',c.KG_TO_KTONS,c.KTONS_PER_ML_TO_MG_PER_L),
    'Sediment - Coarse':('kt',c.KG_TO_KTONS,c.KTONS_PER_ML_TO_MG_PER_L),
    'N_Particulate':('t',c.KG_TO_TONS,c.TONS_PER_ML_TO_MG_PER_L),
    'N_DIN':('t',c.KG_TO_TONS,c.TONS_PER_ML_TO_MG_PER_L),
    'N_DON':('t',c.KG_TO_TONS,c.TONS_PER_ML_TO_MG_PER_L),
    'P_Particulate':('t',c.KG_TO_TONS,c.TONS_PER_ML_TO_MG_PER_L),
    'P_FRP':('t',c.KG_TO_TONS,c.TONS_PER_ML_TO_MG_PER_L),
    'P_DOP':('t',c.KG_TO_TONS,c.TONS_PER_ML_TO_MG_PER_L)
}

WATERFALL_MAP={
    'Total':'none',
    'Supply':'relative',
    'Loss':'-relative',
    'Residual':'-relative',
    'Export':'-relative',
    'Other':'-relative'
}

def constituent_units(con):
    return UNITS.get(con,('kg',1.0,1.0))

def pc_of_total(df,process,vs_process=None):
    if vs_process is None:
        vs_process = process

    result = 100.0*df[process].transpose()/df['Total', vs_process]
    result['Process'] = process
    result = result.reset_index()
    result = result.set_index(['Process','BudgetElement'])
    return result

class MassBalanceBuilder(object):
    def __init__(self,runs,dataset):
        self.ds = dataset
        self.runs = runs

    def build(self):
        logger.info('Building mass balance summaries')
        for run in self.runs:
            self.build_run(run)

    def build_run(self,run_name):
        results = gbr.Results(run_name)
        region = run_name.split('_')[0]
        logger.info(f'{region}: Loading run {run_name}')
        raw = results.get('RawResults')
        mba_final, mass_balance_percentages, mass_balance_loss_v_supply = self.build_run_for_raw_results(raw,results.runDetails.yearsOfRecording)

        tags = {
            'run':run_name,
            'region':region,
            'timestep':'mean annual'
        }
        self.ds.add_table(mba_final,purpose='mass-balance',**tags)
        self.ds.add_table(mass_balance_percentages,purpose='mass-balance-percentages',**tags)
        self.ds.add_table(mass_balance_loss_v_supply,purpose='mass-balance-loss-v-supply',**tags)

    def build_run_for_raw_results(self,raw,years):
        EXCLUDED_BUDGET_ELEMENTS=[
            'Link Intial Load',
            'Link In Flow',
            'Link Yield',
            'Node Yield',
            'Evaporation',
            'Infiltration',
            'Rainfall',
            'Node Injected Inflow'
        ]

        #Remove budget elements not required (the last 3 elements refer to flow)
        filteredRaw = raw[~raw.BudgetElement.isin(EXCLUDED_BUDGET_ELEMENTS)]

        grouped_rawResults = filteredRaw.reset_index().groupby(['Process', 'BudgetElement', 'Constituent']).sum()

        #Groupd values by budget element for each constituent
        massBalanceKg = pd.pivot_table(grouped_rawResults.reset_index(),index = ['Process','BudgetElement'], columns = 'Constituent', values = 'Total_Load_in_Kg')
        #Convert from a total to an annual result
        massBalanceKgAnnual = massBalanceKg/years

        #Convert units
        massBalanceAnnual = massBalanceKgAnnual.copy()
        for col in massBalanceAnnual.columns:
            massBalanceAnnual[col] *= constituent_units(col)[1]

        #Reorder colums to the expected arrangement
        cols = list(massBalanceAnnual)
        cols.insert(0, cols.pop(cols.index('Sediment - Fine')))
        cols.insert(1, cols.pop(cols.index('Sediment - Coarse')))
        cols.insert(2, cols.pop(cols.index('P_Particulate')))
        cols.insert(3, cols.pop(cols.index('P_FRP')))
        cols.insert(4, cols.pop(cols.index('P_DOP')))
        cols.insert(5, cols.pop(cols.index('N_Particulate')))
        cols.insert(6, cols.pop(cols.index('N_DIN')))
        cols.insert(7, cols.pop(cols.index('N_DON')))
        massBalanceAnnual = massBalanceAnnual.loc[:, cols]

        #Drop Pesticides and Flow
        #massBalanceAnnual.drop(['Atrazine', 'Diuron', 'Hexazinone', 'S-metolachlor', 'Tebuthiuron', 'Flow'], axis=1, inplace=True)
        massBalanceAnnual.drop(['Flow'], axis=1, inplace=True)

        #Transpose the table so totals can be calculated
        massBalanceAnnual_transposed=massBalanceAnnual.transpose()


        #Calculate Totals - there must be a more elegant way of doing this!
        massBalanceAnnual_transposed['Total', 'Supply'] = massBalanceAnnual_transposed['Supply'].fillna(0.0).sum(axis=1)
        massBalanceAnnual_transposed['Total', 'Loss'] = massBalanceAnnual_transposed['Loss'].fillna(0.0).sum(axis=1)
        massBalanceAnnual_transposed['Total', 'Residual'] = massBalanceAnnual_transposed['Residual'].fillna(0.0).sum(axis=1)

        massBalanceAnnual_transposed['Total', 'Export'] = massBalanceAnnual_transposed['Total', 'Supply'].fillna(0) - \
                                                        massBalanceAnnual_transposed['Total', 'Loss'].fillna(0) - \
                                                        massBalanceAnnual_transposed['Total', 'Residual'].fillna(0)

        # Total loss vs supply
        # Floodplain depo vs supply

        mass_balance_percentages = pd.concat([
            pc_of_total(massBalanceAnnual_transposed,'Supply'),
            pc_of_total(massBalanceAnnual_transposed,'Loss'),
            pc_of_total(massBalanceAnnual_transposed,'Residual'),

        ])

        mass_balance_loss_v_supply = pc_of_total(massBalanceAnnual_transposed,'Loss','Supply')
        mass_balance_loss_v_supply = mass_balance_loss_v_supply.transpose()
        mass_balance_loss_v_supply['Loss','Total'] = mass_balance_loss_v_supply.fillna(0.0).sum(axis=1)
        mass_balance_loss_v_supply = mass_balance_loss_v_supply.transpose()

        massBalanceAnnual_transposed2 = massBalanceAnnual_transposed.transpose()
        #massBalanceAnnual_transposed2

        #Swap the order of the rows
        processOrder = ["Supply", "Other", "Loss", "Residual", "Total"]
        processIndex = sorted(massBalanceAnnual_transposed2.index, key=lambda x: processOrder.index(x[0]))
        mba_reordered = massBalanceAnnual_transposed2.reindex(processIndex)
        budElmntOrder = [
            "Hillslope surface soil",
            "Hillslope sub-surface soil",
            "Hillslope no source distinction",
            "Gully",
            "Streambank",
            "Point Source",
            "Node Injected Mass",
            "Diffuse Dissolved",
            "Undefined",
            "Channel Remobilisation",
            "Link Initial Load",
            "Node Initial Load",
            "Seepage",
            "Reservoir Decay",
            "Extraction",
            "Flood Plain Deposition",
            "Node Loss",
            "Reservoir Deposition",
            "Stream Decay",
            "Stream Deposition",
            "DWC Contributed Seepage", # Added
            "Leached", # Added
            "TimeSeries Contributed Seepage", # Added
            "Residual Link Storage",
            "Residual Node Storage",
            "Supply",
            "Loss",
            "Residual",
            "Export"
        ]
        extra_terms = set([x[1] for x in list(mba_reordered.index)]) - set(budElmntOrder)
        if len(extra_terms) > 0:
            # logger.info(extra_terms)
            logger.warning("Extra terms in mass balance: %s",','.join(sorted(extra_terms)))

        budElmntIndex = sorted(mba_reordered.index, key=lambda x: budElmntOrder.index(x[1]))
        mba_reordered2 = mba_reordered.reindex(budElmntIndex)


        #Format and display the final table
        # mba_final = mba_reordered2.where(pd.notnull(mba_reordered2), "")
        mba_final = mba_reordered2.copy()
        mba_final.rename(columns=lambda con: f'{con} ({constituent_units(con)[0]})',inplace=True)
        mba_final['waterfall'] = mba_final.index.map(lambda x: '-relative' if x[1]=='Export' else WATERFALL_MAP[x[0]])

        #mba_final = mba_reordered2.fillna(0)
        #pd.options.display.float_format = '{:,.2f}'.format
        #pd.options.display.float_format = '{0:g}'.format
        return mba_final, mass_balance_percentages, mass_balance_loss_v_supply

