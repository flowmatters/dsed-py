'''
Dynamic SedNet standard reports from Openwater results, in the formats written by Dynamic SedNet in Source.

The reports are assembled from budget tables (dsed.ow.budgets), which describe the results in Openwater's
own terms, and converted to Source's conventions by dsed.ow.source_format.
'''
import os
import shutil
import pandas as pd
import numpy as np
from dsed.const import *
from dsed.ow.reporting import OpenwaterDynamicSednetResults
from dsed.ow.budgets import Budget
from dsed.ow import source_format
import logging
logger = logging.getLogger(__name__)

VALUE_COL = source_format.VALUE_COL

def reports_for(model_fn, outputs_fn=None,start=None,end=None):
    '''
    Factory function to create a DynamicSednetStandardReporting object for a given model and outputs file.

    Args:
        model_fn (str): Path to the model file.
        outputs_fn (str, optional): Path to the outputs file. If not provided, it will be inferred from the model file name.
        start (str or pd.Timestamp, optional): Start date for filtering time series data. Defaults to None (no filtering).
        end (str or pd.Timestamp, optional): End date for filtering time series data. Defaults to None (no filtering).

    Returns:
        DynamicSednetStandardReporting: An instance of the DynamicSednetStandardReporting class initialized
    '''
    r = OpenwaterDynamicSednetResults(model_fn, outputs_fn)
    return DynamicSednetStandardReporting(r, start=start, end=end)

class DynamicSednetStandardReporting(object):
    def __init__(self,ow_impl:OpenwaterDynamicSednetResults,start=None,end=None):
        self.impl = ow_impl
        self.results = ow_impl.results
        self.model = ow_impl.model
        self.budget = Budget(ow_impl,start=start,end=end)
        self._raw_results_table = None

    @property
    def start(self):
        return self.budget.start

    @property
    def end(self):
        return self.budget.end

    @property
    def time_period(self):
        '''(start, end) of the reporting window, for use with OpenwaterResults time_period arguments.'''
        return self.budget.time_period

    def _outlet_links(self):
        outlets = self.impl.outlets()
        return [o['name'] for o in outlets], [o['catchment'] for o in outlets]

    def outlet_nodes_time_series(self,dest,overwrite=False):
        if os.path.exists(dest):
            if overwrite and os.path.isdir(dest):
                shutil.rmtree(dest)
            else:
                raise Exception("Destination exists")
        os.makedirs(dest)

        outlets, final_links = self._outlet_links()
        total_fn = os.path.join(dest,'TotalDaily_%s_ModelTotal_%s.csv')

        flow_l = self.results.time_series('StorageRouting','outflow','catchment')[final_links]*PER_SECOND_TO_PER_DAY * M3_TO_L
        for outlet,final_link in zip(outlets,final_links):
            fn = os.path.join(dest,f'node_flow_{outlet}_Litres.csv')
            flow_l[final_link].to_csv(fn)
        flow_l.sum(axis=1).to_csv(total_fn%('Flow','Litres'))
        for c in self.impl.meta['constituents']:
            mod, flux = self.impl.transport_model(c)
            constituent_loads_kg = self.results.time_series(mod,flux,'catchment',constituent=c)[final_links]*PER_SECOND_TO_PER_DAY
            for outlet,final_link in zip(outlets,final_links):
                fn = os.path.join(dest,f'link_const_{outlet}_{c}_Kilograms.csv')
                constituent_loads_kg[final_link].to_csv(fn)
            constituent_loads_kg.sum(axis=1).to_csv(total_fn%(c,'Kilograms'))

    def outlet_nodes_rates_table(self):
        _, final_links = self._outlet_links()
        flow_l = np.array(self.results.time_series('StorageRouting','outflow','catchment')[final_links])*PER_SECOND_TO_PER_DAY * M3_TO_L
        total_area = sum(self.model.model.parameters('DepthToRate',component='Runoff').area)
        records = []
        for c in self.impl.meta['constituents']:
            mod, flux = self.impl.transport_model(c)
            constituent_loads_kg = np.array(self.results.time_series(mod,flux,'catchment',constituent=c)[final_links])*PER_SECOND_TO_PER_DAY
            records.append(dict(
                Region='ModelTotal',
                Constituent=c,
                Area=total_area,
                Total_Load_in_Kg=constituent_loads_kg.sum(),
                Flow_Litres=flow_l.sum(),
                Concentration=0.0,
                LoadPerArea=0.0,
                NumDays=flow_l.shape[0]
            ))
        return pd.DataFrame(records)

    def climate_table(self):
        areas = self.fu_areas_table().rename(columns={'Catchment':'catchment','CGU':'cgu'})
        return source_format.climate_table(self.budget.climate_table(),areas)

    def fu_areas_table(self):
        tbl = self.model.model.parameters('DepthToRate',component='Runoff')
        tbl = tbl[['catchment','cgu','area']].sort_values(['catchment','cgu']).rename(columns={'catchment':'Catchment','cgu':'CGU'})
        return tbl

    def fu_summary_table(self):
        summary = []
        seen = {}
        for con in self.impl.meta['constituents']:
            for fu in self.impl.meta['fus']:
                combo = self.impl.generation_model(con,fu)
                if not combo in seen:
                    model,flux = combo
                    if model is None:
                      continue
                    tbl = self.results.table(model,flux,'constituent','cgu','sum','sum') * PER_SECOND_TO_PER_DAY
                    seen[combo]=tbl
                tbl = seen[combo]
                summary.append((con,fu,tbl.loc[con,fu]))
        return pd.DataFrame(summary,columns=['Constituent','FU','Total_Load_in_Kg'])

    def runoff_volume_table(self,component='Runoff'):
        flow_ts = self.results.all_time_series('DepthToRate','outflow',component=component)
        return (flow_ts*PER_SECOND_TO_PER_DAY*M3_TO_L).sum().reset_index().rename(columns={'catchment':'ModelElement','cgu':'FU',0:'Flow_Litres'})

    def fu_rates_table(self):
        area_table = self.fu_areas_table().rename(columns={'Catchment':'ModelElement','CGU':'FU','area':'Area'})
        c_summary_table = self.catchment_summary_table()
        c_summary_table = c_summary_table[c_summary_table.Process=='Supply']
        c_summary_table = c_summary_table.groupby(['ModelElement','FU','Constituent']).sum().reset_index()

        flow = self.runoff_volume_table()

        merged = pd.merge(c_summary_table,area_table,on=['ModelElement','FU'],how='left')
        merged = pd.merge(merged,flow,on=['ModelElement','FU'],how='left')
        merged['LoadPerArea']=(merged['Total_Load_in_Kg']/merged['Area']).fillna(0.0)
        merged['Concentration']=(merged['Total_Load_in_Kg']/merged['Flow_Litres']).fillna(0.0)
        return merged

    def regional_summary_table(self):
        'Not implemented'
        tables = [self.mass_balance_summary_table(self,region) for region in self.impl.meta['regions']]
        for tbl,region in zip(tables,self.impl.meta['regions']):
            tbl['SummaryRegion']=region
        return pd.concat(tables)

    def overall_summary_table(self):
        return self.mass_balance_summary_table()

    def mass_balance_summary_table(self,region=None):
        '''
        Supply, Loss, Residual and Export of each constituent, as in Source's OverallSummaryTable.
        '''
        outlets = {o['name'] for o in self.impl.outlets()}
        return source_format.overall_summary(self.raw_summary_table(region),outlets)

    def augment_source_sink_fu_table(self,tbl):
        fus = set(tbl.FU) - {'Stream'}
        columns = set(tbl.columns) - {'Total_Load_in_Kg'}
        meta = self.impl.meta
        extra_rows=[]
        for fu in fus:
            for dn in meta['dissolved_nutrients']:
                extra_rows += [
                    dict(BudgetElement=be,Constituent=dn,FU=fu,Total_Load_in_Kg=0.0) \
                    for be in ['Diffuse Dissolved','Undefined']
                ]
            for c in meta['particulate_nutrients'] + meta['sediments']:
                extra_rows += [
                    dict(BudgetElement=be,Constituent=c,FU=fu,Total_Load_in_Kg=0.0) \
                    for be in ['Hillslope surface soil','Hillslope no source distinction','Undefined','Gully']
                ]

        for c in (set(meta['constituents'])-set(meta['pesticides'])):
            extra_rows += [
                dict(BudgetElement=be,Constituent=c,FU='Stream',Total_Load_in_Kg=0.0) \
                for be in ['Node Loss','Stream Decay']
            ]

        for c in meta['dissolved_nutrients']:
            extra_rows += [
                dict(BudgetElement='Denitrification',Constituent=c,FU='Stream',Total_Load_in_Kg=0.0)
            ]
        for c in meta['sediments']:
            extra_rows += [
                dict(BudgetElement='Flood Plain Deposition',Constituent=c,FU='Stream',Total_Load_in_Kg=0.0)
            ]
        tbl = pd.concat([tbl,pd.DataFrame(extra_rows)]).drop_duplicates(subset=columns,keep='first')
        return tbl

    def source_sink_per_fu_summary_table(self,region=None,include_extraction=False,inflow_headwaters_only=True):
        DROP_ELEMENTS=[
            'Residual Node Storage',
            'Node Initial Load',
            'Node Injected Mass',
            'Node Yield',
            'DWC Contributed Seepage',
            'TimeSeries Contributed Seepage',
            'Leached'
        ]

        if not include_extraction:
          DROP_ELEMENTS.append('Extraction')

        raw = self.raw_summary_table(region)
        df = raw.copy()

        if inflow_headwaters_only:
          headwater_catchments = [sc['properties']['name'] for sc in self.impl.network.headwater_catchments()]
          df = df[(df.BudgetElement!='Link In Flow')|(df.ModelElement.isin(headwater_catchments))]

        df.loc[df['FU'].isin(['Link','Node']),'FU']='Stream'
        ss_by_fu = df.groupby(['FU','Constituent','BudgetElement']).sum(numeric_only=True).reset_index()
        ss_by_fu = ss_by_fu[~ss_by_fu.Constituent.isin(self.impl.meta['pesticides'])]
        ss_by_fu = ss_by_fu[ss_by_fu.Constituent!='Flow']
        ss_by_fu = ss_by_fu[~ss_by_fu.BudgetElement.isin(DROP_ELEMENTS)]
        ss_by_fu = self.augment_source_sink_fu_table(ss_by_fu)
        return ss_by_fu

    def source_sink_summary_table(self,region=None):
      ss_by_fu = self.source_sink_per_fu_summary_table(region,include_extraction=True,inflow_headwaters_only=False)
      source_sink_summary = ss_by_fu.groupby(['Constituent','BudgetElement']).sum(numeric_only=True).reset_index()
      return source_sink_summary

    def raw_summary_table(self,region=None):
      '''Source's RawResults table.'''
      if self._raw_results_table is None:
        self._raw_results_table = source_format.raw_results(self.budget)
      return self._raw_results_table

    def catchment_summary_table(self,region=None):
      '''RawResults rows for constituent generation.'''
      raw = self.raw_summary_table(region)
      return raw[(raw.ModelElementType=='Catchment')&(raw.FU!='Stream')]

    def link_summary_table(self,region=None):
      '''RawResults rows for links (including the in-stream processes Source reports against catchments).'''
      raw = self.raw_summary_table(region)
      return raw[(raw.ModelElementType=='Link')|((raw.ModelElementType=='Catchment')&(raw.FU=='Stream'))]

    def node_summary_table(self,region=None):
      '''RawResults rows for nodes.'''
      raw = self.raw_summary_table(region)
      return raw[raw.ModelElementType=='Node']
