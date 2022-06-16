from openwater.file import _tabulate_model_scalars_from_file
import os
import shutil
import pandas as pd
import numpy as np
from dsed.const import *

PRIMARY_TAGS={
  'Catchment':['catchment'],
  'Link':['catchment'], # NOT link_name
  'Node':['node_name']
}

VALUE_COL='Total_Load_in_Kg'

STANDARD_TRANSPORT_FLUX_NAMES = dict(
  ConstituentDecay={
    'Yield':'outflowLoad',
    'In Flow':'inflowLoad'
  },
  InstreamFineSediment={
    'Yield':'loadDownstream',
    'In Flow':'upstreamMass',
    'Streambank':'reachLocalMass'
  },
  InstreamCoarseSediment={
    'Yield':'loadDownstream',
    'In Flow':'upstreamMass',
    'Streambank':'reachLocalMass'
  },
  InstreamParticulateNutrient={
    'Yield':'loadDownstream',
    'In Flow':'incomingMassUpstream'
  },
  InstreamDissolvedNutrientDecay={
    'Yield':'loadDownstream',
    'In Flow':'incomingMassUpstream'
  }
)

STANDARD_TRANSPORT_STORE_NAMES = dict(
  InstreamFineSediment='totalStoredMass',
  InstreamCoarseSediment='totalStoredMass',
  InstreamParticulateNutrient='instreamStoredMass',
  InstreamDissolvedNutrientDecay='totalStoredMass'
)


def augment_line_item(li,df):
  df = df[~df.ModelElement.str.startswith('dummy-')].copy()
  if li.element_type == 'Link':
    df['FU'] = 'Stream'

  df['BudgetElement']=li.budget_element
  df['Process']=li.process
  df['ModelElementType']=li.element_type
  return df

class TimeSeriesLineItem(object):
  def __init__(self, model, variable, conversion_factor, element_type, budget_element, process, constituent):
    self.model = model
    self.variable = variable
    self.conversion_factor = conversion_factor
    self.element_type = element_type
    self.budget_element = budget_element
    self.process = process
    self.constituent = constituent

  def __call__(self, results):
    tables = []
    for tag in PRIMARY_TAGS[self.element_type]:
      if self.constituent is None:
      # Can call table by catchment/link and constituent
        res = results.results.table(self.model,self.variable,tag,'constituent','sum','sum')
        res = res.melt()
        res = res.rename(columns={tag:'ModelElement','constituent':'Constituent',0:VALUE_COL})
      else:
        if self.constituent=='Flow':
          res = results.results.time_series(self.model,self.variable,tag,'sum').sum().reset_index()
        else:
          res = results.results.time_series(self.model,self.variable,tag,'sum',constituent=self.constituent).sum().reset_index()
        res = res.rename(columns={'index':'ModelElement',0:VALUE_COL})
        res['Constituent']=self.constituent
      tables.append(res)

    return augment_line_item(self,pd.concat(tables))

class StateLineItem(object):
  def __init__(self, selection,model, variable, conversion_factor, element_type, budget_element, process, constituent):
    self.selection = selection
    self.model = model
    self.variable = variable
    self.conversion_factor = conversion_factor
    self.element_type = element_type
    self.budget_element = budget_element
    self.process = process
    self.constituent = constituent

  def __call__(self, results):
    states =  results.get_states[self.selection](self.model)
    tables = []
    for tag in PRIMARY_TAGS[self.element_type]:
      tbl = states[~states[tag].str.startswith('dummy')].copy()
      tbl = tbl.rename(columns={tag:'ModelElement',self.variable:VALUE_COL})
      if 'constituent' in tbl.columns:
        tbl['Constituent']=tbl['constituent']
        if self.constituent is not None:
          tbl = tbl[tbl.Constituent==self.constituent]
      else:
        tbl['Constituent']=self.constituent
      tables.append(tbl[['ModelElement','Constituent',VALUE_COL]])

    return augment_line_item(self,pd.concat(tables))

FIXED_LINE_ITEMS = [
  TimeSeriesLineItem('StorageRouting','outflow',PER_SECOND_TO_PER_DAY,'Link','Link Yield','Yield','Flow'),
  TimeSeriesLineItem('StorageRouting','inflow',PER_SECOND_TO_PER_DAY,'Link','Link In Flow','In Flow','Flow'),
  StateLineItem('final','StorageRouting','S',1.0,'Link','Residual Link Storage','Residual','Flow'),
  StateLineItem('initial','StorageRouting','S',1.0,'Link','Link Initial Load','Supply','Flow'),
]

class DynamicSednetStandardReporting(object):
    def __init__(self,ow_impl):
        self.impl = ow_impl
        self.results = ow_impl.results
        self.model = ow_impl.model
        self.get_states = dict(
          initial=self.get_initial_states,
          final=self.get_final_states
        )

    def _get_states(self,f,model,**tags):
        mmap = self.model._map_model_dims(model)
        return _tabulate_model_scalars_from_file(f,model,mmap,'states',**tags)

    def get_final_states(self,model,**tags):
        f = self.results.results
        return self._get_states(f,model,**tags)

    def get_initial_states(self,model,**tags):
        f = self.results.model
        return self._get_states(f,model,**tags)

    def outlet_nodes_time_series(self,dest,overwrite=False):
        if os.path.exists(dest):
            if overwrite and os.path.isdir(dest):
                shutil.rmtree(dest)
            else:
                raise Exception("Destination exists")
        os.makedirs(dest)

        outlets = self.impl.network.outlet_nodes()
        final_links = [l['properties']['name'].replace('link for catchment ','') \
            for l in sum([self.impl.network.upstream_links(n['properties']['id'])._list for n in outlets],[])]
        assert len(final_links)==len(outlets)

        total_fn = os.path.join(dest,'TotalDaily_%s_ModelTotal_%s.csv')

        flow_l = self.results.time_series('StorageRouting','outflow','catchment')[final_links]*PER_SECOND_TO_PER_DAY * M3_TO_L
        for outlet,final_link in zip(outlets,final_links):
            fn = os.path.join(dest,f'node_flow_{outlet["properties"]["name"]}_Litres.csv')
            flow_l[final_link].to_csv(fn)
        flow_l.sum(axis=1).to_csv(total_fn%('Flow','Litres'))
        for c in self.impl.meta['constituents']:
            mod, flux = self.impl.transport_model(c)
            constituent_loads_kg = self.results.time_series(mod,flux,'catchment',constituent=c)[final_links]*PER_SECOND_TO_PER_DAY
            for outlet,final_link in zip(outlets,final_links):
                fn = os.path.join(dest,f'link_const_{outlet["properties"]["name"]}_{c}_Kilograms.csv')
                constituent_loads_kg[final_link].to_csv(fn)
            constituent_loads_kg.sum(axis=1).to_csv(total_fn%(c,'Kilograms'))

    def outlet_nodes_rates_table(self):
        outlets = [n['properties']['id'] for n in self.impl.network.outlet_nodes()]
        final_links = [l['properties']['name'].replace('link for catchment ','') for l in sum([self.impl.network.upstream_links(n)._list for n in outlets],[])]
        flow_l = np.array(self.results.time_series('StorageRouting','outflow','catchment')[final_links])*PER_SECOND_TO_PER_DAY * M3_TO_L
        total_area = sum(self.model.parameters('DepthToRate',component='Runoff').area)
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
        orig_tbls = []
        melted_tbls = []
        variables = [('rainfall','Rainfall'),('actualET','Actual ET'),('baseflow','Baseflow'),('runoff','Runoff (Quickflow)')]
        for v,lbl in variables:
            tbl = self.results.table('Sacramento',v,'catchment','hru','sum','sum')*MM_TO_M
            if v=='runoff':
                tbl = tbl - orig_tbls[-1]
            orig_tbls.append(tbl)
            tbl = tbl.reset_index().melt(id_vars=['index'],value_vars=list(tbl.columns)).rename(columns={'index':'Catchment','variable':'FU','value':'Depth_m'})
            tbl['Element']=lbl
            melted_tbls.append(tbl)
        return pd.concat(melted_tbls).sort_values(['Catchment','FU','Element'])

    def fu_areas_table(self):
        tbl = self.model.parameters('DepthToRate',component='Runoff')
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

    def regional_summary_table(self):
        'Not implemented'
        tables = [self.mass_balance_summary_table(self,region) for region in self.impl.meta['regions']]
        for tbl,region in zip(tables,self.impl.meta['regions']):
            tbl['SummaryRegion']=region
        return pd.concat(tables)

    def overall_summary_table(self):
        return self.mass_balance_summary_table()

    def _get_final_constituent_store(self,model,store,constituent,region=None):
      values = self.get_final_states(model)
      if constituent is not None:
          values['constituent']=constituent
      tbl = values[['constituent',store]].groupby('constituent').sum().reset_index().rename(columns={
          'constituent':'Constituent',
          store:'Total_Load_in_Kg'
      })
      return tbl

    def constituent_loss_table(self,region=None):
      loss_fluxes = [
          ('InstreamFineSediment','loadToFloodplain','Sediment - Fine'),
          ('InstreamDissolvedNutrientDecay','loadToFloodplain',None),
          ('InstreamParticulateNutrient','loadToFloodplain',None)
          # ('InstreamDissolvedNutrientDecay','loadDecayed',{}), #TODO
      ]

      loss_tables = []
      for model, flux, constituent in loss_fluxes:
        if constituent is None:
          tbl = self.results.time_series(model,flux,'constituent','sum').sum() * PER_SECOND_TO_PER_DAY
        else:
          tbl = pd.DataFrame({constituent:[self.results.time_series(model,flux,'catchment').sum().sum() * PER_SECOND_TO_PER_DAY]})
          tbl = tbl.transpose()[0]
        loss_tables.append(tbl)

      loss_states = [
          ('InstreamFineSediment','channelStoreFine','Sediment - Fine'),
          ('InstreamParticulateNutrient','channelStoredMass',None),
          ('InstreamCoarseSediment','totalStoredMass', 'Sediment - Coarse')
      ]
      for model, state, constituent in loss_states:
        loss_tables.append(self._get_final_constituent_store(model,state,constituent,region).set_index('Constituent')['Total_Load_in_Kg'])
      overall = pd.DataFrame(pd.concat(loss_tables)).reset_index().groupby('index').sum()
      overall = overall.reset_index().rename(columns={'index':'Constituent',0:'Total_Load_in_Kg'})
      return overall

    def residual_constituent_table(self,region=None):
        # Need to query final states (and initial states?)
        mass_states = [
            ('LumpedConstituentRouting','storedMass'),
            ('InstreamFineSediment','totalStoredMass', 'Sediment - Fine'),
            ('InstreamDissolvedNutrientDecay','totalStoredMass')
        ]
        tables = []
        for state in mass_states:
            m = state[0]
            v = state[1]
            constituent = state[2] if len(state)>2 else None
            tables.append(self._get_final_constituent_store(m,v,constituent,region))
        return pd.concat(tables).groupby('Constituent').sum().reset_index()

    def mass_balance_summary_table(self,region=None):
        cols =['Constituent','Total_Load_in_Kg']
        input_tables = {
            'Supply':self.fu_summary_table(),
            'Export':self.outlet_nodes_rates_table(),
            'Loss':self.constituent_loss_table(region),
            'Residual':self.residual_constituent_table(region)
        }

        result = []
        for k,tbl in input_tables.items():
            if tbl is None:
                print(f'Missing table {k}')
                continue
            tbl = tbl[cols].groupby('Constituent').sum().reset_index()
            tbl['MassBalanceElement'] = k
            tbl = tbl[['Constituent','MassBalanceElement','Total_Load_in_Kg']]
            result.append(tbl)

        return pd.concat(result).sort_values(['Constituent','MassBalanceElement'])

    def source_sink_per_fu_summary_table(self,region=None):
      pass

    def raw_summary_table(self,region=None):
      line_items = FIXED_LINE_ITEMS[:]

      transport_models = [(c,self.impl.transport_model(c)[0]) for c in self.impl.meta['constituents']]
      for c,model in transport_models:
        if model is None:
          continue
        items = transport_line_items(model)
        for i in items: i.constituent = c
        line_items += items

      return self._evaluate_line_items(line_items)

    def _evaluate_line_items(self,line_items):
      results = [li(self) for li in line_items]
      return pd.concat(results)

def transport_line_items(model):
  items = [TimeSeriesLineItem(model,v,PER_SECOND_TO_PER_DAY,'Link','Link '+n,n,None) \
    for n,v in STANDARD_TRANSPORT_FLUX_NAMES[model].items()]
  if model in STANDARD_TRANSPORT_STORE_NAMES:
    items += [
      StateLineItem('initial',model,STANDARD_TRANSPORT_STORE_NAMES[model],1.0,'Link','Link Initial Load','Supply',None),
      StateLineItem('final',model,STANDARD_TRANSPORT_STORE_NAMES[model],1.0,'Link','Residual Link Storage','Residual',None)
    ]

  return items


