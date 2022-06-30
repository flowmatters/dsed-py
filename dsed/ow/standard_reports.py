from openwater.file import _tabulate_model_scalars_from_file
import os
import shutil
import pandas as pd
import numpy as np
from itertools import product
from dsed.const import *

M3_PER_S_TO_L_PER_DAY = PER_SECOND_TO_PER_DAY * M3_TO_L

PRIMARY_TAGS={
  'Catchment':['catchment','link_name'],
  'Link':['catchment','link_name'], # NOT link_name
  'Node':['node_name']
}

VALUE_COL='Total_Load_in_Kg'

ONE_OFF_CONSTITUENT_TRANSPORT_MODELS = {
  'InstreamFineSediment':'Sediment - Fine',
  'InstreamCoarseSediment':'Sediment - Coarse'
}

STANDARD_TRANSPORT_FLUX_NAMES = dict(
  ConstituentDecay={
    'Yield':'outflowLoad',
    'In Flow':'inflowLoad'
  },
  InstreamFineSediment={
    'Yield':'loadDownstream',
    'In Flow':'upstreamMass',
    # 'Streambank':'reachLocalMass'
  },
  InstreamCoarseSediment={
    'Yield':'loadDownstream',
    'In Flow':'upstreamMass',
    # 'Streambank':'reachLocalMass'
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
  InstreamDissolvedNutrientDecay='totalStoredMass',
  ConstituentDecay='storedMass'
)

def channel_mob_transform(val):
  return val.map(lambda v: 0.0 if v > 0 else abs(v))

class TimeSeriesLineItem(object):
  def __init__(self, model, model_variable, element_type, budget_element, process,
               conversion_factor=PER_SECOND_TO_PER_DAY,transform=None,timeseries_filter=None,**tags):
    self.model = model
    self.variable = model_variable
    self.conversion_factor = conversion_factor
    self.element_type = element_type
    self.budget_element = budget_element
    self.process = process
    self.transform = transform
    self.tags = tags
    self.filter = timeseries_filter

  def __call__(self, results):
    ow_results = results.results
    query_tags = {k:v for k,v in self.tags.items() if k in ow_results.dims_for_model(self.model)}
    all_timeseries = ow_results.all_time_series(self.model,self.variable,**query_tags)
    primary_tags = PRIMARY_TAGS[self.element_type]
    tables = []
    for tag in primary_tags:
      if not tag in all_timeseries.columns.names:
        continue

      subset = all_timeseries.loc[:,(all_timeseries.columns.get_level_values(tag)!=f'dummy-{tag}')].copy()
      columns = subset.columns
      for other_tag in primary_tags:
        if other_tag==tag: continue
        if other_tag not in columns.names: continue
        columns = columns.droplevel(other_tag)
      subset.columns = columns
      summary = subset.sum() * self.conversion_factor
      if self.transform is not None:
        summary = self.transform(summary)

      summary = summary.reset_index()
      for other_tag in primary_tags:
        if other_tag==tag:continue
        if other_tag in summary.columns:
          print(summary)
          print(other_tag,tag)
          assert False

      for k,v in self.tags.items():
        summary[k] = v
      summary = augment_line_item(self,summary)
      if self.filter is not None:
        summary = summary[summary.apply(self.filter,axis=1)]

      tables.append(summary)
    return pd.concat(tables)

class StateLineItem(object):
  def __init__(self, selection,model, variable, element_type, budget_element, process, constituent,conversion_factor=1.0,**tags):
    self.selection = selection
    self.model = model
    self.variable = variable
    self.conversion_factor = conversion_factor
    self.element_type = element_type
    self.budget_element = budget_element
    self.process = process
    self.constituent = constituent
    self.tags = tags

  def __call__(self, results):
    states =  results.get_states[self.selection](self.model)
    tables = []
    for tag in PRIMARY_TAGS[self.element_type]:
      tbl = states[~states[tag].str.startswith('dummy')].copy()
      tbl[self.variable] *= self.conversion_factor
      tbl = tbl.rename(columns={tag:'ModelElement',self.variable:VALUE_COL})
      if 'constituent' in tbl.columns:
        tbl['Constituent']=tbl['constituent']
        if self.constituent is not None:
          tbl = tbl[tbl.Constituent==self.constituent]
      else:
        tbl['Constituent']=self.constituent
      tables.append(tbl[['ModelElement','Constituent',VALUE_COL]])

    result = pd.concat(tables)
    for k,v in self.tags.items():
      result[k] = v
    return augment_line_item(self,result)

FIXED_LINE_ITEMS = [
  TimeSeriesLineItem('StorageRouting','outflow','Link','Link Yield','Yield',constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
  TimeSeriesLineItem('StorageRouting','inflow','Link','Link In Flow','In Flow',constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
  StateLineItem('final','StorageRouting','S','Link','Residual Link Storage','Residual','Flow',conversion_factor=M3_TO_L),
  StateLineItem('initial','StorageRouting','S','Link','Link Initial Load','Supply','Flow',conversion_factor=M3_TO_L),
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
        self._raw_results_table = None

    def _get_states(self,f,model,**tags):
        mmap = self.model._map_model_dims(model)
        return _tabulate_model_scalars_from_file(f,model,mmap,'states',**tags)

    def get_final_states(self,model,**tags):
        f = self.results.results
        return self._get_states(f,model,**tags)

    def get_initial_states(self,model,**tags):
        f = self.results.model
        return self._get_states(f,model,**tags)

    def _links_to_outlets(self):
      outlets = self.impl.network.outlet_nodes()
      outlets = [o for o in outlets if 'Confluence' in o['properties']['icon']]
      final_links = [l['properties']['name'].replace('link for catchment ','') \
        for l in sum([self.impl.network.upstream_links(n['properties']['id'])._list for n in outlets],[])]
      assert len(final_links)==len(outlets)
      return {l:o['properties']['name'] for l,o in zip(final_links,outlets)}

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
            # fu_rows['FU']=fu
            # fu_rows['Total_Load_in_Kg'] = 0.0
            # ss_by_fu = pd.concat([ss_by_fu,fu_rows])
        for c in meta['sediments']:
            extra_rows += [
                dict(BudgetElement='Flood Plain Deposition',Constituent=c,FU='Stream',Total_Load_in_Kg=0.0)
            ]
        tbl = pd.concat([tbl,pd.DataFrame(extra_rows)]).drop_duplicates(subset=columns,keep='first')
        return tbl

    def source_sink_per_fu_summary_table(self,region=None):
        DROP_ELEMENTS=[
            'Residual Node Storage',
            'Node Initial Load',
            'Node Injected Mass',
            'Node Yield',
            'Extraction',
            'DWC Contributed Seepage',
            'TimeSeries Contributed Seepage',
            'Leached'
        ]

        raw = self.raw_summary_table(region)
        df = raw.copy()

        headwater_catchments = [sc['properties']['name'] for sc in self.impl.network.headwater_catchments()]
        df = df[(df.BudgetElement!='Link In Flow')|(df.ModelElement.isin(headwater_catchments))]

        df.loc[df['FU'].isin(['Link','Node']),'FU']='Stream'
        ss_by_fu = df.groupby(['FU','Constituent','BudgetElement']).sum(numeric_only=True).reset_index()
        ss_by_fu = ss_by_fu[~ss_by_fu.Constituent.isin(self.impl.meta['pesticides'])]
        ss_by_fu = ss_by_fu[ss_by_fu.Constituent!='Flow']
        ss_by_fu = ss_by_fu[~ss_by_fu.BudgetElement.isin(DROP_ELEMENTS)]
        ss_by_fu = self.augment_source_sink_fu_table(ss_by_fu)
        return ss_by_fu

    def raw_summary_table(self,region=None):
      if self._raw_results_table is None:
        self._raw_results_table = self._compute_raw_summary_table(region)
      return self._raw_results_table

    def _compute_raw_summary_table(self,region=None):
      links = self.link_summary_table(region)
      outlets = self.outlet_node_summary_table(links)

      catchments = self.catchment_summary_table(region)
      nodes = self.node_summary_table(region)

      return pd.concat([
        catchments,
        nodes,
        links,
        outlets
      ])

    def catchment_summary_table(self,region=None):
      line_items = [
        ('USLEFineSedimentGeneration','totalFineLoad','Hillslope surface soil','Supply',{'constituent':'Sediment - Fine'}), # Hillslope no source distinction
        ('DynamicSednetGullyAlt','fineLoad','Gully','Supply',{'constituent':'Sediment - Fine'}),
        ('USLEFineSedimentGeneration','totalCoarseLoad','Hillslope surface soil','Supply',{'constituent':'Sediment - Coarse'}), # will be Hillslope no source distinction
        ('DynamicSednetGullyAlt','coarseLoad','Gully','Supply',{'constituent':'Sediment - Coarse'}),
        ('SednetDissolvedNutrientGeneration','totalLoad','Diffuse Dissolved','Supply',{}),
        # ('EmcDwc','totalLoad','Undefined','Supply',{'cgu':'Water'}),
        ('SednetParticulateNutrientGeneration','slowflowConstituent','Undefined','Supply',{}),
        ('SednetParticulateNutrientGeneration','hillslopeContribution','Hillslope no source distinction','Supply',{}),
        ('SednetParticulateNutrientGeneration','gullyContribution','Gully','Supply',{}),
      ]

      line_items += [
        ('Sum','i1','Hillslope surface soil','Supply',{'cgu':'Sugarcane','constituent':f'Sediment - {c}'}) \
          for c in ['Fine', 'Coarse']
      ]

      line_items += [
        ('ApplyScalingFactor','output','Leached','Other',{'cgu':'Sugarcane','constituent':'NLeached'})
      ] # NOT QUITE. NEEDS CONVERTED LOAD BEFORE DELIVERY RATIO!

      line_items += [
        ('EmcDwc','slowLoad','Undefined','Supply',{'cgu':'Sugarcane','constituent':'N_DON'})
      ] # N_DON possibly totalLoad?

      line_items += [
        ('ApplyScalingFactor','output','TimeSeries Contributed Seepage','Other',{'cgu':'Sugarcane','constituent':'NLeached'}),
        ('EmcDwc','slowLoad','DWC Contributed Seepage','Other',{'cgu':'Sugarcane','constituent':'N_DIN'})# GOOD
      ]

      hillslope_emc_cgus = [
        'Dryland Cropping',
        'Irrigated Cropping',
        'Horticulture',
        'Other',
        'Urban'
      ]
      for cgu in hillslope_emc_cgus:
        line_items+= [
          ('EmcDwc','totalLoad','Hillslope surface soil','Supply',{'cgu':cgu,'constituent':f'Sediment - {c}'}) \
            for c in ['Fine', 'Coarse']]

      line_items = [TimeSeriesLineItem(mod,variable,'Catchment',budget,process,**tags) \
        for mod,variable,budget,process,tags in line_items]

      def emc_undefined_supply_filter(row):
        if row.FU=='Water':
          return True
        if (row.Constituent in self.impl.meta['pesticides']) and (row.FU not in self.impl.meta['pesticide_cgus']):
          return True
        return False
      line_items += [
        TimeSeriesLineItem('EmcDwc','totalLoad','Catchment','Undefined','Supply',timeseries_filter=emc_undefined_supply_filter)
      ]

      def filter_sugarcane_seepage(row):
        if row.Constituent in ['P_DOP','P_FRP']:
          return True
        if row.Constituent in self.impl.meta['pesticides']:
          return True
        return False
      line_items += [
        TimeSeriesLineItem('EmcDwc','slowLoad','Catchment','Seepage','Supply',timeseries_filter=filter_sugarcane_seepage,cgu='Sugarcane'),
        TimeSeriesLineItem('EmcDwc','quickLoad','Catchment','Hillslope no source distinction','Supply',timeseries_filter=filter_sugarcane_seepage,cgu='Sugarcane')
      ]

      line_items += [
        TimeSeriesLineItem('Sum','i2','Catchment','Seepage','Supply',cgu='Sugarcane',constituent='N_DIN'),
        TimeSeriesLineItem('ApplyScalingFactor','output','Catchment','Hillslope no source distinction','Supply',cgu='Sugarcane',constituent='N_DIN')
      ]

      return pd.concat([
        self._evaluate_line_items(line_items),
        self._zero_lines(Constituent=['Sediment - Fine','Sediment - Coarse'],
                         FU='Water',
                         BudgetElement='Undefined',
                         Process='Supply')
      ])

    def _zero_lines(self,ModelElementType='Catchment',**kwargs):
      kwargs['ModelElementType'] = ModelElementType
      if ('ModelElement' not in kwargs):
        if ModelElementType=='Catchment':
          kwargs['ModelElement'] = [c for c in self.impl.results.dim('catchment') if c != 'dummy-catchment']
        elif ModelElementType=='Storage':
          kwargs['ModelElementType'] = 'Node'
          kwargs['ModelElement'] = list(self.impl.model.parameters('Storage')['node_name'])

      columns = list(kwargs.keys())
      vals = [kwargs[c] for c in columns]
      vals = [[v] if isinstance(v, str) else v for v in vals]
      combos = product(*vals)
      table = pd.DataFrame(combos,columns=columns)
      table['Total_Load_in_Kg'] = 0.0
      return table

    def node_summary_table(self,region=None):
      return pd.concat([
        self.storage_summary_table(),
        self.extraction_summary_table(),
        self.inflow_summary_table()
      ])

    def link_summary_table(self,region=None):
      line_items = FIXED_LINE_ITEMS[:]

      transport_models = [(c,self.impl.transport_model(c)[0]) for c in self.impl.meta['constituents']]
      for c,model in transport_models:
        if model is None:
          continue
        items = transport_line_items(model,c)
        # for i in items: i.constituent = c
        line_items += items

      line_items += [
        TimeSeriesLineItem('BankErosion',v,'Catchment','Streambank','Supply',constituent=c,FU='Stream') \
          for c,v in [('Sediment - Fine','bankErosionFine'),('Sediment - Coarse','bankErosionCoarse')]
      ]
      line_items += [
        TimeSeriesLineItem('InstreamParticulateNutrient','loadFromStreambank','Catchment','Streambank','Supply',FU='Stream')
      ]

      line_items += [
        TimeSeriesLineItem('InstreamFineSediment','loadToFloodplain','Link','Flood Plain Deposition','Loss',constituent='Sediment - Fine',FU='Link'),
        StateLineItem('final','InstreamFineSediment','channelStoreFine','Link','Stream Deposition','Loss','Sediment - Fine',FU='Link'),
        TimeSeriesLineItem('InstreamFineSediment','loadToChannelDeposition','Catchment','Channel Remobilisation','Supply',transform=channel_mob_transform,constituent='Sediment - Fine',FU='Stream'),
        TimeSeriesLineItem('InstreamParticulateNutrient','loadToFloodplain','Link','Flood Plain Deposition','Loss',FU='Link'),
        StateLineItem('final','InstreamCoarseSediment','channelStore','Link','Stream Deposition','Loss','Sediment - Coarse',FU='Link'),
        StateLineItem('final','InstreamParticulateNutrient','channelStoredMass','Link','Stream Deposition','Loss',None,FU='Link'),
        TimeSeriesLineItem('InstreamParticulateNutrient','loadDeposited','Catchment','Channel Remobilisation','Supply',transform=channel_mob_transform,FU='Stream'),
      ]

      line_items += [
        TimeSeriesLineItem('InstreamDissolvedNutrientDecay','loadFromPointSource','Catchment','Point Source','Supply',FU='Stream'),
        TimeSeriesLineItem('InstreamDissolvedNutrientDecay','decayedLoad','Link','Stream Decay','Loss',FU='Link')
      ]

      line_items += [
        TimeSeriesLineItem('ConstituentDecay','decayedLoad','Link','Stream Decay','Loss',FU='Link')
      ]

      df = self._evaluate_line_items(line_items)
      return pd.concat([
        df,
        self._zero_lines(Constituent='Sediment - Coarse',FU='Stream',BudgetElement='Channel Remobilisation',Process='Supply')
      ])

    def storage_summary_table(self):
      line_items = [
        TimeSeriesLineItem('Storage','outflow','Node','Node Yield','Yield',Constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
        TimeSeriesLineItem('Storage','rainfallVolume','Node','Rainfall','Supply',Constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
        TimeSeriesLineItem('Storage','evaporationVolume','Node','Evaporation','Loss',Constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
        StateLineItem('initial','Storage','currentVolume','Node','Node Initial Load','Supply','Flow',conversion_factor=M3_TO_L),
        StateLineItem('final','Storage','currentVolume','Node','Residual Node Storage','Residual','Flow',conversion_factor=M3_TO_L)
      ]

      line_items += [
        TimeSeriesLineItem('LumpedConstituentRouting','outflowLoad','Node','Node Yield','Yield'),
        TimeSeriesLineItem('StorageParticulateTrapping','outflowLoad','Node','Node Yield','Yield'),
        TimeSeriesLineItem('StorageParticulateTrapping','trappedMass','Node','Reservoir Deposition','Loss',conversion_factor=1.0),
        StateLineItem('initial','StorageParticulateTrapping','storedMass','Node','Node Initial Load','Supply',None),
        StateLineItem('final','StorageParticulateTrapping','storedMass','Node','Residual Node Storage','Residual',None),
        StateLineItem('initial','LumpedConstituentRouting','storedMass','Node','Node Initial Load','Supply',None),
        StateLineItem('final','LumpedConstituentRouting','storedMass','Node','Residual Node Storage','Residual',None)
      ]
      df = self._evaluate_line_items(line_items)
      return pd.concat([
        df,
        self._zero_lines('Storage',
                         BudgetElement='Reservoir Decay',
                         Process='Loss',
                         FU='Node',
                         Constituent=['N_DIN','N_DON','P_FRP','P_DOP']+self.impl.meta['pesticides']),
        self._zero_lines('Storage',
                         BudgetElement='Infiltration',
                         Process='Loss',
                         FU='Node',
                         Constituent='Flow')
      ])

    def extraction_summary_table(self):
      line_items = [
        TimeSeriesLineItem('PartitionDemand',v,'Node',be,p,constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY) \
          for (v,be,p) in \
            [
              ('outflow','Node Yield','Yield'),
              ('extraction','Extraction','Loss')
            ]
      ]

      line_items += [
        TimeSeriesLineItem('VariablePartition',v,'Node',be,p) \
          for (v,be,p) in \
            [
              ('output2','Node Yield','Yield'),
              ('output1','Extraction','Loss')
            ]
      ]

      return self._evaluate_line_items(line_items)

    def inflow_summary_table(self):
      line_items = [
        TimeSeriesLineItem('Input','output','Node','Node Injected Inflow','Supply',variable='inflow',Constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
        TimeSeriesLineItem('Input','output','Node','Node Yield','Yield',variable='inflow',Constituent='Flow',conversion_factor=M3_PER_S_TO_L_PER_DAY),
        TimeSeriesLineItem('PassLoadIfFlow','inputLoad','Node','Node Injected Mass','Supply',catchment='dummy-catchment',FU='Node'),
        TimeSeriesLineItem('PassLoadIfFlow','outputLoad','Node','Node Yield','Yield',catchment='dummy-catchment',FU='Node'),
      ]

      return self._evaluate_line_items(line_items)

    def outlet_node_summary_table(self,link_table:pd.DataFrame) -> pd.DataFrame:
      lookup = self._links_to_outlets()
      rows = link_table[(link_table.BudgetElement=='Link Yield')&(link_table.ModelElement.isin(lookup.keys()))].copy()
      rows['ModelElement'] = rows['ModelElement'].map(lookup)
      rows.BudgetElement = 'Node Yield'
      rows.FU = 'Node'
      rows.ModelElementType = 'Node'
      return rows

    def _evaluate_line_items(self,line_items):
      if not len(line_items):
        return pd.DataFrame()

      results = [li(self) for li in line_items]
      return pd.concat(results)

def transport_line_items(model,constituent):
  tags = {}
  # if model in ONE_OFF_CONSTITUENT_TRANSPORT_MODELS:
  #   tags['constituent'] = ONE_OFF_CONSTITUENT_TRANSPORT_MODELS[model]
  # else:
  tags['constituent'] = constituent
  items = [TimeSeriesLineItem(model,v,'Link','Link '+n,n,**tags) \
    for n,v in STANDARD_TRANSPORT_FLUX_NAMES[model].items()]
  if model in STANDARD_TRANSPORT_STORE_NAMES:
    items += [
      StateLineItem('initial',model,STANDARD_TRANSPORT_STORE_NAMES[model],'Link','Link Initial Load','Supply',constituent),
      StateLineItem('final',model,STANDARD_TRANSPORT_STORE_NAMES[model],'Link','Residual Link Storage','Residual',constituent)
    ]

  return items


def augment_line_item(li:TimeSeriesLineItem,df:pd.DataFrame):
  location_tags = {'catchment','link_name','node_name'}
  if len(set(df.columns).intersection(location_tags)) > 1:
    for lt in location_tags:
      if lt not in df.columns: continue
      if set(df[lt])=={f'dummy-{lt}'}:
        df = df.drop(columns=[lt])

  if len(set(df.columns).intersection(location_tags)) > 1:
    print(li.model,li.variable,li.process,li.budget_element,li.element_type)
    print(df)
    print(df.dtypes)
    assert False

  if 'cgu' in df.columns and 'FU' in df.columns:
    df = df.drop(columns=['cgu'])

  df = df.rename(columns={
    'constituent':'Constituent',
    'hru':'FU',
    'cgu':'FU',
    'catchment':'ModelElement',
    'link_name':'ModelElement',
    'node_name':'ModelElement',
    0:'Total_Load_in_Kg'
  })

  if 'ModelElement' not in df.columns:
    print(li.model,li.variable,li.process,li.budget_element,li.element_type)
    print(df)

  try:
    df = df[~df.ModelElement.str.startswith('dummy-')].copy()
  except:
    print(li.model,li.variable,li.process,li.budget_element,li.element_type)
    print(df)
    raise

  if (li.element_type == 'Link') and ('FU' not in df.columns):
    df['FU'] = 'Stream'

  df['BudgetElement']=li.budget_element
  df['Process']=li.process
  df['ModelElementType']=li.element_type
  df['Constituent'] = df['Constituent'].map(lambda c: 'N_DIN' if c=='NLeached' else c)

  drop_columns = ['variable']
  for col in drop_columns:
    if col in df.columns:
      df = df.drop(columns=[col])

  if ('FU' not in df.columns) and set(df['ModelElementType']=={'Node'}):
    df['FU'] = 'Node'

  if set(df.columns) != {'FU','ModelElement','BudgetElement','Process','ModelElementType','Constituent','Total_Load_in_Kg'}:
    print(li.model,li.variable,li.process,li.budget_element,li.element_type)
    print(df)
    raise Exception('Invalid columns')

  EFFECTIVELY_ZERO=1e-15 # was 1e-6, then 1e-7
  df.loc[abs(df.Total_Load_in_Kg)<EFFECTIVELY_ZERO,'Total_Load_in_Kg']=0.0

  return df
