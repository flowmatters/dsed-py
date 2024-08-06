from openwater.template import OWLink
import openwater.nodes as n
from openwater.config import DataframeInputs, ParameterTableAssignment, UniformParameteriser

def apply_daily_musle_with_runoff_rebalancing(catchment_tpl,**kwargs):
    EXISTING_MODEL='USLEFineSedimentGeneration'
    NEW_MODEL='MUSLEFineSedimentGeneration'
    cgus = {n.label.split(':')[1]:n for n in catchment_tpl.nested if n.label != 'reach'}
    cgus = {c:n for c,n in cgus.items() if len(n.matching_nodes(nested=True,_model=EXISTING_MODEL))}
    # + len(n.matching_nodes(nested=True,_model=NEW_MODEL))
    # print(cgus.keys())

    # Switch to MUSLE
    for cgu,cgu_tpl in cgus.items():
        usle = cgu_tpl.get_node(True,_model=EXISTING_MODEL)
        usle.set_model_type(NEW_MODEL)
    assert len(catchment_tpl.matching_nodes(True,_model=EXISTING_MODEL))==0
    assert len(catchment_tpl.matching_nodes(True,_model=NEW_MODEL))==len(cgus)

    # Add cover scaling
    runoff_scaling = catchment_tpl.add_node(model_type='MUSLERunoffScaling',process='runoff_scaling',**kwargs)
    precip = catchment_tpl.get_node(False,_model='Input',variable='rainfall')

    # Add adjust cover
    for cgu,cgu_tpl in cgus.items():
        ro_tpl = cgu_tpl.nested[0]
        gen_tpl = cgu_tpl.nested[1]
        assert gen_tpl.label.startswith('cgu:')

        cover_node = gen_tpl.add_node(model_type=n.MUSLECoverMetric,process='cover_scaling',cgu=cgu,**kwargs)
        # runoff_node = gen_tpl.get_node(False,_model='DepthToRate',component='Runoff')
        runoff_node = ro_tpl.get_node(False,_model='Sacramento')

        musle_node = gen_tpl.get_node(False,_model=NEW_MODEL)


        catchment_tpl.add_link(OWLink(precip,'output',cover_node,'P'))
        catchment_tpl.add_link(OWLink(cover_node,'cover_metric',runoff_scaling,'precipMetric'))
        # catchment_tpl.add_link(OWLink(runoff_node,'outflow',runoff_scaling,'totalFlow'))
        catchment_tpl.add_link(OWLink(runoff_node,'surfaceRunoff',runoff_scaling,'totalFlow'))
        catchment_tpl.add_link(OWLink(runoff_scaling,'Rcm',musle_node,'Rcm'))

    return catchment_tpl

def apply_event_based_musle(catchment_tpl,**kwargs):
    EXISTING_MODEL='USLEFineSedimentGeneration'
    NEW_MODEL='CoreUSLE'

    cgus = {n.label.split(':')[1]:n for n in catchment_tpl.nested if n.label != 'reach'}
    cgus = {c:n for c,n in cgus.items() if len(n.matching_nodes(nested=True,_model=EXISTING_MODEL))}
    # + len(n.matching_nodes(nested=True,_model=NEW_MODEL))
    # print(cgus.keys())

    # Switch to MUSLE
    for cgu,cgu_tpl in cgus.items():
        usle = cgu_tpl.get_node(True,_model=EXISTING_MODEL)
        usle.set_model_type(NEW_MODEL)
    assert len(catchment_tpl.matching_nodes(True,_model=EXISTING_MODEL))==0
    assert len(catchment_tpl.matching_nodes(True,_model=NEW_MODEL))==len(cgus)

    precip = catchment_tpl.get_node(False,_model='Input',variable='rainfall')

    # Add adjust cover
    for cgu,cgu_tpl in cgus.items():
        ro_tpl = cgu_tpl.nested[0]
        gen_tpl = cgu_tpl.nested[1]
        assert gen_tpl.label.startswith('cgu:')

        rfactor_node = gen_tpl.add_node(model_type=n.MUSLEEventBasedRFactor,process='rfactor',cgu=cgu,**kwargs)
        # runoff_node = gen_tpl.get_node(False,_model='DepthToRate',component='Runoff')
        runoff_node = ro_tpl.get_node(False,_model='Sacramento')

        musle_node = gen_tpl.get_node(False,_model=NEW_MODEL)

        catchment_tpl.add_link(OWLink(precip,'output',rfactor_node,'rainfall'))
        catchment_tpl.add_link(OWLink(runoff_node,'surfaceRunoff',rfactor_node,'totalFlow'))
        catchment_tpl.add_link(OWLink(rfactor_node,'R',musle_node,'rFactor'))

    return catchment_tpl

def parameterise_musle_runoff_rebalancing(model,vis_cover,cfactor,cgu_area,catchment_centroids):
  timeseries_inputs = DataframeInputs()
  model._parameteriser.append(timeseries_inputs)
  MUSLE_MOD='MUSLEFineSedimentGeneration'
  for mod in [MUSLE_MOD,'MUSLECoverMetric']:
      timeseries_inputs.inputter(
          vis_cover,
          'cover',
          '${catchment}-${cgu}-visCov',
          model=mod)

  timeseries_inputs.inputter(
      cfactor,
      'cFactor',
      '${catchment}-${cgu}-cf',
      model=MUSLE_MOD)

  model._parameteriser.append(ParameterTableAssignment(
      cgu_area,
      'MUSLECoverMetric',
      dim_columns=['catchment','cgu']
  ))

  model._parameteriser.append(ParameterTableAssignment(
      catchment_centroids[['catchment','latitude']],
      'MUSLEFineSedimentGeneration',
      dim_columns=['catchment']
  ))

  for mod in ['MUSLECoverMetric','MUSLEFineSedimentGeneration']:
      model._parameteriser.append(UniformParameteriser(
          mod,
          gamma=0.02,
          cr=0.6
      ))

  model._parameteriser.append(UniformParameteriser(
      'MUSLEFineSedimentGeneration',
      a = 89.45,
      b1 = 0.56,
      b2 = 0.56
  ))

def parameterise_musle_event(model,cfactor):
  MUSLE_MOD='CoreUSLE'
  timeseries_inputs = DataframeInputs()
  model._parameteriser.append(timeseries_inputs)
  timeseries_inputs.inputter(
      cfactor,
      'cFactor',
      '${catchment}-${cgu}-cf',
      model=MUSLE_MOD)
