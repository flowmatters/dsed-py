import os
import logging
from openwater import clip as ow_clip
from openwater.template import ModelFile
import json

def clip_ds_meta(meta,new_dims):
  new_meta = {}
  for key,values in meta.items():
    if key=='fus' or key.endswith('cgus'):
      new_values = [v for v in values if v in new_dims['cgu']]
    elif key=='constituents' or key=='sediments' or key.endswith('nutrients'):
      new_values = [v for v in values if v in new_dims['constituent']]
    else:
      new_values = values
    new_meta[key] = new_values
  return new_meta

def clip_to_network_nodes(model, dest_fn, nodes, constituents, exact_node_match=False):
  '''
  Clip a Dynamic Sednet Openwater Model to one or more downstream nodes, filtering out
  elements related to particular constituents (+ flow)

  model: OpenwaterDynamicSednetModel
  dest_fn: str
    Destination file name
  nodes: list of str
    Node names to clip to (possibly partial names, eg a gauge number)
  constituents: list of str
    Constituents to keep
  '''
  catchments = [model.catchment_for_node(
    n, exact=exact_node_match) for n in nodes]
  logging.info('Clipping to %d catchments', len(catchments))
  return clip_to_catchments(model, dest_fn, catchments, constituents)


def clip_to_catchments(model, dest_fn, catchments, constituents):
  '''
  Clip a Dynamic Sednet Openwater Model to one or more catchments, filtering out
  elements related to particular constituents (+ flow)
  
  model: OpenwaterDynamicSednetModel
  dest_fn: str
    Destination file name
  catchments: list of str
    Catchment names to clip to
  constituents: list of str
    Constituents to keep
  '''
  clip_points = []
  for catchment in catchments:
    clip_points.append(('StorageRouting', {'catchment': catchment}))
    for constituent in constituents:
      tags = {'catchment': catchment}
      model_type = model.transport_model(constituent)[0]
      if 'constituent' in model.model.dims_for_model(model_type):
        tags['constituent'] = constituent
      clip_points.append((model_type, tags))

  logging.info('Clipping to %d clip points', len(clip_points))
  return clip_to_tagged_models(model, dest_fn, clip_points)

def clip_to_tagged_models(model,dest_fn,clip_points):
  nodes = []
  for model_type, tags in clip_points:
    node_match = model.model.nodes_matching(model_type, **tags)
    node_ids = list(node_match._run_idx)
    assert len(node_ids) == 1
    nodes += [(model_type, node_id) for node_id in node_ids]
  
  return clip(model, dest_fn, nodes)

def clip(model, dest_fn, clip_to):
  ow_clip.clip(model.model,dest_fn,clip_to)

  new_model = ModelFile(dest_fn)
  new_dims = new_model._dimensions

  with open(dest_fn.replace('.h5', '.meta.json'), 'w') as fp:
    json.dump(clip_ds_meta(model.meta, new_dims), fp, indent=2, default=str)
  catchments = model.catchments[model.catchments.name.isin(new_dims['catchment'])]
  links = model.links[model.links.name.isin(new_dims.get('link_name',[]))|model.links.id.isin(catchments.link)]
  nodes = model.nodes[model.nodes.name.isin(new_dims.get('node_name',[]))|model.nodes.id.isin(links.from_node)|model.nodes.id.isin(links.to_node)]
  catchments.to_file(dest_fn.replace('.h5', '.catchments.json'), driver='GeoJSON')
  links.to_file(dest_fn.replace('.h5', '.links.json'), driver='GeoJSON')
  nodes.to_file(dest_fn.replace('.h5', '.nodes.json'), driver='GeoJSON')

  new_model.close()
