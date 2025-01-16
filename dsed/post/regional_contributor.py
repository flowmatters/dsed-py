import pandas as pd
import numpy as np
import os

def apply_catchment_names_for_upstream_nodes(results,network):
    results = results.copy()
    catchment_names = set(results[results.ModelElementType=='Catchment'].ModelElement)
    _catchment_link_map = {n:network.by_name(n)['properties']['link'] for n in catchment_names}
    catchment_upstream_nodes = {network.by_id(network.by_id(l)['properties']['from_node'])['properties']['name']:n for n,l in _catchment_link_map.items()}
    results['ModelElement'] = results['ModelElement'].replace(catchment_upstream_nodes)
    return results

def merge_feature_lookup(dest,source):
    for r, features in source.items():
        dest[r] = dest.get(r,set()).union(features)

def map_reporting_regions(network,node,lookup,existing_regions=None):
    result = {}
    if existing_regions is None:
        existing_regions = set()

    reporting_region = None
    upstream_links = network.upstream_links(node)
    for l in upstream_links:
        l_id = l['properties']['id']
        catchments = network['features'].find_by_link(l_id)
        assert len(catchments)==1,f'Expected 1 catchment, got {len(catchments)}'
        if reporting_region is not None:
            assert reporting_region == lookup[catchments[0]['properties']['name']]
        reporting_region = lookup[catchments[0]['properties']['name']]
        existing_regions = existing_regions.union({reporting_region})
        next_node = network.by_id(l['properties']['from_node'])
        feature_names = set([f['properties']['name'] for f in [catchments[0],l,node,next_node]])
        for r in existing_regions:
            result[r] = result.get(r,set()).union(feature_names)

        nested = map_reporting_regions(network,next_node,lookup,existing_regions)
        merge_feature_lookup(result,nested)
    return result

def subset_network(network,feature_names):
    return network.subset(lambda f:f['properties']['name'] in feature_names)

def accumulate_rsdr(result,link,network,rsdrs,downstream_rsdr):
    catchments = network['features'].find_by_link(link['properties']['id'])
    assert len(catchments)==1
    catchment_name = catchments[0]['properties']['name']
    local_rsdrs = rsdrs[rsdrs.ModelElement==catchment_name]
    assert len(local_rsdrs)==1,f'Expected 1 RSDR matching {catchment_name}. Had {len(local_rsdrs)}'

    link_rsdr = local_rsdrs.iloc[0].RSDR * downstream_rsdr
    result[catchment_name] = link_rsdr
    from_node = network.by_id(link['properties']['from_node'])

    for l in network.upstream_links(from_node):
        accumulate_rsdr(result,l,network,rsdrs,link_rsdr)

def build_rsdr_lookup(network,rsdrs):
    result = {}
    outlets = network.outlet_nodes()

    for o in outlets:
        links = network.upstream_links(o)
        for l in links:
            accumulate_rsdr(result,l,network,rsdrs,1.0)
    return result


def regional_contributor(network=None,reporting_regions=None,results_directory=None,v=None,raw=None,rsdr=None,climate=None,fu_areas=None):
    '''

    Parameters:
    - network: Source/Veneer Node link network. If None, provide v (Veneer client)
    - reporting_regions: GeoDataFrame of reporting regions using subcatchment polygons or dictionary lookup from subcatchment name to reporting region name

    '''

    if raw is None:
        raw = pd.read_csv(os.path.join(results_directory,'RawResults.csv'))
    if rsdr is None:
        rsdr = pd.read_csv(os.path.join(results_directory,'RSDRTable.csv'))
    if climate is None:
        climate = pd.read_csv(os.path.join(results_directory,'climateTable.csv'))
    if fu_areas is None:
        fu_areas = pd.read_csv(os.path.join(results_directory,'fuAreasTable.csv'))

    if network is None:
        network = v.network()

    raw = apply_catchment_names_for_upstream_nodes(raw,network)
    raw['BudgetElement'] = raw['BudgetElement'].replace({'Node Injected Inflow':'Node Injected Mass','Rainfall':'Storage Rainfall'})

    catchment_supply_rows = raw[(raw.ModelElementType=='Catchment')&(raw.Process=='Supply')]

    system_supply_types = {'Node Injected Mass', 'Node Initial Load', 'Link Initial Load', 'Storage Rainfall'}
    system_supply = raw[raw.BudgetElement.isin(system_supply_types)]
    system_supply['FU']='System Supply'

    climate['Element'] = climate['Element'].replace({'Runoff (QuickFlow)':'Quickflow','Baseflow':'Slowflow'})
    runoff_supply_types = {'Quickflow', 'Slowflow'}
    runoff_rows = climate[climate.Element.isin(runoff_supply_types)]
    runoff_rows = runoff_rows.rename(columns={'Catchment':'ModelElement','Element':'BudgetElement'})
    runoff_rows['Constituent']='Flow'

    constituents = set(raw['Constituent'])
    all_supply = pd.concat([catchment_supply_rows,system_supply,runoff_rows])

    if hasattr(reporting_regions,'geometry'):
        repreg_lookup = dict(zip(reporting_regions['Catchmt'],reporting_regions['RepReg']))
    else:
        repreg_lookup = reporting_regions

    outlets = network.outlet_nodes()
    reporting_region_features={}

    for n in outlets:
        merge_feature_lookup(reporting_region_features,map_reporting_regions(network,n,repreg_lookup))

    regional_rsdrs = {} # Lookup from region name to series/dictionary from link name to rsdr
    for region,features in reporting_region_features.items():
        network_subset = subset_network(network,features)
        print(f'Processing {region}')
        for c in constituents:
            rsdr_subset = rsdr[rsdr.Constituent==c]
            regional_rsdrs[(region,c)] = build_rsdr_lookup(network_subset,rsdr_subset)

    COLUMN_ORDER=['Rep_Region', 'ModelElement', 'Constituent', 'FU', 'AreaM2', 'Process',
           'LoadToStream (kg)', 'LoadToRegExport (kg)', 'RSDR', 'Num_Days']
    COLUMNS=set(COLUMN_ORDER)

    region_results = []
    print('Running for %d regions/constituents'%len(regional_rsdrs))
    for (reg, constituent), rsdrs in regional_rsdrs.items():
        joined = pd.merge(all_supply[all_supply.Constituent==constituent],pd.Series(rsdrs,name='RSDR'),how='inner',left_on='ModelElement',right_index=True)
        joined = pd.merge(joined,fu_areas,how='left',left_on=['ModelElement','FU'],right_on=['Catchment','FU'])
        joined['Rep_Region'] = reg
        joined = joined.drop(columns=['ModelElementType','Process','Catchment'])
        joined.rename(columns={'Total_Load_in_Kg':'LoadToStream (kg)','BudgetElement':'Process','Area':'AreaM2'},inplace=True)

        flow_kg = joined['Depth_m'] * joined['AreaM2'] * 1e3
        joined['LoadToStream (kg)'] = joined['LoadToStream (kg)'].fillna(flow_kg)

        joined['LoadToRegExport (kg)'] = joined['LoadToStream (kg)'] * joined['RSDR']
        joined['AreaM2'] = joined['AreaM2'].fillna(0.0)

        MISSING = COLUMNS - set(joined.columns)
        for c in MISSING:
            joined[c] = np.nan
        joined = joined[COLUMN_ORDER]
        region_results.append(joined)

    result = pd.concat(region_results)
    return result

