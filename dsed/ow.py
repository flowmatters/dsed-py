'''
Running Dynamic Sednet simulations using OpenWater
'''
from openwater import OWTemplate, OWLink
from openwater.template import TAG_PROCESS
import openwater.nodes as n

def default_generation_model(constituent,landuse):
    if constituent=='TSS':
        return n.USLEFineSedimentGeneration
    return n.EmcDwc

def build_catchment_template(constituents,hrus,landuses,generation_model=default_generation_model):
    template = OWTemplate()
    routing_node = template.add_node(n.Muskingum,process='FlowRouting')
    for con in constituents:
        # transport_node = 'Transport-%s'%(con)
        transport_node = template.add_node(n.LumpedConstituentRouting,process='ConstituentRouting',constituent=con)
        template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))

    for hru in hrus:
        runoff_node = template.add_node(n.Simhyd,process='RR',hru=hru)
        runoff_scale_node = template.add_node(n.DepthToRate,process='ArealScale',hru=hru,component='Runoff')
        quickflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',hru=hru,component='Quickflow')
        baseflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',hru=hru,component='Baseflow')

        template.add_link(OWLink(runoff_node,'runoff',runoff_scale_node,'input'))      
        template.add_link(OWLink(runoff_node,'quickflow',quickflow_scale_node,'input'))      
        template.add_link(OWLink(runoff_node,'baseflow',baseflow_scale_node,'input'))      

        template.add_link(OWLink(runoff_scale_node,'outflow',routing_node,'lateral'))

        for con in constituents:
            # transport_node = 'Transport-%s'%(con)
            transport_node = template.add_node(n.LumpedConstituentRouting,process='ConstituentRouting',constituent=con)
            template.add_link(OWLink(runoff_scale_node,'outflow',transport_node,'inflow'))
            for lu in landuses[hru]:
                #gen_node = 'Generation-%s-%s'%(con,lu)
                gen_node = template.add_node(generation_model(con,lu),process='ConstituentGeneration',constituent=con,lu=lu)
                template.add_link(OWLink(quickflow_scale_node,'outflow',gen_node,'quickflow'))
                template.add_link(OWLink(baseflow_scale_node,'outflow',gen_node,'baseflow'))
                template.add_link(OWLink(gen_node,'totalLoad',transport_node,'lateralLoad'))

    return template

def link_catchments(graph,from_cat,to_cat,constituents):
    linkages = [('%d-FlowRouting (Muskingum)','outflow','inflow')] + \
               [('%%d-ConstituentRouting-%s (LumpedConstituentRouting)'%c,'outflowLoad','inflowLoad') for c in constituents]
    for (lt,src,dest) in linkages:
        dest_node = lt%from_cat
        src_node = lt%to_cat#'%d/%s'%(to_cat,lt)
        graph.add_edge(src_node,dest_node,src=[src],dest=[dest])
