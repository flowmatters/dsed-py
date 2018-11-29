'''
Running Dynamic Sednet simulations using OpenWater
'''
from openwater import OWTemplate, OWLink
from openwater.template import TAG_MODEL
import openwater.nodes as n
from collections import defaultdict

FINE_SEDIMENT = 'Sediment - Fine'
COARSE_SEDIMENT = 'Sediment - Coarse'

SEDIMENT_CLASSES = [FINE_SEDIMENT,COARSE_SEDIMENT]
STANDARD_NUTRIENTS = ['TN','TP']

STANDARD_CONSTITUENTS = SEDIMENT_CLASSES + STANDARD_NUTRIENTS

# def default_generation_model(constituent,landuse):
#     if constituent=='TSS':
#         return n.USLEFineSedimentGeneration
#     return n.EmcDwc

# def build_catchment_template(constituents,hrus,landuses,generation_model=default_generation_model):
#     template = OWTemplate()
#     routing_node = template.add_node(n.Muskingum,process='FlowRouting')
#     for con in constituents:
#         # transport_node = 'Transport-%s'%(con)
#         transport_node = template.add_node(n.LumpedConstituentRouting,process='ConstituentRouting',constituent=con)
#         template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))

#     for hru in hrus:
#         runoff_node = template.add_node(n.Simhyd,process='RR',hru=hru)
#         runoff_scale_node = template.add_node(n.DepthToRate,process='ArealScale',hru=hru,component='Runoff')
#         quickflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',hru=hru,component='Quickflow')
#         baseflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',hru=hru,component='Baseflow')

#         template.add_link(OWLink(runoff_node,'runoff',runoff_scale_node,'input'))
#         template.add_link(OWLink(runoff_node,'quickflow',quickflow_scale_node,'input'))
#         template.add_link(OWLink(runoff_node,'baseflow',baseflow_scale_node,'input'))

#         template.add_link(OWLink(runoff_scale_node,'outflow',routing_node,'lateral'))

#         for con in constituents:
#             # transport_node = 'Transport-%s'%(con)
#             transport_node = template.add_node(n.LumpedConstituentRouting,process='ConstituentRouting',constituent=con) #!!!ERROR
#             template.add_link(OWLink(runoff_scale_node,'outflow',transport_node,'inflow'))
#             for lu in landuses[hru]:
#                 #gen_node = 'Generation-%s-%s'%(con,lu)
#                 gen_node = template.add_node(generation_model(con,lu),process='ConstituentGeneration',constituent=con,lu=lu)
#                 template.add_link(OWLink(quickflow_scale_node,'outflow',gen_node,'quickflow'))
#                 template.add_link(OWLink(baseflow_scale_node,'outflow',gen_node,'baseflow'))
#                 template.add_link(OWLink(gen_node,'totalLoad',transport_node,'lateralLoad'))

#     return template

# def link_catchments(graph,from_cat,to_cat,constituents):
#     linkages = [('%d-FlowRouting (Muskingum)','outflow','inflow')] + \
#                [('%%d-ConstituentRouting-%s (LumpedConstituentRouting)'%c,'outflowLoad','inflowLoad') for c in constituents]
#     for (lt,src,dest) in linkages:
#         dest_node = lt%from_cat
#         src_node = lt%to_cat#'%d/%s'%(to_cat,lt)
#         graph.add_edge(src_node,dest_node,src=[src],dest=[dest])

# def generation_models(constituent,cgu):
#     if constituent in STANDARD_NUTRIENTS:
#         return n.EmcDwc


#     # if pesticide
#     return n.EmcDwc

class Reach(object):
    pass

class HydrologicalResponseUnit(object):
    pass

class DynamicSednetCGU(object):
    def generation_model(self,constituent,catchment):
        return catchment.model_for(catchment.cg,constituent)

    def get_template(self,catchment,**kwargs):
        tag_values = list(kwargs.values())
        template = OWTemplate('cgu:%s'%kwargs.get('cgu','?'))

        runoff_scale_node = template.add_node(n.DepthToRate,process='ArealScale',component='Runoff',**kwargs)
        quickflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',component='Quickflow',**kwargs)
        baseflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',component='Baseflow',**kwargs)

        template.define_input(runoff_scale_node,'input','runoff')
        template.define_input(quickflow_scale_node,'input','quickflow')
        template.define_input(baseflow_scale_node,'input','baseflow')
        template.define_output(runoff_scale_node,'outflow','lateral')
        # This should be able to be done automatically... any input not defined

        sed_gen = template.add_node(n.USLEFineSedimentGeneration,process="SedimentGeneration")
        template.add_link(OWLink(quickflow_scale_node,'outflow',sed_gen,'quickflow'))
        template.add_link(OWLink(baseflow_scale_node,'outflow',sed_gen,'baseflow'))
        template.define_output(sed_gen,'totalLoadFine','generatedLoad',constituent=FINE_SEDIMENT)
        template.define_output(sed_gen,'totalLoadCoarse','generatedLoad',constituent=COARSE_SEDIMENT)

        for con in catchment.constituents:
            model = self.generation_model(con,catchment)
            if model is None:
                print('No regular constituent generation model for %s'%con)
                continue

            gen_node = template.add_node(model,process='ConstituentGeneration',constituent=con,**kwargs)
            template.add_link(OWLink(quickflow_scale_node,'outflow',gen_node,'quickflow'))
            template.add_link(OWLink(baseflow_scale_node,'outflow',gen_node,'baseflow'))
            template.define_output(gen_node,'totalLoad','generatedLoad')

        return template

class DynamicSednetAgCGU(DynamicSednetCGU):
    pass
    # def generation_model(self,constituent,catchment):
    #     if constituent == FINE_SEDIMENT:
    #         return n.USLEFineSedimentGeneration
    #     return super(DynamicSednetAgCGU,self).generation_model(constituent)

class NilCGU(DynamicSednetCGU):
    def generation_model(self,*args):
        return None

def cgu_factory(cgu):
    if cgu=='Water/lakes':
        return NilCGU()
    if cgu in ['Dryland', 'Irrigation', 'Horticulture', 'Irrigated Grazing']:
        return DynamicSednetAgCGU()
    return DynamicSednetCGU()

class DynamicSednetCatchment(object):
    def __init__(self,disolved_nutrients=['DisN','DisP'],particulate_nutrients=['PartN','PartP'],pesticides=['Pesticide1']):
        self.hrus = ['HRU']
        self.cgus = ['CGU']
        self.cgu_hrus = {'CGU':'HRU'}
        self.constituents = SEDIMENT_CLASSES + disolved_nutrients + particulate_nutrients + pesticides

        self.rr = n.Sacramento
        self.cg = defaultdict(lambda:n.EmcDwc,{
            FINE_SEDIMENT:None,
            COARSE_SEDIMENT:None
        })

        self.routing = n.Muskingum
        self.transport = defaultdict(lambda:n.LumpedConstituentRouting,{
            FINE_SEDIMENT:n.InstreamFineSediment,
            COARSE_SEDIMENT:n.InstreamCoarseSediment
        })

        for dn in disolved_nutrients:
            self.cg[dn] = n.SednetDissolvedNutrientGeneration
            self.transport[dn] = n.InstreamDissolvedNutrientDecay
        for pn in particulate_nutrients:
            self.cg[pn] = n.SednetParticulateNutrientGeneration
            # self.transport[pn] = n.InstreamParticulateNutrient

    def model_for(self,provider,*args):
        if hasattr(provider,'__call__'):
            return provider(*args)
        if hasattr(provider,'__getitem__'):
            return provider[args[0]]
        return provider

    def reach_template(self,**kwargs):
        tag_values = list(kwargs.values())
        reach_template = OWTemplate('reach')

        routing_node = reach_template.add_node(self.routing,process='FlowRouting',**kwargs)
        reach_template.define_input(routing_node,'lateral')
        reach_template.define_output(routing_node,'outflow')
        transport = {}

        bank_erosion = reach_template.add_node(n.BankErosion,process='BankErosion',**kwargs)
        reach_template.add_link(OWLink(routing_node,'outflow',bank_erosion,'downstreamFlowVolume'))

        dis_nut_models = []
        fine_sed_model = None
        for con in self.constituents:
            model_type = self.model_for(self.transport,con,*tag_values)

            transport_node = reach_template.add_node(model_type,process='ConstituentRouting',constituent=con,**kwargs)

            if model_type == n.InstreamFineSediment:
                reach_template.define_input(transport_node,'incomingMass','generatedLoad')
                reach_template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))

                reach_template.add_link(OWLink(bank_erosion,'bankErosionFine',transport_node,'lateralLoad'))
                reach_template.define_output(transport_node,'loadDownstream','outflowLoad')
                fine_sed_model = transport_node

            elif model_type == n.InstreamCoarseSediment:
                reach_template.define_input(transport_node,'incomingMass','generatedLoad')

                reach_template.add_link(OWLink(bank_erosion,'bankErosionCoarse',transport_node,'lateralLoad'))
                reach_template.define_output(transport_node,'loadDownstream','outflowLoad')

            elif model_type == n.InstreamDissolvedNutrientDecay:
                dis_nut_models.append(transport_node)
                reach_template.define_input(transport_node,'incomingMassLateral','generatedLoad')
                reach_template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))

                reach_template.define_output(transport_node,'loadDownstream','outflowLoad')

            else:
                reach_template.define_input(transport_node,'lateralLoad','generatedLoad')
                reach_template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))
                reach_template.define_output(transport_node,'outflowLoad')

        if fine_sed_model is not None:
            for dnm in dis_nut_models:
                reach_template.add_link(OWLink(fine_sed_model,'floodplainDepositionFraction',dnm,'floodplainDepositionFraction'))

        return reach_template

    def get_template(self,**kwargs):
        tag_values = list(kwargs.values())
        template = OWTemplate('catchment')

        hrus={}
        for hru in self.hrus:
            runoff_template = OWTemplate('runoff:%s'%hru)

            runoff_node = runoff_template.add_node(self.model_for(self.rr,hru,*tag_values),process='RR',hru=hru,**kwargs)
            runoff_template.define_output(runoff_node,'runoff')
            runoff_template.define_output(runoff_node,'quickflow')
            runoff_template.define_output(runoff_node,'baseflow')
            hru_template = OWTemplate('hru:%s'%hru)
            hru_template.nest(runoff_template)
            hrus[hru] = hru_template
            template.nest(hru_template)

        for cgu in self.cgus:
            hru = self.cgu_hrus[cgu]
            cgu_builder = cgu_factory(cgu)
            cgu_template = cgu_builder.get_template(self,cgu=cgu)
            hrus[hru].nest(cgu_template)

        template.nest(self.reach_template(**kwargs))

        return template

    def link_catchments(self,graph,upstream,downstream):
        STANDARD_LINKS = defaultdict(lambda:[None,None],{
            n.InstreamFineSediment.name: ('incomingMass','loadDownstream'),
            n.InstreamCoarseSediment.name: ('incomingMass','loadDownstream'),
        })

        linkages = [('%s-FlowRouting','outflow','inflow')] + \
                   [('%%s-ConstituentRouting-%s'%c,'outflowLoad','inflowLoad') for c in self.constituents]
        for (lt,src,dest) in linkages:

            src_node = lt%(str(upstream))
            dest_node = lt%(str(downstream))#'%d/%s'%(to_cat,lt)
            src_node = [n for n in graph.nodes if n.startswith(src_node)][0]
            dest_node = [n for n in graph.nodes if n.startswith(dest_node)][0]

            src_model = graph.nodes[src_node][TAG_MODEL]
            dest_model = graph.nodes[dest_node][TAG_MODEL]
            src = STANDARD_LINKS[src_model][0] or src
            dest = STANDARD_LINKS[dest_model][1] or dest
            print(src_node,src,dest_node,dest)
            graph.add_edge(src_node,dest_node,src=[src],dest=[dest])
