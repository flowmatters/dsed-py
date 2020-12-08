'''
Running Dynamic Sednet simulations using OpenWater
'''
from openwater import OWTemplate, OWLink
from openwater.template import TAG_MODEL
import openwater.nodes as n
from collections import defaultdict

LANDSCAPE_CONSTITUENT_SOURCES=['Hillslope','Gully']

FINE_SEDIMENT = 'Sediment - Fine'
COARSE_SEDIMENT = 'Sediment - Coarse'
CGUS_TS_N_DIN = ['Sugarcane','Bananas']

SEDIMENT_CLASSES = [FINE_SEDIMENT,COARSE_SEDIMENT]
STANDARD_NUTRIENTS = ['TN','TP']

STANDARD_CONSTITUENTS = SEDIMENT_CLASSES + STANDARD_NUTRIENTS
QUICKFLOW_INPUTS = ['quickflow','flow']
BASEFLOW_INPUTS = ['baseflow','slowflow']

NIL_MODELS = {
    'Dynamic_SedNet.Models.SedNet_Blank_Constituent_Generation_Model',
    'RiverSystem.Catchments.Models.ContaminantGenerationModels.NilConstituent'
}

MODEL_NAME_TRANSLATIONS = {

}
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
    def __init__(self,cropping_cgu=True,sediment_fallback_model=False,gully_cgu=False,hillslope_cgu=False,ts_load_with_dwc=None):
        self.cropping_cgu = cropping_cgu
        # self.erosion_processes = erosion_processes
        self.gully_cgu = gully_cgu
        self.hillslope_cgu = hillslope_cgu
        self.sediment_fallback_model = sediment_fallback_model
        self.ts_load_with_dwc = ts_load_with_dwc
        assert (not bool(gully_cgu)) or (not bool(sediment_fallback_model))

    def generation_model(self,constituent,catchment_template,**kwargs):
        return catchment_template.model_for(catchment_template.cg,constituent,**kwargs)

    def get_template(self,catchment_template,**kwargs):
        tag_values = list(kwargs.values())
        cgu = kwargs.get('cgu','?')
        template = OWTemplate('cgu:%s'%cgu)

        runoff_scale_node = None
        quickflow_scale_node = None
        baseflow_scale_node = None
        if catchment_template.rr is not None:
            runoff_scale_node = template.add_node(n.DepthToRate,process='ArealScale',component='Runoff',**kwargs)
            quickflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',component='Quickflow',**kwargs)
            baseflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',component='Baseflow',**kwargs)

        def link_runoff(dest_node,qf_input,bf_input):
            if quickflow_scale_node is None:
                return
            if qf_input is not None:
                template.add_link(OWLink(quickflow_scale_node,'outflow',dest_node,qf_input))

            if bf_input is not None:
                template.add_link(OWLink(baseflow_scale_node,'outflow',dest_node,bf_input))

        def add_emc_dwc(con):
            dwc_node = template.add_node(n.EmcDwc,process='ConstituentDryWeatherGeneration',constituent=con,**kwargs)
            link_runoff(dwc_node,'quickflow','baseflow')
            return dwc_node

        if runoff_scale_node is not None:
            template.define_input(runoff_scale_node,'input','runoff')
            template.define_input(quickflow_scale_node,'input','quickflow')
            template.define_input(baseflow_scale_node,'input','baseflow')
            template.define_output(runoff_scale_node,'outflow','lateral')
        # This should be able to be done automatically... any input not defined

        hillslope_fine_sed_gen = None
        hillslope_coarse_sed_gen = None
        hillslope_fine_sed_gen_flux = None
        hillslope_coarse_sed_gen_flux = None

        # fine_ts_scale = None
        # coarse_ts_scale = None
        gully_gen = None

        if self.hillslope_cgu:
            # Hillslope
            sed_gen = template.add_node(n.USLEFineSedimentGeneration,process="HillslopeGeneration",**kwargs)
            link_runoff(sed_gen,'quickflow','baseflow')

            hillslope_fine_sed_gen = sed_gen
            hillslope_coarse_sed_gen = sed_gen
            hillslope_fine_sed_gen_flux = 'generatedLoadFine'
            hillslope_coarse_sed_gen_flux = 'generatedLoadCoarse'

        if self.gully_cgu:
            # Gully
            gully_gen = template.add_node(n.DynamicSednetGullyAlt,process="GullyGeneration",**kwargs)
            link_runoff(gully_gen,'quickflow',None)

            fine_sum = template.add_node(n.Sum,process='ConstituentGeneration',constituent=FINE_SEDIMENT,**kwargs)
            coarse_sum = template.add_node(n.Sum,process='ConstituentGeneration',constituent=COARSE_SEDIMENT,**kwargs)

            template.add_link(OWLink(gully_gen,'fineLoad',fine_sum,'i2'))
            template.add_link(OWLink(gully_gen,'coarseLoad',coarse_sum,'i2'))

            if self.hillslope_cgu:
                template.add_link(OWLink(sed_gen,'totalLoad',fine_sum,'i1')) # was quickLoadFine
                template.add_link(OWLink(sed_gen,'quickLoadCoarse',coarse_sum,'i1'))
            else:
                fine_dwc_node = add_emc_dwc(FINE_SEDIMENT)
                template.add_link(OWLink(fine_dwc_node,'totalLoad',fine_sum,'i1'))

                coarse_dwc_node = add_emc_dwc(COARSE_SEDIMENT)
                template.add_link(OWLink(coarse_dwc_node,'totalLoad',coarse_sum,'i1'))

                if self.cropping_cgu:
                    ts_node = template.add_node(n.PassLoadIfFlow,process='ConstituentOtherGeneration',constituent=FINE_SEDIMENT,**kwargs)
                    link_runoff(ts_node,'flow',None)

                    ts_split_node = template.add_node(n.FixedPartition,process='FineCoarseSplit',**kwargs)
                    template.add_link(OWLink(ts_node,'outputLoad',ts_split_node,'input'))

                    fine_ts_scale = template.add_node(n.ApplyScalingFactor,process='ConstituentScaling',constituent=FINE_SEDIMENT,**kwargs)
                    template.add_link(OWLink(ts_split_node,'output1',fine_ts_scale,'input')) # fraction

                    fine_ts_sdr = template.add_node(n.DeliveryRatio,process='SDR',constituent=FINE_SEDIMENT,**kwargs)
                    template.add_link(OWLink(fine_ts_scale,'output',fine_ts_sdr,'input')) # fraction

                    coarse_ts_sdr = template.add_node(n.DeliveryRatio,process='SDR',constituent=COARSE_SEDIMENT,**kwargs)
                    template.add_link(OWLink(ts_split_node,'output2',coarse_ts_sdr,'input')) # 1-fraction

                    template.add_link(OWLink(fine_ts_sdr,'output',fine_sum,'i1'))
                    template.add_link(OWLink(coarse_ts_sdr,'output',coarse_sum,'i1'))

                    #TODO Will this always be the right thing to link? Should it ideally be i1 of the sum node going into other constituent models?
                    hillslope_fine_sed_gen = fine_ts_scale
                    hillslope_fine_sed_gen_flux = 'output'

                    hillslope_coarse_sed_gen = ts_split_node
                    hillslope_coarse_sed_gen_flux = 'output2'
                    # HACK - Just seeing if this is what we need in order to get
                    # the pre-SDR sediment loads?
                    # But it includes the load conversion factor on the fine...
                    # So *probably not*

                    # hillslope_fine_sed_gen = fine_ts_scale
                    # hillslope_fine_sed_gen_flux = 'output'

                    # hillslope_coarse_sed_gen = coarse_ts_scale
                    # hillslope_coarse_sed_gen_flux = 'output'
                else:
                    # TODO: HACK - Check that quickLoad should apply whenever we have an EMC/DWC model for sediment
                    hillslope_fine_sed_gen = fine_dwc_node
                    hillslope_fine_sed_gen_flux = 'quickLoad'

                    hillslope_coarse_sed_gen = coarse_dwc_node
                    hillslope_coarse_sed_gen_flux = 'quickLoad'

            template.define_output(fine_sum,'out','generatedLoad',constituent=FINE_SEDIMENT)
            template.define_output(coarse_sum,'out','generatedLoad',constituent=COARSE_SEDIMENT)

        if self.cropping_cgu:
            for con in catchment_template.pesticides:
                dwc_node = add_emc_dwc(con)

                ts_node = template.add_node(n.PassLoadIfFlow,process='ConstituentOtherGeneration',constituent=con,**kwargs)
                link_runoff(ts_node,'flow',None)

                sum_node = template.add_node(n.Sum,process='ConstituentGeneration',constituent=con,**kwargs)
                template.add_link(OWLink(dwc_node,'totalLoad',sum_node,'i1'))
                template.add_link(OWLink(ts_node,'outputLoad',sum_node,'i2'))

                template.define_output(sum_node,'out','generatedLoad')

        for con in catchment_template.constituents:
            if not self.sediment_fallback_model and (con in [FINE_SEDIMENT,COARSE_SEDIMENT]):
                continue

            if con in catchment_template.pesticides:
                continue

            ts_cane_din = (cgu in CGUS_TS_N_DIN) and (con=='N_DIN')
            ts_crop_part_p = (con == 'P_Particulate') and self.cropping_cgu and not cgu=='Sugarcane'
            ts_load_with_dwc = self.ts_load_with_dwc and \
                               (con in self.ts_load_with_dwc['constituents']) and \
                               (cgu in self.ts_load_with_dwc['cgus'])

            if ts_cane_din or ts_crop_part_p or ts_load_with_dwc:
                ts_node = template.add_node(n.PassLoadIfFlow,process='ConstituentOtherGeneration',constituent=con,**kwargs)
                link_runoff(ts_node,'flow',None)

                ts_scale_node = template.add_node(n.ApplyScalingFactor,process='ConstituentScaling',constituent=con,**kwargs)
                template.add_link(OWLink(ts_node,'outputLoad',ts_scale_node,'input'))

                dwc_node = add_emc_dwc(con)

                sum_node = template.add_node(n.Sum,process='ConstituentGeneration',constituent=con,**kwargs)
                template.add_link(OWLink(ts_scale_node,'output',sum_node,'i1'))
                template.add_link(OWLink(dwc_node,'totalLoad',sum_node,'i2'))

                if ts_cane_din:
                    leached_ts_node = template.add_node(n.PassLoadIfFlow,process='ConstituentOtherGeneration',constituent='NLeached',**kwargs)
                    link_runoff(leached_ts_node,None,'flow')

                    leached_ts_scale_node = template.add_node(n.ApplyScalingFactor,process='ConstituentScaling',constituent='NLeached',**kwargs)
                    template.add_link(OWLink(leached_ts_node,'outputLoad',leached_ts_scale_node,'input'))

                    template.add_link(OWLink(leached_ts_scale_node,'output',sum_node,'i2'))

                template.define_output(sum_node,'out','generatedLoad')
                continue

            model = self.generation_model(con,catchment_template,**kwargs)
            if model is None:
                print('No regular constituent generation model for %s'%con)
                continue

            gen_node = template.add_node(model,process='ConstituentGeneration',constituent=con,**kwargs)
            if quickflow_scale_node is not None:
                template.add_conditional_link(quickflow_scale_node,'outflow',gen_node,QUICKFLOW_INPUTS,model)
                template.add_conditional_link(baseflow_scale_node, 'outflow',gen_node,BASEFLOW_INPUTS,model)

            if model.name == 'SednetParticulateNutrientGeneration':
                template.add_link(OWLink(gully_gen,'generatedFine',gen_node,'fineSedModelFineGullyGeneratedKg'))
                template.add_link(OWLink(gully_gen,'generatedCoarse',gen_node,'fineSedModelCoarseGullyGeneratedKg'))

                template.add_link(OWLink(hillslope_fine_sed_gen,hillslope_fine_sed_gen_flux,
                                         gen_node,'fineSedModelFineSheetGeneratedKg'))
                template.add_link(OWLink(hillslope_coarse_sed_gen,hillslope_coarse_sed_gen_flux,
                                         gen_node,'fineSedModelCoarseSheetGeneratedKg'))

            template.define_output(gen_node,main_output_flux(model),'generatedLoad')

        return template

class DynamicSednetAgCGU(DynamicSednetCGU):
    pass
    # def generation_model(self,constituent,catchment):
    #     if constituent == FINE_SEDIMENT:
    #         return n.USLEFineSedimentGeneration
    #     return super(DynamicSednetAgCGU,self).generation_model(constituent)

class NilCGU(DynamicSednetCGU):
    def generation_model(self,*args,**kwargs):
        return None


class DynamicSednetCatchment(object):
    def __init__(self,
                 dissolved_nutrients=['DisN','DisP'],
                 particulate_nutrients=['PartN','PartP'],
                 pesticides=['Pesticide1'],
                 particulate_nutrient_cgus=None,
                 ts_load_with_dwc=None):
        self.hrus = ['HRU']
        self.cgus = ['CGU']
        self.cgu_hrus = {'CGU':'HRU'}
        self.constituents = SEDIMENT_CLASSES + dissolved_nutrients + particulate_nutrients + pesticides
        self.particulate_nutrients = particulate_nutrients
        self.particulate_nutrient_cgus = particulate_nutrient_cgus
        self.dissolved_nutrients = dissolved_nutrients
        self.pesticides = pesticides
        self.pesticide_cgus = None
        self.timeseries_sediment_cgus = None
        self.hillslope_cgus = None
        self.gully_cgus = None
        self.sediment_fallback_cgu = None
        self.ts_load_with_dwc = ts_load_with_dwc

        self.rr = n.Sacramento
        self.cg = defaultdict(lambda:n.EmcDwc,{})
        # {
        #     FINE_SEDIMENT:None,
        #     COARSE_SEDIMENT:None
        # })

        self.routing = n.Muskingum
        self.transport = defaultdict(lambda:n.LumpedConstituentRouting,{
            FINE_SEDIMENT:n.InstreamFineSediment,
            COARSE_SEDIMENT:n.InstreamCoarseSediment
        })

        self._g = None
        self._g_lookup = {}

        def get_model_dissolved_nutrient(*args,**kwargs):
            cgu = kwargs['cgu']

            if cgu in self.pesticide_cgus:
                constituent = args[0]
                if cgu=='Sugarcane':
                    if constituent=='N_DIN':
                        return None
                    elif constituent=='N_DON':
                        return n.EmcDwc
                    elif constituent.startswith('P'):
                        return n.EmcDwc
                if constituent.startswith('P'):
                    return n.PassLoadIfFlow
            # if cgu is a cropping FU:
            #   look at constituent
            #   constituent = args[0]

            if cgu in ['Water']: # 'Conservation','Horticulture','Other','Urban','Forestry'
                return n.EmcDwc

            # print(args)
            # print(kwargs)
            return n.SednetDissolvedNutrientGeneration

        def get_model_particulate_nutrient(*args,**kwargs):
            cgu = kwargs['cgu']

            if self.particulate_nutrient_cgus is None:
                if cgu in ['Water','Conservation','Horticulture','Other','Urban','Forestry']:
                    return n.EmcDwc
                # if cropping (but not sugarcane) and constituent == P_Particulate
                # Timeseries model...

                return n.SednetParticulateNutrientGeneration

            if cgu in self.particulate_nutrient_cgus:
                return n.SednetParticulateNutrientGeneration
            return n.EmcDwc

        for dn in dissolved_nutrients:
            self.cg[dn] = get_model_dissolved_nutrient
            self.transport[dn] = n.InstreamDissolvedNutrientDecay
        for pn in particulate_nutrients:
            self.cg[pn] = get_model_particulate_nutrient
            # self.transport[pn] = n.InstreamParticulateNutrient

    def model_for(self,provider,*args,**kwargs):
        if hasattr(provider,'__call__'):
            return self.model_for(provider(*args,**kwargs),*args,**kwargs)
        if hasattr(provider,'__getitem__'):
            return self.model_for(provider[args[0]],*args,**kwargs)
        return provider

    def reach_template(self,**kwargs):
        tag_values = list(kwargs.values())
        reach_template = OWTemplate('reach')

        routing_node = None
        if self.routing is not None:
            lag_node = reach_template.add_node(n.Lag,process='FlowLag',constituent='_flow',**kwargs)
            reach_template.define_input(lag_node,'inflow','lateral')

            routing_node = reach_template.add_node(self.routing,process='FlowRouting',**kwargs)
            reach_template.add_link(OWLink(lag_node,'outflow',routing_node,'lateral'))
            reach_template.define_output(routing_node,'outflow')

        bank_erosion = reach_template.add_node(n.BankErosion,process='BankErosion',**kwargs)
        if routing_node is not None:
            reach_template.add_link(OWLink(routing_node,'storage',bank_erosion,'totalVolume'))
            reach_template.add_link(OWLink(routing_node,'outflow',bank_erosion,'downstreamFlowVolume'))

        dis_nut_models = []
        fine_sed_model = None
        for con in self.constituents:
            model_type = self.model_for(self.transport,con,*tag_values)
            constituent_lag_node = reach_template.add_node(n.Lag,process='FlowLag',constituent=con,**kwargs)
            reach_template.define_input(constituent_lag_node,'inflow','generatedLoad')

            transport_node = reach_template.add_node(model_type,process='ConstituentRouting',constituent=con,**kwargs)

            if model_type == n.InstreamFineSediment:
                # reach_template.define_input(transport_node,'incomingMass','generatedLoad')
                reach_template.add_link(OWLink(constituent_lag_node,'outflow',transport_node,'incomingMass'))
                if self.routing is not None:
                    reach_template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))
                    reach_template.add_link(OWLink(routing_node,'storage',transport_node,'reachVolume'))

                reach_template.add_link(OWLink(bank_erosion,'bankErosionFine',transport_node,'incomingMass'))
                reach_template.define_output(transport_node,'loadDownstream','outflowLoad')
                fine_sed_model = transport_node

            elif model_type == n.InstreamCoarseSediment:
                # reach_template.define_input(transport_node,'incomingMass','generatedLoad')
                reach_template.add_link(OWLink(constituent_lag_node,'outflow',transport_node,'incomingMass'))

                reach_template.add_link(OWLink(bank_erosion,'bankErosionCoarse',transport_node,'incomingMass'))
                reach_template.define_output(transport_node,'loadDownstream','outflowLoad')

            elif model_type == n.InstreamDissolvedNutrientDecay:
                dis_nut_models.append(transport_node)
                # reach_template.define_input(transport_node,'incomingMassLateral','generatedLoad')
                reach_template.add_link(OWLink(constituent_lag_node,'outflow',transport_node,'incomingMassLateral'))
                if self.routing is not None:
                    reach_template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))
                    reach_template.add_link(OWLink(routing_node,'storage',transport_node,'reachVolume'))

                reach_template.define_output(transport_node,'loadDownstream','outflowLoad')
#            elif model_type == n.InstreamParticulateNutrient: TODO
            else:
                # Lumped constituent routing
                # reach_template.define_input(transport_node,'lateralLoad','generatedLoad')
                reach_template.add_link(OWLink(constituent_lag_node,'outflow',transport_node,'lateralLoad'))
                # reach_template.add_link(OWLink(lag_node,'outflow',transport_node,'inflow')) # inflow removed from LumpedConstituentRouting. Unused
                if self.routing is not None:
                    reach_template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))
                    reach_template.add_link(OWLink(routing_node,'storage',transport_node,'storage'))
                reach_template.define_output(transport_node,'outflowLoad')

        if fine_sed_model is not None:
            for dnm in dis_nut_models:
                reach_template.add_link(OWLink(fine_sed_model,'floodplainDepositionFraction',dnm,'floodplainDepositionFraction'))

        return reach_template

    def cgu_factory(self,cgu):
        cropping_cgu = (self.pesticide_cgus is not None) and (cgu in self.pesticide_cgus)
        cropping_cgu = cropping_cgu or ((self.timeseries_sediment_cgus is not None) and (cgu in self.timeseries_sediment_cgus))
        gully_proc = (self.gully_cgus is None) or (cgu in self.gully_cgus)
        hillslope_proc = (self.hillslope_cgus is None) or (cgu in self.hillslope_cgus)

        emc_proc = False
        if self.sediment_fallback_cgu is not None:
            emc_proc = cgu in self.sediment_fallback_cgu

        if cgu=='Water/lakes':
            return NilCGU()
        # if cgu in ['Dryland', 'Irrigation', 'Horticulture', 'Irrigated Grazing']:
        #     return DynamicSednetAgCGU()
        return DynamicSednetCGU(cropping_cgu=cropping_cgu,
                                sediment_fallback_model=emc_proc,
                                gully_cgu=gully_proc,
                                hillslope_cgu=hillslope_proc,
                                ts_load_with_dwc=self.ts_load_with_dwc)

    def get_template(self,**kwargs):
        tag_values = list(kwargs.values())
        template = OWTemplate('catchment')

        hrus={}
        for hru in self.hrus:
            hru_template = OWTemplate('hru:%s'%hru)

            if self.rr is not None:
                runoff_template = OWTemplate('runoff:%s'%hru)
                runoff_node = runoff_template.add_node(self.model_for(self.rr,hru,*tag_values),process='RR',hru=hru,**kwargs)
                runoff_template.define_output(runoff_node,'runoff')
                runoff_template.define_output(runoff_node,'surfaceRunoff','quickflow')
                runoff_template.define_output(runoff_node,'baseflow')
                hru_template.nest(runoff_template)
            hrus[hru] = hru_template
            template.nest(hru_template)

        for cgu in self.cgus:
            hru = self.cgu_hrus[cgu]
            cgu_builder = self.cgu_factory(cgu)
            cgu_template = cgu_builder.get_template(self,cgu=cgu,**kwargs)
            hrus[hru].nest(cgu_template)

        template.nest(self.reach_template(**kwargs))

        return template

    def link_catchments(self,graph,upstream,downstream):
        STANDARD_LINKS = defaultdict(lambda:[None,None],{
            n.InstreamFineSediment.name: ('incomingMass','loadDownstream'),
            n.InstreamCoarseSediment.name: ('incomingMass','loadDownstream'),
            n.InstreamDissolvedNutrientDecay.name: ('incomingMassUpstream','loadDownstream')
        })

        if (self._g == None)  or (self._g != graph):
            self._g = graph
            self._g_lookup = {}

        def match_node(nm):
            if nm in self._g_lookup:
                return self._g_lookup[nm]
            for nn in graph.nodes:
                if nn.startswith(nm):
                    self._g_lookup[nm] = nn
                    return nn
            return None
            
        linkages = [('%s-FlowRouting','outflow','inflow')] + \
                   [('%%s-ConstituentRouting-%s'%c,'outflowLoad','inflowLoad') for c in self.constituents]
        for (lt,src,dest) in linkages:

            src_node = lt%(str(upstream))
            dest_node = lt%(str(downstream))#'%d/%s'%(to_cat,lt)
            src_node = match_node(src_node)#[n for n in graph.nodes if n.startswith(src_node)][0]
            dest_node = match_node(dest_node)#[n for n in graph.nodes if n.startswith(dest_node)][0]

            if (src_node is None) and (dest_node is None):
                # If both are missing then assume process is not being modelled
                continue
            src_model = graph.nodes[src_node][TAG_MODEL]
            dest_model = graph.nodes[dest_node][TAG_MODEL]
            src = STANDARD_LINKS[src_model][1] or src
            dest = STANDARD_LINKS[dest_model][0] or dest
            # print(src_node,src,dest_node,dest)
            graph.add_edge(src_node,dest_node,src=[src],dest=[dest])

def main_output_flux(model):
    if model.name=='PassLoadIfFlow':
        return 'outputLoad'
    return 'totalLoad'

