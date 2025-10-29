import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

RR='Retreat Rate'
BE='Bank Erodibility'
MABE='Mean Annual Bank Erosion'
MCF='Mass Conversion Factor'
MRR='Modelled Retreat Rate'

ADDED_PARAMETERS = [RR,BE,MABE,MRR,MCF]

def compute_streambank_parameters(params,*args):
    logger.info('Computing derived parameters (Retreat Rate)')
    BEMF='Bank Erosion Management Factor'
    BEC='Bank Erosion Coefficent'
    SLOPE='Link Slope'
    BFD='Bank Full Flow'
    SBD='Sediment Bulk Density'
    BH='Bank Height'
    LL='Link Length'
    SE = 'Soil Erodibility'
    RVP = 'Riparian Vegetation Percentage'
    MRVE = 'Maximum Riparian Vegetation Effectiveness'

    streambank_params = params[(params.ELEMENT!='Node')&params.PARAMETER.isin([BEMF,BEC,SLOPE,BFD,SBD,BH,LL,SE,RVP,MRVE])]

    index_cols = set(streambank_params.columns)-{'PARAMETER','VALUE'}
    streambank_params = streambank_params.pivot(index=index_cols,columns='PARAMETER',values='VALUE')
    streambank_params = streambank_params.astype('f')
    density_water = 1000.0 # kg.m^-3
    gravity = 9.81        # m.s^-2

    streambank_params[RR] = streambank_params[BEC] * \
                            streambank_params[BEMF] * \
                            streambank_params[BFD] * \
                            streambank_params[SLOPE] * \
                            density_water * \
                            gravity
    soil_erodibility = streambank_params[SE] * 0.01
    riparian_efficacy = np.minimum(streambank_params[RVP],streambank_params[MRVE]) * 0.01
    streambank_params[BE] = (1 - riparian_efficacy) * soil_erodibility
    streambank_params[MCF] = streambank_params[SBD] * streambank_params[LL] * streambank_params[BH]
    streambank_params[MRR] = streambank_params[RR] * streambank_params[BE]
    streambank_params[MABE] = streambank_params[MRR] * streambank_params[MCF]

    streambank_params = pd.melt(streambank_params,ignore_index=False,value_name='VALUE').reset_index()
    computed_params = streambank_params[streambank_params.PARAMETER.isin([RR,BE,MABE,MRR,MCF])]

    params = pd.concat([params,computed_params])
    return params
