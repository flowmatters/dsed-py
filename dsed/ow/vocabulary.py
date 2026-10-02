'''
Vocabulary for Openwater Dynamic SedNet budgets.

Budget tables describe each budget term with an element type, a budget element and a process. The names
follow Openwater where it has a natural name, and Source's Dynamic SedNet reporting otherwise. Converting to
Source's own reporting terminology is done separately (see dsed.ow.source_format).
'''

# Element types: where a budget term applies
CATCHMENT = 'Catchment'
LINK = 'Link'
NODE = 'Node'

# Processes: the role of a budget term in the mass balance
#   Initial + Supply + In Flow = Loss + Yield + Residual
INITIAL = 'Initial'
SUPPLY = 'Supply'
IN_FLOW = 'In Flow'
LOSS = 'Loss'
YIELD = 'Yield'
RESIDUAL = 'Residual'

# Budget elements
# Generation (catchment)
HILLSLOPE = 'Hillslope'
GULLY = 'Gully'
QUICKFLOW = 'Quickflow'       # concentration (EMC) or time series loads carried by quickflow
BASEFLOW = 'Baseflow'         # concentration (DWC) loads carried by baseflow
LEACHED = 'Leached'           # time series loads carried by baseflow
# In-stream (link)
STREAMBANK = 'Streambank'
POINT_SOURCE = 'Point Source'
FLOOD_PLAIN_DEPOSITION = 'Flood Plain Deposition'
CHANNEL_DEPOSITION = 'Channel Deposition'  # net: negative when remobilising
CHANNEL_STORAGE = 'Channel Storage'        # mass in the channel bed store
STREAM_DECAY = 'Stream Decay'
# Nodes
EXTRACTION = 'Extraction'
RESERVOIR_DEPOSITION = 'Reservoir Deposition'
RAINFALL = 'Rainfall'
EVAPORATION = 'Evaporation'
# Links and nodes
INFLOW = 'Inflow'
OUTFLOW = 'Outflow'
STORAGE = 'Storage'

FLOW = 'Flow'  # the constituent name used for water

KG = 'kg'
M3 = 'm3'
