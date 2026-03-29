"""
Constants for APSIM and HowLeaky cropping parameterisation.

Replaces C# StringConstants, GBRStringConstants, and SedCons field/constituent
name definitions used in the Dynamic SedNet GBR extension.
"""

# --- Separator used in file naming (Catchment$FU$Constituent) ---
SEPARATOR = "$"

# --- Shapefile field names (from Dynamic_SedNet.Tools.StringConstants) ---
SUBCATCH_FIELD = "SUBCATS"
FU_FIELD = "FUS"
AREA_FIELD = "AREA"
SLOPE_FIELD = "SLOPE"
CLAYPERC_FIELD = "CLAYPERC"
FRAMEWORK_FIELD = "FRAMES"
CLIMATE_FIELD = "CLIMATE"

# Surface nutrient concentration fields
SURF_P_FIELD = "Surf_P"
SURF_N_FIELD = "Surf_N"
ORGANIC_CARBON_FIELD = "OC"

# Australian Soil Classification
ASC_FIELD = "ASC"

# Colwell P
COLWELL_P_FIELD = "Colwell_P"

# SDR / DWC / delivery ratio fields (used by HowLeaky TS importer)
SDR_FINE_FIELD = "SDR_Fine"
SDR_COARSE_FIELD = "SDR_Coarse"
DWC_FIELD = "DWC"
DWC_FINE_FIELD = "DWC_Fine"
DWC_COARSE_FIELD = "DWC_Coarse"
DELIV_RATIO_FIELD = "Del_Ratio"
DELIV_RATIO_DISSOLVED_FIELD = "DR_Diss"
CONV_FACT_FIELD = "Conv_Fact"

# --- APSIM-specific shapefile fields (from GBRStringConstants) ---
APSIM_SLOPE_FIELD = "APSIMSlope"
APSIM_SLOPELENGTH_FIELD = "APSIMLen"
APSIM_NLEACHED_FIELD = "NLeached"
APSIM_REGION_FIELD = "APSIMReg"
APSIM_SOIL_FIELD = "APSIMSoil"
APSIM_CLIMATE_FIELD = "APSIMClim"
APSIM_MGTZONE_FIELD = "APSIMZone"
APSIM_PERM_FIELD = "APSIMPerm"
APSIM_RUN_K_FIELD = "RunKmet"
APSIM_FU_K_FIELD = "FuKsi"

# --- HowLeaky-specific shapefile fields ---
HL_SLOPE_FIELD = "HLSlope"
HL_SLOPELENGTH_FIELD = "HLLength"

# --- Constituent names (from Dynamic_SedNet.Tools.SedCons) ---
FINE_SED = "Sediment - Fine"
COARSE_SED = "Sediment - Coarse"
N_DIN = "N_DIN"
N_DON = "N_DON"
N_PARTICULATE = "N_Particulate"
P_PARTICULATE = "P_Particulate"
P_FRP = "P_FRP"
P_DOP = "P_DOP"

# --- Pesticide names ---
AMETRYN = "Ametryn"
ATRAZINE = "Atrazine"
DIURON = "Diuron"
HEXAZINONE = "Hexazinone"
IMIDACLOPRID = "Imidacloprid"
ISOXAFLUTOLE = "Isoxaflutole"
METOLACHLOR = "S-metolachlor"
METRIBUZIN = "Metribuzin"
METSULFURON_METHYL = "Metsulfuron-methyl"
TEBUTHIURON = "Tebuthiuron"
TWOFOUR_D = "24-D"

# --- Pesticide alternative name mapping ---
PESTICIDE_ALT_NAMES = {
    METOLACHLOR: "SMetolachlor",
    TWOFOUR_D: "twofourD",
}

# HowLeaky Rattray alternative for 2,4-D
RATTRAY_TWOFOUR_D = "24_d"

# --- Phase names (for pesticide timeseries file naming) ---
SED_PHASE = "SedimentPhase"
WATER_PHASE = "WaterPhase"

# --- Aggregated runoff string ---
AGGREGATED_RUNOFF = "Aggregated Runoff"

# --- DOP / FRP proportions ---
P_DOP_PROPORTION = 0.2
P_FRP_PROPORTION = 0.8

# --- Unit strings used in file naming ---
UNITS_T_PER_HA = "TperHa"
UNITS_KG_PER_HA = "KgPerHa"
UNITS_G_PER_HA = "gPerHa"
UNITS_MM = "mm"

CONSTITUENT_UNITS = {
    FINE_SED: UNITS_T_PER_HA,
    N_DIN: UNITS_KG_PER_HA,
    N_DON: UNITS_KG_PER_HA,
    N_PARTICULATE: UNITS_KG_PER_HA,
    P_PARTICULATE: UNITS_KG_PER_HA,
    P_FRP: UNITS_KG_PER_HA,
    P_DOP: UNITS_KG_PER_HA,
    AGGREGATED_RUNOFF: UNITS_MM,
}

# --- Adjusted timeseries output filename component ---
TS_DAILY_ADJUSTED = "Daily_Total_Adjusted_Load"
TS_DAILY_NOT_ADJUSTED = "Daily_Not_Adjusted_Load"

# --- HowLeaky output field name strings (for Rattray naming variants) ---
HL_EROSION_RATTRAY = "Erosion"
HL_N03_RUNOFF_RATTRAY = "N03NRunoffLoad"
HL_NO3_RUNOFF_CORRECT_RATTRAY = "NO3NRunoffLoad"
HL_DIN_DRAINAGE_RATTRAY = "DIN_Drainage"
HL_PEST_SED_PHASE_RATTRAY = "Pest_Lost_In_Runoff_Sediment"
HL_PEST_WATER_PHASE_RATTRAY = "Pest_Lost_In_Runoff_Water"
HL_DISSOLVED_P_OUTPUT = "Dissolved P Export"
