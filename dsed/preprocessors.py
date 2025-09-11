'''
Running the Dynamic Sednet preprocessors from Python
'''

import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)


PREPROCESSORS=[
    ('climate','Dynamic_SedNet.Parameterisation.Models.ClimateCollationModel',None),
    ('gully','Dynamic_SedNet.Parameterisation.Models.GullyParameterisationModel','GullyParameters'),
    ('usle','Dynamic_SedNet.Parameterisation.Models.CoverTimeSeries_SpatialPreprocessorModel',['ResultsTable','globalAverageResultsTable']),
    ('gbrusle','GBR_DynSed_Extension.Parameterisation.Models.GBRUSLECoverTimeSeriesSpatialPreprocessorModel',None),
    ('lewis','GBR_DynSed_Extension.Parameterisation.Models.LewisTrappingParameterisationModel',None)
]

def _generate_preprocess_fn(current_module,name,klass,output_name):
    logger.debug(f'{name}, {klass}')
    def new_fn(v,**kwargs):
        return run_preprocessor(v,klass,output_name,**kwargs)
    new_fn.__name__ = 'run_%s'%name
    setattr(current_module,new_fn.__name__,new_fn)
    doc = 'See %s_default_params() for parameter names and default values\n'%name
    if output_name:
        doc += '\nReturns: %s'%str(output_name)

    def param_fn(v):
        return v.model.find_default_parameters(klass)
    param_fn.__name__ = '%s_default_params'%name
    setattr(current_module,param_fn.__name__,param_fn)

def _generate_preprocessor_functions():
    import sys
    current_module = sys.modules[__name__]
    for (name,klass,output_name) in PREPROCESSORS:
        _generate_preprocess_fn(current_module,name,klass,output_name)

def convert_parameter(param):
    if type(param)==list and len(param):
        return "tuple(%s)"%param
#        if type(param[0])==int:
#            return "Array[int](%s)"%param
#        elif type(param[0])==float:
#            return "Array[float](%s)"%param
#        else:
#            return "Array[str](%s)"%param
    return str(param)

def run_preprocessor(v,preprocessor,result_name=None,**kwargs):
    short_name = preprocessor.split('.')[-1]
    outputs = v.model.find_outputs(preprocessor)
    
    script = """
    import %s as %s
    import Dynamic_SedNet.Tools.ToolsModel as ToolsModel
    
    p = %s()
    p.Scenario = scenario
    p.CatPolys = scenario.GeographicData.IntersectedSCFUPolygons[1]
    p.FuPolys = scenario.GeographicData.IntersectedSCFUPolygons[1]
    """%(preprocessor,short_name,short_name)
    script = v.model.clean_script(script)
    for param,val in kwargs.items():
        script += 'p.%s = %s\n'%(param,convert_parameter(val))
    script += 'p.runTimeStep()\n'
#    script += 'result = ToolsModel.PopulateGullyParameterDataTable(scenario)\n'
    script += 'result = {}\n'
    for output in outputs:
        script += 'result["%s"] = p.%s\n'%(output,output)
    result = v.model._safe_run(script)
    result = v.model.simplify_response(result['Response'])
    if result_name is None:
        return result

    if isinstance(result_name,str):
        return pd.DataFrame(result[result_name])

    return {rn:pd.DataFrame(result[rn]) for rn in result_name}

_generate_preprocessor_functions()

def raster_path(basename):
    return "'"+os.path.join(os.getcwd(),'Inputs/',basename+'.asc')+"'"

