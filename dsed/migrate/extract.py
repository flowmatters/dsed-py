import io
import json
import os
import pandas as pd
from veneer.extract_config import SourceExtractor as VanillaSourceExtractor, extract
import veneer.extract_config as ec
from veneer.actions import get_big_data_source
import veneer

PLUGINS=[
    'Dynamic_SedNet.dll',
    'GBR_DynSed_Extension.dll'
]

def _BEFORE_BATCH_NOP(slf,x,y):
    pass

class DynamicSednetExtractor(VanillaSourceExtractor):
    def __init__(self,v,dest,results=None,progress=print):
        super().__init__(v,dest,results,progress)

    def _extract_runoff_configuration(self):
        runoff_params = self.v.model.catchment.runoff.tabulate_parameters()
        actual_rr_types = set(self.v.model.catchment.runoff.get_param_values('theBaseRRModel'))
        if len(actual_rr_types) != 1:
            msg = f'Expected a single rainfall runoff model to be used everywhere. Have {actual_rr_types}'
            print(msg)
            self.progress(msg)
            assert False

        model_type = actual_rr_types.pop()
        rr_parameters = {k:self.v.model.catchment.runoff.get_param_values('theBaseRRModel.%s'%k) for k in self.v.model.find_parameters(model_type)}
        name_columns = self.v.model.catchment.runoff.name_columns
        rr_names = list(self.v.model.catchment.runoff.enumerate_names())
        rr_names = {col:[v[i] for v in rr_names] for i,col in enumerate(name_columns)}
        runoff_parameters = pd.DataFrame(dict(**rr_names,**rr_parameters))
        runoff_inputs = self.v.model.catchment.runoff.tabulate_inputs('Dynamic_SedNet.Models.Rainfall.DynSedNet_RRModelShell')
        self.write_csv('runoff_params',runoff_parameters)

        self.progress('Getting climate data')
        climate = get_big_data_source(self.v,'Climate Data',self.data_sources,self.progress)

        self.write_csv('climate',climate)

        runoff_models = pd.DataFrame(dict(**rr_names))
        runoff_models['model'] = model_type
        self.write_csv('runoff_models',runoff_models)

    def _extract_generation_configuration(self):
        try:
            self.progress('Getting usle timeseries')
            usle_ts = self.v.data_source('USLE Data')
            usle_timeseries = usle_ts['Items'][0]['Details']
            self.write_csv('usle_timeseries',usle_timeseries)
        except:
            self.progress('No USLE timeseries')

        try:
            self.progress('Getting gully timeseries')
            gully_ts = self.v.data_source('Gully Data')
            gully_timeseries = gully_ts['Items'][0]['Details']
            self.write_csv('gully_timeseries',gully_timeseries)
        except:
            self.progress('No gully timeseries')

        try:
            self.progress('Getting cropping metadata')
            cropping_ts = get_big_data_source(self.v,'Cropping Data',self.data_sources,self.progress)
            self.write_csv('cropping',cropping_ts)
        except:
            self.progress('No cropping metadata')

        super()._extract_generation_configuration()

def _arg_parser():
    parser = ec._arg_parser()
    parser.add_argument('--pluginpath',help='Path to Dynamic Sednet plugins')
    return parser

if __name__=='__main__':
    args = ec._parsed_args(_arg_parser())
    args['plugins'] += [os.path.abspath(os.path.join(args['pluginpath'] or '.',p)) for p in PLUGINS]
    extract(DynamicSednetExtractor,**args)

