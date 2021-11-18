import io
import json
import os
import pandas as pd
from veneer.extract_config import *
from veneer.actions import get_big_data_source
import veneer

def _BEFORE_BATCH_NOP(slf,x,y):
    pass

class DynamicSednetExtractor(SourceExtractor):
    def __init__(self,v,dest,results=None,progress=print):
        super().__init__(v,dest,results,progress)

    def _extract_runoff_configuration(self):
        runoff_params = self.v.model.catchment.runoff.tabulate_parameters()
        actual_rr_types = set(self.v.model.catchment.runoff.get_param_values('theBaseRRModel'))
        assert len(actual_rr_types) == 1
        sac_parameters = {k:self.v.model.catchment.runoff.get_param_values('theBaseRRModel.%s'%k) for k in self.v.model.find_parameters('TIME.Models.RainfallRunoff.Sacramento.Sacramento')}
        name_columns = self.v.model.catchment.runoff.name_columns
        sac_names = list(self.v.model.catchment.runoff.enumerate_names())
        sac_names = {col:[v[i] for v in sac_names] for i,col in enumerate(name_columns)}
        runoff_parameters = pd.DataFrame(dict(**sac_names,**sac_parameters))

        runoff_inputs = self.v.model.catchment.runoff.tabulate_inputs('Dynamic_SedNet.Models.Rainfall.DynSedNet_RRModelShell')
        self.write_csv('runoff_params',runoff_parameters)

        self.progress('Getting climate data')
        climate = get_big_data_source(self.v,'Climate Data',self.data_sources,self.progress)

        self.write_csv('climate',climate)

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


