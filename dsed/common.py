
from .compare_results import compare
from . import stats
from . import timeclass

def init(base_directory,obs_directory=None,timeperiods=None):
	global BASE_DIRECTORY,OBSERVED_DIRECTORY,TIME_PERIODS
	BASE_DIRECTORY = base_directory
	OBSERVED_DIRECTORY = obs_directory
	TIME_PERIODS = timeperiods

def _globFilenamesOnly(directory,search):
    from glob import glob
    import os
    return [fn.split(os.path.sep)[-1] for fn in glob(os.path.join(directory,search))]

def available():
    return _globFilenamesOnly(BASE_DIRECTORY,'*')

def value_diff(dfL,dfR):
    import numpy as np
    oCols = list(dfL.columns[np.where(dfL.dtypes=='object')])
    numericCols = list(set(dfL.columns)-set(oCols))
    delta = dfL.copy()
    for col in numericCols:
        delta[col] = dfL[col] - dfR[col]
    return delta

def Veneer(*args,**kwargs):
    from veneer import Veneer
    result = Veneer(*args,**kwargs)

    def configureOptions(self,options):
        self.configure_options(options)

    def configure_options(self,options):
        lines = ["# Generated Script","from Dynamic_SedNet.PluginSetup import DSScenarioDetails"]
        lines += ["DSScenarioDetails.%s = %s"%(k,v) for (k,v) in options.items()]
        script = '\n'.join(lines)
        #print(script)
        res = self.model._safe_run(script)

    def set_run_name(self,name):
        self.model.set('scenario.CurrentConfiguration.runName',name,literal=True)

    def set_output_path(self,name):
        self.model.set('scenario.CurrentConfiguration.ModelOutputPath',name.replace('\\','\\\\'),literal=True)

    def run_contributor(self,results,fn):
        import os
        script = """
import Dynamic_SedNet.Results.Contributor.RegionalReportingContributor as RegionalReportingContributor
import Dynamic_SedNet.Tools.ToolsGeneral as ToolsGeneral
contrib = RegionalReportingContributor()
contrib.UseResults('%s')
contrib.Scenario = scenario
contrib.runTimeStep()
if contrib.Success:
  result = None
  ToolsGeneral.ExportTableToCsvString(contrib.contribsTab,True,'%s')
else:
  result = contrib.Error
"""%(results.path,os.path.join(results.path,'%s.csv'%fn))
        self.model.runScript(script)

    result.configureOptions = configureOptions.__get__(result,result.__class__)
    result.configure_options = configure_options.__get__(result,result.__class__)
    result.set_run_name = set_run_name.__get__(result,result.__class__)
    result.set_output_path = set_output_path.__get__(result,result.__class__)
    result.run_contributor = run_contributor.__get__(result,result.__class__)
    return result

class Results(object):
    def __init__(self,run_name):
        from .queries import ResultsQueries
        import os
        self._time_periods = None
        self.run_name = run_name
        self.path = os.path.join(BASE_DIRECTORY,run_name)
        self.runDetails = RunDetails(self.path)
        self.queries = ResultsQueries(self)
        self._loaded = {}

    def _apply_time_series_helpers(self,dataframe):
        from pandas import DataFrame
#        import types
        def of_timeclass(df_self,timeclass):
            time_periods = self.time_periods().ix[df_self.index]
            if isinstance(timeclass,str):
                result = df_self[time_periods['cat']==timeclass]
            else:
                result = df_self[time_periods['cat'].apply(timeclass)]
            self._apply_time_series_helpers(result)
            return result
           

        def by_timeclass(df_self):
            time_periods = self.time_periods().ix[df_self.index]
            result = df_self.groupby(time_periods['cat'])
            #self._apply_time_series_helpers(result)
            return result


        def by_wateryear(df_self):
            time_periods = self.time_periods().ix[df_self.index]
            result = df_self.groupby(time_periods.Water_year)
            #self._apply_time_series_helpers(result)
            return result

        def of_month(df_self,month):
            result = df_self[df_self.index.month==month]
            self._apply_time_series_helpers(result)
            return result

        def by_month(df_self):
            result = df_self.groupby(df_self.index.month)
#            self._apply_time_series_helpers(result)
            return result

        def of_year(df_self,year):
            result = df_self[df_self.index.year==year]
            self._apply_time_series_helpers(result)
            return result

        def by_year(df_self):
            result = df_self.groupby(df_self.index.year)
#            self._apply_time_series_helpers(result)
            return result

        meths = {'of_timeclass':of_timeclass,
            'by_timeclass':by_timeclass,
            'by_wateryear':by_wateryear,
            'of_month':of_month,
            'by_month':by_month,
            'of_year':of_year,
            'by_year':by_year,
        }

        dataframe.__class__ = type('JoelsDataFrame',(DataFrame,),meths)
#        dataframe.__setattr__('of_timeclass',types.MethodType(of_timeclass,dataframe))
#        dataframe.by_timeclass = types.MethodType(by_timeclass,dataframe)
#        dataframe.by_wateryear = types.MethodType(by_wateryear,dataframe)
#        dataframe.of_month = types.MethodType(of_month,dataframe)
#        dataframe.by_month = types.MethodType(by_month,dataframe)
#        dataframe.of_year = types.MethodType(of_year,dataframe)
#        dataframe.by_year = types.MethodType(by_year,dataframe)

    
#        dataframe.__orig_dir__ = dataframe.__dir__
#        def new_dir(df_self):
#            return df_self.__orig_dir__() + ['of_timeclass','of_month','of_year',
#                'by_year', 'by_month', 'by_wateryear', 'by_timeclass']

#        dataframe.__setattr__('__dir__',types.MethodType(new_dir,dataframe))

    def available(self):
        return [fn[0:-4] for fn in _globFilenamesOnly(self.path,'*.csv')]

    def get(self,results_type):
        if not results_type in self._loaded:
            import pandas as pd
            import os
            self._loaded[results_type] = pd.DataFrame.from_csv(os.path.join(self.path,results_type+'.csv'))
        return self._loaded[results_type]

    def get_node_ts(self,site,constituents,header=True):
        import pandas as pd
        import os
        if isinstance(constituents,str):
            constituents = [constituents]

        def _fn(con):
            return os.path.join(self.path,'TimeSeries',con,"%s_%s_kilograms.csv"%(con,site))
        def _read(con):
            header_rows = 1 if header else 0
            return pd.read_csv(_fn(con),index_col=0,parse_dates=True,dayfirst=True,names = ['Date', con],skiprows=1)
        result = pd.DataFrame([_read(c)[c] for c in constituents]).transpose()
        self._apply_time_series_helpers(result)
        return result

    def load_observed_ts(self,site,fn=None,location=None):
        """
        Load a time series file of observed loads for a given site.   
        """
        import pandas as pd
        if fn is None:
            if location is None:
                location = OBSERVED_DIRECTORY
            import os
            from glob import glob
            matching_files = glob(os.path.join(location,"*%s*.csv"%site))
            fn = matching_files[0]
        result = pd.read_csv(fn,index_col=0,parse_dates=True,dayfirst=True)
        self._apply_time_series_helpers(result)
        return result


    def time_periods(self,fn=None):
        if self._time_periods is None:
            if fn is None:
                fn = TIME_PERIODS
            import pandas as pd
            self._time_periods = pd.read_csv(fn,index_col = 1, parse_dates = [1], dayfirst = True, names=["Water_year","date","cat"], skiprows =1)
        return self._time_periods

class DifferenceResults(object):
    def __init__(self,run1,run2):
        from .queries import ResultsQueries
        self._run1 = Results(run1)
        self._run2 = Results(run2)
        self.run_name = '('+self._run1.run_name + ' - ' + self._run2.run_name+')'
        self.path = None
        self.runDetails = self._run1.runDetails
        self.queries = ResultsQueries(self)

    def available():
        return self._run1.available()

    def get(self,results_type):
        return value_diff(self._run1.get(results_type),self._run2.get(results_type))

    def get_node_ts(self,site,constituents):
        return value_diff(self._run1.get_node_ts(site,constituents),self._run2.get(site,constituents))

    def load_observed_ts(self,site,fn=None,location=None):
        return self._run1.load_observed_ts(site,fn,location)

class RunDetails(object):
    def __init__(self,path):
        import os
        import xml.etree.ElementTree
        self._data = xml.etree.ElementTree.parse(os.path.join(path,'DSScenarioRunInfo.xml')).getroot()
        self.start = self._extractDate('startDate')
        self.startRecording = self._extractDate('startRecDate')
        self.end = self._extractDate('endDate')
        simulationLength = self.end - self.startRecording
        self.yearsOfRecording = simulationLength.days / 365.25

    def _extractDate(self,element):
        from datetime import datetime
        return datetime.strptime(self._data.find(element).text,'%Y-%M-%dT00:00:00')


class MyTest(object):
    def __dir__(self):
        return ['Hi']
