import io
import json
import os


def extract_source_config(v,dest,progress=print):
    def writeable(fn):
        return open(os.path.join(dest,fn),'w')

    def write_json(fn,data):
        json.dump(data,writeable(fn+'.json'))

    def write_csv(fn,df):
        df.to_csv(os.path.join(dest,fn+'.csv'))

    fus = set(v.model.catchment.get_functional_unit_types())
    constituents = v.model.get_constituents()
    constituent_sources = v.model.get_constituent_sources()
    assert len(constituent_sources)==1 # We don't use constituent source

    network = v.network()
    network_df = network.as_dataframe()

    fu_areas = v.retrieve_csv('/tables/fus')
    fu_areas = pd.DataFrame.from_csv(io.StringIO(fu_areas))

    runoff_params = v.model.catchment.runoff.tabulate_parameters()
    actual_rr_types = set(v.model.catchment.runoff.get_param_values('theBaseRRModel'))
    assert len(actual_rr_types) == 1
    sac_parameters = {k:v.model.catchment.runoff.get_param_values('theBaseRRModel.%s'%k) for k in v.model.find_parameters('TIME.Models.RainfallRunoff.Sacramento.Sacramento')}
    name_columns = v.model.catchment.runoff.name_columns
    sac_names = list(v.model.catchment.runoff.enumerate_names())
    sac_names = {col:[v[i] for v in sac_names] for i,col in enumerate(name_columns)}
    runoff_parameters = pd.DataFrame(dict(**sac_names,**sac_parameters))

    runoff_inputs = v.model.catchment.runoff.tabulate_inputs('Dynamic_SedNet.Models.Rainfall.DynSedNet_RRModelShell')

    progress('Getting data sources')
    data_sources = v.data_sources()
    #[ds['FullName'] for ds in data_sources]
    progress('Getting climate data')
    climate = get_big_data_source(v,'Climate Data',data_sources,progress)

    progress('Getting usle timeseries')
    usle_ts = v.data_source('USLE Data')
    usle_timeseries = usle_ts['Items'][0]['Details']

    progress('Getting gully timeseries')
    gully_ts = v.data_source('Gully Data')
    gully_timeseries = gully_ts['Items'][0]['Details']

    progress('Getting cropping metadata')
    cropping_ts = get_big_data_source(v,'Cropping Data',data_sources,progress)

    generation_models = v.model.catchment.generation.model_table()
    generation_parameters = v.model.catchment.generation.tabulate_parameters()

    link_models = v.model.link.routing.model_table()
    link_params = v.model.link.routing.tabulate_parameters()

    transport_models = v.model.link.constituents.model_table()
    transport_params = v.model.link.constituents.tabulate_parameters()

    write_json('constituents',constituents)
    write_json('fus',list(fus))
    writeable('network.json').write(network_df.to_json())
    write_csv('runoff_params',runoff_parameters)
    write_csv('climate',climate)
    write_csv('usle_timeseries',usle_timeseries)
    write_csv('gully_timeseries',gully_timeseries)
    write_csv('cropping',cropping_ts)

    for model_type, table in generation_parameters.items():
        write_csv('cg-%s'%model_type,table)

    write_csv('cgmodels',generation_models)
    write_csv('routing_models',link_models)
    write_csv('routing_params',link_params)
    write_csv('transportmodels',transport_models)

    for model_type, table in transport_params.items():
        write_csv('cr-%s'%model_type,table)

    write_csv('fu_areas',fu_areas)


def extract_source_results(v,dest,progress=print,start=None,end=None):
    def _ensure():
        if not os.path.exists(dest):
            os.makedirs(dest)

    def writeable(fn):
        _ensure()
        return open(os.path.join(dest,fn),'w')

    def write_json(fn,data):
        json.dump(data,writeable(fn+'.json'))

    def write_csv(fn,df):
        _ensure()
        df.to_csv(os.path.join(dest,fn+'.csv'))

    constituents = v.model.get_constituents()
    recorders = [
        {'RecordingElement':'Downstream Flow Volume'},
        {'RecordingElement':'Upstream Flow Volume'},
        {'RecordingVariable':'Quick Flow'},
        {'RecordingVariable':'Slow Flow'},
        {'RecordingVariable':'Total Flow'},
    ]
    recorders += [{'RecordingVariable':'Constituents@%s@Downstream Flow Mass'%c} for c in constituents]
    recorders += [{'RecordingVariable':'Constituents@%s@Total Flow Mass'%c} for c in constituents]

    v.configure_recording(enable=recorders)
    progress('Configured recorders')

    v.drop_all_runs()
    v.model.simulation.configure_assurance_rule(level='Warning',category='Data Sources')

    v.run_model(start=start,end=end)
    progress('Simulation done.')

    r = v.retrieve_run()
    results_df = r['Results'].as_dataframe()
    write_csv('results',results_df)
    variables = set(results_df.RecordingVariable)

    downstream = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Downstream Flow Volume'},name_fn=veneer.name_for_location)
    progress('Got downstream flow')

    upstream = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Downstream Flow Volume'},name_fn=veneer.name_for_location)
    progress('Got upstream flow')

    write_csv('upstream_vol',upstream)
    write_csv('downstream_vol',downstream)

    runoff = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Quick Flow'},name_fn=veneer.name_for_fu_and_sc)
    baseflow = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Slow Flow'},name_fn=veneer.name_for_fu_and_sc)
    totalflow = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Total Flow'},name_fn=veneer.name_for_fu_and_sc)
    write_csv('runoff',runoff)
    write_csv('baseflow',baseflow)
    write_csv('totalflow',totalflow)

    def download_constituent_outputs(suffix,fn_suffix,name_fn=veneer.name_for_location):
        constituent_variables = [v for v in variables if v.startswith('Constituents@') and v.endswith(suffix)]
        progress(constituent_variables)
        for cv in constituent_variables:
            con = cv.split('@')[1].replace(' ','')
            progress(con)
            ts = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':cv},name_fn=name_fn)
            write_csv(con+fn_suffix,ts)
            progress('Downloaded %s %s'%(con,fn_suffix))

    download_constituent_outputs('Downstream Flow Mass','network')
    download_constituent_outputs('Total Flow Mass','generation',veneer.name_for_fu_and_sc)