# Dynamic Sednet in OpenWater

This document describes and demonstrates the use of Dynamic Sednet within the Openwater system.

## Openwater implementation of Dynamic Sednet

The Openwater implementation of Dynamic Sednet is intended to produce the same results as the implementation of Dynamic Sednet as plugins to eWater Source. Additionally, the Openwater implementation also aims achieve performance and flexibility benefits compared to the eWater Source implementation.

## Openwater

Openwater is an open source hydrological modelling framework, able to run on Windows and Linux.

Like eWater Source, Openwater is built around the concept of component models, connected together to form a model of a system. Unlike Source, the Openwater component models are connected together in an arbitrary graph, such that the FU-Catchment-Link-Node structure is just *one* possible way of organising models. Openwater concepts are described [here](https://github.com/flowmatters/openwater/tree/master/doc)

Openwater has no graphical user interface. Rather, users work with openwater models through a Python scripting interface. The main [Openwater](https://github.com/flowmatters/openwater) package contains *generic* functionality for defining model graphs, parameterising and assigning input timeseries, running models and retrieving generic timeseries and tabular results. This package ([dsed-py](https://github.com/flowmatters/dsed-py)) builds upon the more generic package to provide additional functionality specific to Dynamic Sednet applications of Openwater.

## Example (existing model)

TODO: Notebook showing basic runtime, reporting with existing model

## Software

The following software packages are required to work with Dynamic Sednet in Openwater:

* **Core software**: Operating specific packages required
  * **openwater-core**: The main model engine software for Openwater. Developed in Go.
  * **HDF5**: Main data access library used by Openwater
* **Openwater Python packages**:
  * **openwater**: Core Openwater model setup and reporting functionality
  * **dsed-py**: Dynamic Sednet specific functionality, including model migration from Source
* **Required Python packages**:
  * **NetworkX**: Graph processing
  * **h5py**: HDF5 file management (for model files and for results)
  * **Standard Python numerical packages**: numpy, pandas
* **Optional Python packages**:
  * Spatial and terrain analysis (eg GDAL, TauDEM) - used for model setup from spatial data
  * Veneer-py - For accessing the Source version of Dynamic Sednet, including for extracting model setup for conversion to Openwater

## Model setup

At this point, the main way of configuring an Openwater Dynamic Sednet model is by converting an existing model from Source. This process is mostly automated with scripts for three key steps:

1. Extract model setup and model results from Source. Model setup includes the network definition, list of functional units and constituents, the model types used for each functional unit and constituent for runoff, constituent generation, streamflow routing and constituent transport. For the purposes of later comparison of model results, the script extracts full, daily timeseries from Source, rather than the summary results typically used with Dynamic Sednet.
2. Build Openwater implementation using extracted model setup data.
3. Run the Openwater implementation and compare the generated time series with the results from Source.

An Openwater Dynamic Sednet model could also be built wholly or partially 'from scratch' using Source data and relevent Python packages for spatial data analysis. The Openwater and dsed-py packages are source-data agnostic and will support model setup using a range of approaches. An example of setting up a simple catchment model (not Dynamic Sednet) can be found at **(TODO)**

In general, the model setup is a two step process:

1. **Structural setup** - which involves [building](https://github.com/flowmatters/openwater/blob/master/doc/templates.md), and [organising](https://github.com/flowmatters/openwater/blob/master/doc/dimensions.md), the graph of component models, such as all the rainfall runoff, constituent generation and routing models and how they interact.
2. **Parametiersation** - which involves [assigning](https://github.com/flowmatters/openwater/blob/master/doc/parameterisation.md) input timeseries and scalar parameters to all the component models that make up the model graph.

### Structural setup

For Dynamic Sednet, the most common way to establish the model graph is to convert an existing model from Source, using the functionality in the `migrate` namespace:

```python
import veneer
from openwater import discovery as disco
from dsed import migrate

disco.set_exe_path('D:/ow_install_dir')
disco.discover()

# 1. Extract model setup from Source
SOURCE_FILES = 'D:/dest_for_source_files'
v = veneer.Veneer()
extractor = migrate.extract.SourceExtractor(v,SOURCE_FILES)
extractor.extract_source_config()

# 2. Build the Openwater model
builder = migrate.build.SourceOpenwaterDynamicSednetMigrator(SOURCE_FILES)
model, meta, network = builder.build_ow_model()

START='1986/07/01'
END='2014/06/30'
TIME_PERIOD = pd.date_range(START,END)

# 3. Run the Openwater model and retrieve results
model.write_model('D:/shiny_new_model.h5',len(TIME_PERIOD))
results = model.run(TIME_PERIOD)
```

More generally, Openwater model graphs are typically constructed using a series of nested templates, that then get replicated across a model domain. For example, a 'template functional unit' that is replicated, for each type of functional unit, to form a 'template catchment', which is then replicated for each catchment to form the overall model. More details on this process are [here](https://github.com/flowmatters/openwater/blob/master/doc/templates.md).

### Parameterisation

Once an Openwater model is configured _structurally_, individual component models can be [configured](https://github.com/flowmatters/openwater/blob/master/doc/parameterisation.md) with values for:

1. Input timeseries
2. Scalar parameters
3. Initial state values

Openwater includes generic functionality for parameterising models from various data sources. 

The parameterisation process is configured using the Openwater concept of model _dimensions_, where the various component models, in the model graph, are organised according to a user defined set of dimensions. In the case of the Dynamic Sednet, the dimensions are catchment (which is synonymous with link), constituent, constituent generation unit (cgu) and hydrological response unit (hru). Both cgu and hru map to the Functional Units within Source. The Functional Unit concept is split into two dimensions (cgu and hru) to allow them to potentially be configured independently.

Openwater dimensions are described in more detail [here](https://github.com/flowmatters/openwater/blob/master/doc/dimensions.md). This is perhaps the most significant Openwater concept to grasp from an end user perspective as it is central to both parameterisation and reporting.

**Note:** When converting an existing model from Source, the conversion scripts apply all the existing model parameters and input time series.

## Running

A configured model can be run from Python using the `run` method in Openwater. This effectively runs the external `ow-sim` executable, which can also be run directly from the command line or another script.

There are currently limited options for `run`/`ow-sim`, although these will be expanded in the future, particularly around providing more options for handling outputs.

## Reporting

The `run` method returns an `OpenwaterResults` object, which provides access to the model results (and the input timeseries).

There are two main [data retrieval](https://github.com/flowmatters/openwater/blob/master/doc/reporting.md) methods in `OpenwaterResults`:

* `timeseries`: For retrieving tables of timeseries
* `table`: For retrieving tables of scalar results

It is expected that Dynamic Sednet specific reporting tools will be built upon these basic Openwater mehtods.

## Limitations

There are currently some limitations on the use of Dynamic Sednet within Openwater, including several model components that aren't integrated, and some functional limitations impacting performance.

### Model components

At present, the key water quality components are implemented, but, in the hydrological components, only the unregulated components are integrated.  Specifically, the process to convert models from Source doesn't include water storages and water offtakes. This work is ongoing. This means that instream results, including catchment constituent export, do not match the Source implementations. Subcatchment / FU level generation results are unaffected.

### Software functionality

Beyond the core modelling functionality, there are a number of areas where the software, including the Python packages, will be improved to offer more flexibility and user control, particularly with a view to improved performance.

Importantly, when a simulation is run, the model writes out full resolution timeseries for every defined output of every model, **and** for every input of every model (with the exception of models that don't receive any inputs from linked models). So, for example, every constituent generation model that is dependent on runoff, will have its input runoff recorded and written to disk. This leads to very large output files, and the time taken to writen these files becomes the most significant component of the model runtime.

We are exploring a number of ways to provide more control over model output production, including:

* Options for writing out results of different component models in parallel (currently, only one set of model results can be written at a time)
* Options to specify which components models to write outputs for, to reduce the overall output file size (and reduce runtimes)
* Options to provide 'custom' reporting tools that post-process the raw results and write out summaries, avoiding the need to save the full timeseries.

### System performance

Because the current Openwater software writes out very large output files, the software is typically I/O bound and CPU resources become underutilised. Where possible it is worth running Openwater with both input and output files on fast filesystems, such as SSDs.


