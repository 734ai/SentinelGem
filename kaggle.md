@https://www.kaggle.com/docs/api = ./kaggle.json
//How to use Kaggle
Public API
Create Datasets, Notebooks, and connect with Kaggle
Getting Started: Installation & Authentication
The easiest way to interact with Kaggle’s public API is via our command-line tool (CLI) implemented in Python. This section covers installation of the kaggle package and authentication.

Installation
Ensure you have Python and the package manager pip installed. Run the following command to access the Kaggle API using the command line: pip install kaggle (You may need to do pip install --user kaggle on Mac/Linux. This is recommended if problems come up during the installation process.) Follow the authentication steps below and you’ll be able to use the kaggle CLI tool.

If you run into a kaggle: command not found error, ensure that your python binaries are on your path. You can see where kaggle is installed by doing pip uninstall kaggle and seeing where the binary is. For a local user install on Linux, the default location is ~/.local/bin. On Windows, the default location is $PYTHON_HOME/Scripts.

Authentication
In order to use the Kaggle’s public API, you must first authenticate using an API token. Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.

If you are using the Kaggle CLI tool, the tool will look for this token at ~/.kaggle/kaggle.json on Linux, OSX, and other UNIX-based operating systems, and at C:\Users\<Windows-username>\.kaggle\kaggle.json on Windows. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.

If you are using the Kaggle API directly, where you keep the token doesn’t matter, so long as you are able to provide your credentials at runtime.

Interacting with Competitions
The Kaggle API and CLI tool provide easy ways to interact with Competitions on Kaggle. The commands available can make participating in competitions a seamless part of your model building workflow.

If you haven’t installed the package needed to use the command line tool or generated an API token, check out the getting started steps first.

Just like participating in a Competition normally through the user interface, you must read and accept the rules in order to download data or make submissions. You cannot accept Competition rules via the API. You must do this by visiting the Kaggle website and accepting the rules there.

Some of the commands for interacting with Competitions via CLI include:

kaggle competitions list: list the currently active competitions

kaggle competitions download -c [COMPETITION]: download files associated with a competition

kaggle competitions submit -c [COMPETITION] -f [FILE] -m [MESSAGE]: make a competition submission

View all available commands on the official documentation on GitHub and keep up-to-date with the latest features and bug fixes in the changelog.

To explore additional CLI arguments, remember that you can always append -h after any call to see the help menu for that command.

Submitting to a Competition
Assuming that you have already accepted the terms of a Competition (this can only be done through the website, and not through the CLI), you may use the Kaggle CLI to submit predictions to the Competition and have them scored. To do so, run the command kaggle competitions submit -c [COMPETITION NAME] -f [FILE PATH].

You can list all previous submission to a Competition you have entered using the command kaggle competitions submissions -c [COMPETITION NAME].

To explore some further CLI arguments, remember that you can always append -h after any call to see the help menu for that command.

Interacting with Datasets
The Kaggle API and CLI tool provide easy ways to interact with Datasets on Kaggle. The commands available can make searching for and downloading Kaggle Datasets a seamless part of your data science workflow.

If you haven’t installed the Kaggle Python package needed to use the command line tool or generated an API token, check out the getting started steps first.

Some of the commands for interacting with Datasets via CLI include:

kaggle datasets list -s [KEYWORD]: list datasets matching a search term

kaggle datasets download -d [DATASET]: download files associated with a dataset

If you are creating or updating a dataset on Kaggle, you can also use the API to make maintenance convenient or even programmatic. View all available commands on the official documentation on GitHub and keep up-to-date with the latest features and bug fixes in the changelog.

To explore additional CLI arguments, remember that you can always append -h after any call to see the help menu for that command.

Other than the Kaggle API, there is also a Kaggle connector on DataStudio! You can select Kaggle Datasets as a data source to import directly into DataStudio. Work in DataStudio to easily create beautiful and effective dashboards on Kaggle Datasets!

Creating and Maintaining Datasets
The Kaggle API can be used to to create new Datasets and Dataset versions on Kaggle from the comfort of the command-line. This can make sharing data and projects on Kaggle a simple part of your workflow. You can even use the API plus a tool like crontab to schedule programmatic updates of your Datasets to keep them well maintained.

If you haven’t installed the Kaggle Python package needed to use the command line tool or generated an API token, check out the getting started steps first.

Create a New Dataset
Here are the steps you can follow to create a new dataset on Kaggle:

Create a folder containing the files you want to upload

Run kaggle datasets init -p /path/to/dataset to generate a metadata file

Add your dataset’s metadata to the generated file, datapackage.json

Run kaggle datasets create -p /path/to/dataset to create the dataset

Your dataset will be private by default. You can also add a -u flag to make it public when you create it, or navigate to “Settings” > “Sharing” from your dataset’s page to make it public or share with collaborators.

Create a New Dataset Version
If you’d like to upload a new version of an existing dataset, follow these steps:

Run kaggle datasets init -p /path/to/dataset to generate a metadata file (if you don’t already have one)

Make sure the id field in dataset-metadata.json (or datapackage.json) points to your dataset

Run kaggle datasets version -p /path/to/dataset -m "Your message here"

These instructions are the basic commands required to get started with creating and updating Datasets on Kaggle. You can find out more details from the official documentation on GitHub:

Initializing metadata

Create a Dataset

Update a Dataset

Working with Dataset Metadata
If you want a faster way to complete the required dataset-metadata.json file (for example, if you want to add column-level descriptions for many tabular data files), we recommend using Frictionless Data’s Data Package Creator. Simply upload the dataset-metadata.json file that you’ve initialized for your dataset, fill out metadata in the user interface, and download the result.

To explore some further CLI arguments, remember that you can always append -h after any call to see the help menu for that command.

Interacting with Notebooks
The Kaggle API and CLI tool provide easy ways to interact with Notebooks on Kaggle. The commands available enable both searching for and downloading published Notebooks and their metadata as well as workflows for creating and running Notebooks using computational resources on Kaggle.

If you haven’t installed the Kaggle Python package needed to use the command line tool or generated an API token, check out the getting started steps first.

Some of the commands for interacting with Notebooks via CLI include:

kaggle kernels list -s [KEYWORD]: list Notebooks matching a search term

kaggle kernels push -k [KERNEL] -p /path/to/folder : create and run a Notebook on Kaggle

kaggle kernels pull [KERNEL] -p /path/to/download -m: download code files and metadata associated with a Notebook

If you are creating a new Notebook or running a new version of an existing Notebook on Kaggle, you can also use the API to make this workflow convenient or even programmatic. View all available commands on the official documentation on GitHub and keep up-to-date with the latest features and bug fixes in the changelog.

To explore additional CLI arguments, remember that you can always append -h after any call to see the help menu for that command.

Creating and Running a New Notebook
The Kaggle API can be used to to create new Notebooks and Notebook versions on Kaggle from the comfort of the command-line. This can make executing and sharing code on Kaggle a simple part of your workflow.

If you haven’t installed the Kaggle Python package needed to use the command line tool or generated an API token, check out the getting started steps first.

Here are the steps you can follow to create and run a new Notebook on Kaggle:

Create a local folder containing the code files you want to upload (e.g., your Python or R notebooks, scripts, or RMarkdown files)

Run kaggle kernels init -p /path/to/folder to generate a metadata file

Add your Notebook's metadata to the generated file, kernel-metadata.json; As you add your title and slug, please be aware that Notebook titles and slugs are linked to each other. A Notebook slug is always the title lowercased with dashes (-) replacing spaces and removing special characters.

Run kaggle kernels push -p /path/to/folder to create and run the Notebook on Kaggle

Your Notebook will be private by default unless you set it to public in the metadata file. You can also navigate to "Options" > “Sharing” from your published Notebook's page to make it public or share with collaborators.

Creating and Running a New Notebook Version
If you’d like to create and run a new version of an existing Notebook, follow these steps:

Run kaggle kernels pull [KERNEL] -p /path/to/download -m to download your Notebook's most recent code and metadata files (if you your local copies aren't current)

Make sure the id field in kernel-metadata.json points to your Notebook; you no longer need to include the title field which is optional for Notebook versions unless you want to rename your Notebook (make sure to update the id field in your next push AFTER the rename is complete)

Run kaggle kernels push -p /path/to/folder

These instructions are the basic commands required to get started with creating, running, and updating Notebooks on Kaggle. You can find out more details from the official documentation on GitHub:

Initializing metadata

Push a Notebook

Pull a Notebook

Retrieve a Notebook's output

Using Models in Notebooks
Models can be downloaded via notebooks using the following code:

    import kagglehub
    
    # Authenticate
    kagglehub.login() # This will prompt you for your credentials.
    # We also offer other ways to authenticate (credential file & env variables): https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate
    
    # Download latest version
    path = kagglehub.model_download("google/gemma/pyTorch/2b")
    
    # Download specific version (here version 1)
    path = kagglehub.model_download("google/gemma/pyTorch/2b/1")
    
    print("Path to model files:", path)
    
Models can be uploaded via notebooks using the following code:

    import kagglehub
    from kagglehub.config import get_kaggle_credentials
    
    # Other ways to authenticate also available: https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate
    kagglehub.login() 
    
    username, _ = get_kaggle_credentials()
    
    # For PyTorch framework & `2b` variation.
    # Replace the framework with "jax", "other" based on which framework you are uploading to.
    kagglehub.model_upload(f'{username}/my_model/pyTorch/2b', 'path/to/local/model/files', 'Apache 2.0')
    
    # Run the same command again to upload a new version for an existing variation.
    


@https://github.com/Kaggle/kagglehub
kagglehub
The kagglehub library provides a simple way to interact with Kaggle resources such as datasets, models, notebook outputs in Python.

This library also integrates natively with the Kaggle notebook environment. This means the behavior differs when you download a Kaggle resource with kagglehub in the Kaggle notebook environment:

In a Kaggle notebook:
The resource is automatically attached to your Kaggle notebook.
The resource will be shown under the "Input" panel in the Kaggle notebook editor.
The resource files are served from the shared Kaggle resources cache (not using the VM's disk).
Outside a Kaggle notebook:
The resource files are downloaded to a local cache folder.
Installation
Install the kagglehub package with pip:

pip install kagglehub
Usage
Authenticate
Note

kagglehub is authenticated by default when running in a Kaggle notebook.

Authenticating is only needed to access public resources requiring user consent or private resources.

First, you will need a Kaggle account. You can sign up here.

After login, you can download your Kaggle API credentials at https://www.kaggle.com/settings by clicking on the "Create New Token" button under the "API" section.

You have four different options to authenticate. Note that if you use kaggle-api (the kaggle command-line tool) you have already done Option 3 and can skip this.

Option 1: Calling kagglehub.login()
This will prompt you to enter your username and token:

import kagglehub

kagglehub.login()
Option 2: Read credentials from environment variables
You can also choose to export your Kaggle username and token to the environment:

export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
Option 3: Read credentials from kaggle.json
Store your kaggle.json credentials file at ~/.kaggle/kaggle.json.

Alternatively, you can set the KAGGLE_CONFIG_DIR environment variable to change this location to $KAGGLE_CONFIG_DIR/kaggle.json.

Note for Windows users: The default directory is %HOMEPATH%/kaggle.json.

Option 4: Read credentials from Google Colab secrets
Store your username and key token as Colab secrets KAGGLE_USERNAME and KAGGLE_KEY.

Instructions on adding secrets in both Colab and Colab Enterprise can be found in this article.

Download Model
The following examples download the answer-equivalence-bem variation of this Kaggle model: https://www.kaggle.com/models/google/bert/tensorFlow2/answer-equivalence-bem

import kagglehub

# Download the latest version.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem')

# Download a specific version.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem/1')

# Download a single file.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem', path='variables/variables.index')

# Download a model or file, even if previously downloaded to cache.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem', force_download=True)
Upload Model
Uploads a new variation (or a new variation's version if it already exists).

import kagglehub

# For example, to upload a new variation to this model:
# - https://www.kaggle.com/models/google/bert/tensorFlow2/answer-equivalence-bem
# 
# You would use the following handle: `google/bert/tensorFlow2/answer-equivalence-bem`
handle = '<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>'
local_model_dir = 'path/to/local/model/dir'

kagglehub.model_upload(handle, local_model_dir)

# You can also specify some version notes (optional)
kagglehub.model_upload(handle, local_model_dir, version_notes='improved accuracy')

# You can also specify a license (optional)
kagglehub.model_upload(handle, local_model_dir, license_name='Apache 2.0')

# You can also specify a list of patterns for files/dirs to ignore.
# These patterns are combined with `kagglehub.models.DEFAULT_IGNORE_PATTERNS` 
# to determine which files and directories to exclude. 
# To ignore entire directories, include a trailing slash (/) in the pattern.
kagglehub.model_upload(handle, local_model_dir, ignore_patterns=["original/", "*.tmp"])
Load Dataset
Loads a file from a Kaggle Dataset into a python object based on the selected KaggleDatasetAdapter:

KaggleDatasetAdapter.PANDAS → pandas DataFrame (or multiple given certain files/settings)
KaggleDatasetAdapter.HUGGING_FACE→ Hugging Face Dataset
KaggleDatasetAdapter.POLARS → polars LazyFrame or DataFrame (or multiple given certain files/settings)
NOTE: To use these adapters, you must install the optional dependencies (or already have them available in your environment)

KaggleDatasetAdapter.PANDAS → pip install kagglehub[pandas-datasets]
KaggleDatasetAdapter.HUGGING_FACE→ pip install kagglehub[hf-datasets]
KaggleDatasetAdapter.POLARS→ pip install kagglehub[polars-datasets]
KaggleDatasetAdapter.PANDAS
This adapter supports the following file types, which map to a corresponding pandas.read_* method:

File Extension	pandas Method
.csv, .tsv1	pandas.read_csv
.json, .jsonl2	pandas.read_json
.xml	pandas.read_xml
.parquet	pandas.read_parquet
.feather	pandas.read_feather
.sqlite, .sqlite3, .db, .db3, .s3db, .dl33	pandas.read_sql_query
.xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt4	pandas.read_excel
dataset_load also supports pandas_kwargs which will be passed as keyword arguments to the pandas.read_* method. Some examples include:

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load a DataFrame with a specific version of a CSV
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "unsdsn/world-happiness/versions/1",
    "2016.csv",
)

# Load a DataFrame with specific columns from a parquet file
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet",
    pandas_kwargs={"columns": ["image_id", "bbox", "points", "area"]}
)

# Load a dictionary of DataFrames from an Excel file where the keys are sheet names 
# and the values are DataFrames for each sheet's data. NOTE: As written, this requires 
# installing the default openpyxl engine.
df_dict = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "theworldbank/education-statistics",
    "edstats-excel-zip-72-mb-/EdStatsEXCEL.xlsx",
    pandas_kwargs={"sheet_name": None},
)

# Load a DataFrame using an XML file (with the natively available etree parser)
df = dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "parulpandey/covid19-clinical-trials-dataset",
    "COVID-19 CLinical trials studies/COVID-19 CLinical trials studies/NCT00571389.xml",
    pandas_kwargs={"parser": "etree"},
)

# Load a DataFrame by executing a SQL query against a SQLite DB
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)
KaggleDatasetAdapter.HUGGING_FACE
The Hugging Face Dataset provided by this adapater is built exclusively using Dataset.from_pandas. As a result, all of the file type and pandas_kwargs support is the same as KaggleDatasetAdapter.PANDAS. Some important things to note about this:

Because Dataset.from_pandas cannot accept a collection of DataFrames, any attempts to load a file with pandas_kwargs that produce a collection of DataFrames will result in a raised exception
hf_kwargs may be provided, which will be passed as keyword arguments to Dataset.from_pandas
Because the use of pandas is transparent when pandas_kwargs are not needed, we default to False for preserve_index—this can be overridden using hf_kwargs
Some examples include:

import kagglehub
from kagglehub import KaggleDatasetAdapter
# Load a Dataset with a specific version of a CSV, then remove a column
dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "unsdsn/world-happiness/versions/1",
    "2016.csv",
)
dataset = dataset.remove_columns('Region')

# Load a Dataset with specific columns from a parquet file, then split into test/train splits
dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet",
    pandas_kwargs={"columns": ["image_id", "bbox", "points", "area"]}
)
dataset_with_splits = dataset.train_test_split(test_size=0.8, train_size=0.2)

# Load a Dataset by executing a SQL query against a SQLite DB, then rename a column
dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)
dataset = dataset.rename_column('season', 'year')
KaggleDatasetAdapter.POLARS
This adapter supports the following file types, which map to a corresponding polars.scan_* or polars.read_* method:

File Extension	polars Method
.csv, .tsv1	polars.scan_csv or polars.read_csv
.json	polars.read_json
.jsonl	polars.scan_ndjson or polars.read_ndjson
.parquet	polars.scan_parquet or polars.read_parquet
.feather	polars.scan_ipc or polars.read_ipc
.sqlite, .sqlite3, .db, .db3, .s3db, .dl32	polars.read_database
.xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt3	polars.read_excel
dataset_load also supports polars_kwargs which will be passed as keyword arguments to the polars.scan_* or polars_read_* method.

LazyFrame vs DataFrame
Per polars documentation, LazyFrame "allows for whole-query optimisation in addition to parallelism, and is the preferred (and highest-performance) mode of operation for polars." As such, scan_* methods are used by default whenever possible--and when not possible the result of the read_* method is returned after calling .lazy(). If a DataFrame is preferred, dataset_load supports an optional polars_frame_type and PolarsFrameType.DATA_FRAME may be passed in. This will force a read_* method to be used with no .lazy() call. NOTE: For file types that support scan_*, changing the polars_frame_type may affect which polars_kwargs are acceptable to the underlying method since it will force a read_* method to be used rather than a scan_* method.

Some examples include:

import kagglehub
from kagglehub import KaggleDatasetAdapter, PolarsFrameType

# Load a LazyFrame with a specific version of a CSV
lf = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "unsdsn/world-happiness/versions/1",
    "2016.csv",
)

# Load a LazyFramefrom a parquet file, then select specific columns
lf = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet",
)
lf.select(["image_id", "bbox", "points", "area"]).collect()

# Load a DataFrame with specific columns from a parquet file
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet",
    polars_frame_type=PolarsFrameType.DATA_FRAME,
    polars_kwargs={"columns": ["image_id", "bbox", "points", "area"]}
)

# Load a dictionary of LazyFrames from an Excel file where the keys are sheet names 
# and the values are LazyFrames for each sheet's data. NOTE: As written, this requires 
# installing the default fastexcel engine.
lf_dict = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "theworldbank/education-statistics",
    "edstats-excel-zip-72-mb-/EdStatsEXCEL.xlsx",
    # sheet_id of 0 returns all sheets
    polars_kwargs={"sheet_id": 0},
)

# Load a LazyFrame by executing a SQL query against a SQLite DB
lf = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)
Download Dataset
The following examples download the Spotify Recommendation Kaggle dataset: https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation

import kagglehub

# Download the latest version.
kagglehub.dataset_download('bricevergnou/spotify-recommendation')

# Download a specific version.
kagglehub.dataset_download('bricevergnou/spotify-recommendation/versions/1')

# Download a single file.
kagglehub.dataset_download('bricevergnou/spotify-recommendation', path='data.csv')

# Download a dataset or file, even if previously downloaded to cache.
kagglehub.dataset_download('bricevergnou/spotify-recommendation', force_download=True)
Upload Dataset
Uploads a new dataset (or a new version if it already exists).

import kagglehub

# For example, to upload a new dataset (or version) at:
# - https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation
# 
# You would use the following handle: `bricevergnou/spotify-recommendation`
handle = '<KAGGLE_USERNAME>/<DATASET>'
local_dataset_dir = 'path/to/local/dataset/dir'

# Create a new dataset
kagglehub.dataset_upload(handle, local_dataset_dir)

# You can then create a new version of this existing dataset and include version notes (optional).
kagglehub.dataset_upload(handle, local_dataset_dir, version_notes='improved data')

# You can also specify a list of patterns for files/dirs to ignore.
# These patterns are combined with `kagglehub.datasets.DEFAULT_IGNORE_PATTERNS` 
# to determine which files and directories to exclude. 
# To ignore entire directories, include a trailing slash (/) in the pattern.
kagglehub.dataset_upload(handle, local_dataset_dir, ignore_patterns=["original/", "*.tmp"])
Download Competition
The following examples download the Digit Recognizer Kaggle competition: https://www.kaggle.com/competitions/digit-recognizer

import kagglehub

# Download the latest version.
kagglehub.competition_download('digit-recognizer')

# Download a single file.
kagglehub.competition_download('digit-recognizer', path='train.csv')

# Download a competition or file, even if previously downloaded to cache. 
kagglehub.competition_download('digit-recognizer', force_download=True)
Download Notebook Outputs
The following examples download the Titanic Tutorial notebook output: https://www.kaggle.com/code/alexisbcook/titanic-tutorial

import kagglehub

# Download the latest version.
kagglehub.notebook_output_download('alexisbcook/titanic-tutorial')

# Download a specific version of the notebook output.
kagglehub.notebook_output_download('alexisbcook/titanic-tutorial/versions/1')

# Download a single file.
kagglehub.notebok_output_download('alexisbcook/titanic-tutorial', path='submission.csv')
Install Utility Script
The following example installs the utility script Physionet Challenge Utility Script Utility Script: https://www.kaggle.com/code/bjoernjostein/physionet-challenge-utility-script. Using this command allows the code from this script to be available in your python environment.

import kagglehub

# Install the latest version.
kagglehub.utility_script_install('bjoernjostein/physionet-challenge-utility-script')
Options
Change the default cache folder
By default, kagglehub downloads files to your home folder at ~/.cache/kagglehub/.

You can override this path by setting the KAGGLEHUB_CACHE environment variable.

Development
Prequisites
We use hatch to manage this project.

Follow these instructions to install it.

Tests
# Run all tests for current Python version.
hatch test

# Run all tests for all Python versions.
hatch test --all

# Run all tests for a specific Python version.
hatch test -py 3.11

# Run a single test file
hatch test tests/test_<SOME_FILE>.py
Integration Tests
To run integration tests on your local machine, you need to set up your Kaggle API credentials. You can do this in one of these two ways described in the earlier sections of this document. Refer to the sections:

Using environment variables
Using credentials file
After setting up your credentials by any of these methods, you can run the integration tests as follows:

# Run all tests
hatch test integration_tests
Run kagglehub from source
Option 1: Execute a one-liner of code from the command line
# Download a model & print the path
hatch run python -c "import kagglehub; print('path: ', kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem'))"
Option 2: Run a saved script from the /tools/scripts directory
# This runs the same code as the one-liner above, but reads it from a 
# checked in script located at tool/scripts/download_model.py
hatch run python tools/scripts/download_model.py
Option 3: Run a temporary script from the root of the repo
Any script created at the root of the repo is gitignore'd, so they're just temporary scripts for testing in development. Placing temporary scripts at the root makes the run command easier to use during local development.

# Test out some new changes
hatch run python test_new_feature.py
Lint / Format
# Lint check
hatch run lint:style
hatch run lint:typing
hatch run lint:all     # for both

# Format
hatch run lint:fmt
Coverage report
hatch test --cover
Build
hatch build
Running hatch commands inside Docker
This is useful to run in a consistent environment and easily switch between Python versions.

The following shows how to run hatch run lint:all but this also works for any other hatch commands:

# Use default Python version
./docker-hatch run lint:all

# Use specific Python version (Must be a valid tag from: https://hub.docker.com/_/python)
./docker-hatch -v 3.9 run lint:all

# Run test in docker with specific Python version
./docker-hatch -v 3.9 test

# Run python from specific environment (e.g. one with optional dependencies installed)
./docker-hatch run extra-deps-env:python -c "print('hello world')"

# Run commands with other root-level hatch options (everything after -- gets passed to hatch)
./docker-hatch -v 3.9 -- -v env create debug-env-with-verbose-logging
VS Code setup
Prerequisites
Install the recommended extensions.

Instructions
Configure hatch to create virtual env in project folder.

hatch config set dirs.env.virtual .env
After, create all the python environments needed by running hatch test --all.

Finally, configure vscode to use one of the selected environments: cmd + shift + p -> python: Select Interpreter -> Pick one of the folders in ./.env






Of course. Here is a revised and improved version of the provided documentation in Markdown format.

Generated markdown
# Kaggle Public API and `kagglehub` Library Guide

This guide provides a comprehensive overview of how to interact with Kaggle's resources using the public API and the `kagglehub` Python library. You can manage competitions, datasets, models, and notebooks programmatically.

## Getting Started: The Kaggle Command-Line Tool (CLI)

The easiest way to begin is with the Kaggle CLI, which is a Python-based tool.

### Installation

1.  **Prerequisites**: Make sure you have Python and the `pip` package manager installed on your system.
2.  **Install the Kaggle package**:
    ```bash
    pip install kaggle
    ```
    *   **Note for Mac/Linux users**: If you encounter permission issues, you might need to install it for the current user only:
        ```bash
        pip install --user kaggle
        ```
3.  **Troubleshooting `command not found` errors**: If you see a `kaggle: command not found` error, you need to add the directory containing the `kaggle` executable to your system's PATH.
    *   Find the installation path by running `pip uninstall kaggle` and noting the location of the files.
    *   Default locations:
        *   **Linux (user install)**: `~/.local/bin`
        *   **Windows**: `C:\Users\<Your-Username>\AppData\Local\Programs\Python\PythonXX\Scripts`

### Authentication

To use the Kaggle API, you must authenticate with an API token.

1.  **Create an API Token**:
    *   Go to your Kaggle account settings page: [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
    *   In the "API" section, click on **Create New Token**.
    *   This will download a file named `kaggle.json` containing your API credentials.

2.  **Place the `kaggle.json` file**:
    *   The Kaggle CLI expects this file to be in a specific location to work automatically.
    *   **Linux / macOS / other Unix-like systems**: Move the file to `~/.kaggle/kaggle.json`.
    *   **Windows**: Move the file to `C:\Users\<Windows-username>\.kaggle\kaggle.json`.

    If you are using the Kaggle API directly in your code without the CLI, you can place this file anywhere, as long as you can provide the credentials at runtime.

---

## Interacting with Kaggle Resources via the CLI

### Competitions

You can manage Kaggle competitions directly from your terminal.

*   **Important**: You must accept the rules of a competition on the Kaggle website before you can download data or submit results via the API.

**Common Commands**:
*   **List current competitions**:
    ```bash
    kaggle competitions list
    ```
*   **Download competition files**:
    ```bash
    kaggle competitions download -c [COMPETITION_NAME]
    ```
*   **Submit to a competition**:
    ```bash
    kaggle competitions submit -c [COMPETITION_NAME] -f [FILE_PATH] -m "Your submission message"
    ```
*   **View your past submissions**:
    ```bash
    kaggle competitions submissions -c [COMPETITION_NAME]
    ```

> **Tip**: Append `-h` to any command to see all available options and arguments.

### Datasets

Seamlessly search for and download datasets.

**Common Commands**:
*   **Search for datasets**:
    ```bash
    kaggle datasets list -s [SEARCH_KEYWORD]
    ```
*   **Download a dataset**:
    ```bash
    kaggle datasets download -d [OWNER_SLUG]/[DATASET_SLUG]
    ```

### Notebooks (Kernels)

Manage and run your Kaggle Notebooks from the command line.

**Common Commands**:
*   **Search for notebooks**:
    ```bash
    kaggle kernels list -s [SEARCH_KEYWORD]
    ```
*   **Push a local notebook to Kaggle to run**:
    ```bash
    kaggle kernels push -p /path/to/your/notebook/folder
    ```
*   **Download notebook files and metadata**:
    ```bash
    kaggle kernels pull [USERNAME]/[NOTEBOOK_SLUG] -p /path/to/download -m
    ```

---

## The `kagglehub` Python Library

The `kagglehub` library offers a high-level interface for interacting with Kaggle resources directly within your Python scripts and notebooks.

### Installation

```bash
pip install kagglehub

Authentication for kagglehub

Authentication is automatic when using kagglehub inside a Kaggle Notebook. For local development or other environments, you have four options:

Interactive Login: Run this in your script to get a prompt for your credentials.

Generated python
import kagglehub
kagglehub.login()
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

Environment Variables: Set the following environment variables.

Generated bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

kaggle.json File: This is the same file used by the CLI. Place it at ~/.kaggle/kaggle.json.

Google Colab Secrets: Store your KAGGLE_USERNAME and KAGGLE_KEY as secrets in your Colab notebook.

Using kagglehub
Models

Download a Model:

Generated python
import kagglehub

# Download the latest version of a model
model_path = kagglehub.model_download("google/bert/tensorFlow2/answer-equivalence-bem")

# Download a specific version
model_path_v1 = kagglehub.model_download("google/bert/tensorFlow2/answer-equivalence-bem/1")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

Upload a Model:

Generated python
import kagglehub

handle = "your_username/your_model/tensorflow/my_variation"
local_model_directory = "/path/to/your/model/files"

kagglehub.model_upload(handle, local_model_directory, license_name="Apache 2.0")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
Datasets

Download a Dataset:```python
import kagglehub

Download the latest version of a dataset

dataset_path = kagglehub.dataset_download("bricevergnou/spotify-recommendation")

Download a single file from the dataset

file_path = kagglehub.dataset_download("bricevergnou/spotify-recommendation", path="data.csv")

Generated code
**Load Datasets with Adapters**:
The library can directly load data into popular data formats. You may need to install optional dependencies:
*   **pandas**: `pip install kagglehub[pandas-datasets]`
*   **Hugging Face**: `pip install kagglehub[hf-datasets]`
*   **Polars**: `pip install kagglehub[polars-datasets]`

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load a CSV into a pandas DataFrame
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "unsdsn/world-happiness/versions/1",
    "2016.csv"
)

# Load a Parquet file into a Hugging Face Dataset
hf_dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet"
)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Competitions

Download Competition Files:

Generated python
import kagglehub

# Download all files for a competition
competition_path = kagglehub.competition_download("digit-recognizer")

# Download a specific file
train_csv_path = kagglehub.competition_download("digit-recognizer", path="train.csv")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
Notebooks

Download Notebook Outputs:

Generated python
import kagglehub

# Download the output from the latest version of a notebook
output_path = kagglehub.notebook_output_download("alexisbcook/titanic-tutorial")

# Download a specific output file
submission_path = kagglehub.notebook_output_download("alexisbcook/titanic-tutorial", path="submission.csv")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
Advanced Options

Changing the Cache Folder: By default, kagglehub caches files in ~/.cache/kagglehub/. You can change this by setting the KAGGLEHUB_CACHE environment variable.

Forcing Downloads: To re-download files instead of using the cache, add the force_download=True argument to any download function.

Generated code
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END