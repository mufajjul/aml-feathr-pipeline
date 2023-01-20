import argparse
import os

from datetime import timedelta
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

import feathr
from feathr import (
    FeathrClient,
    # Feature data types
    BOOLEAN, FLOAT, INT32, ValueType,
    # Feature data sources
    INPUT_CONTEXT, HdfsSource,
    # Feature aggregations
    TypedKey, WindowAggTransformation,
    # Feature types and anchor
    DerivedFeature, Feature, FeatureAnchor,
    # Materialization
    BackfillTime, MaterializationSettings, RedisSink,
    # Offline feature computation
    FeatureQuery, ObservationSettings,
)
from feathr.datasets import nyc_taxi
from feathr.spark_provider.feathr_configurations import SparkExecutionConfiguration
from feathr.utils.config import generate_config
from feathr.utils.job_utils import get_result_df
from feathr.utils.platform import is_databricks, is_jupyter

print(f"Feathr version: {feathr.__version__}")

parser = argparse.ArgumentParser()
parser.add_argument("--resource_prefix", type=str, help="Resource prefix used for all resource names (not necessarily resource group name).")
parser.add_argument("--resource_group", type=str, help="The name of the resource group.")
parser.add_argument("--azure_client_id", type=str)
parser.add_argument("--azure_tenant_id", type=str)
parser.add_argument("--azure_client_secret", type=str)
parser.add_argument("--azure_subscription_id", type=str)
args = parser.parse_args()

RESOURCE_PREFIX = args.resource_prefix
RESOURCE_GROUP = args.resource_group
os.environ['AZURE_CLIENT_ID'] = args.azure_client_id
os.environ['AZURE_TENANT_ID'] = args.azure_tenant_id
os.environ['AZURE_CLIENT_SECRET'] = args.azure_client_secret
os.environ['AZURE_SUBSCRIPTION_ID'] = args.azure_subscription_id
print("AZURE_TENANT_ID:", os.environ['AZURE_TENANT_ID'])
# RESOURCE_PREFIX = "rizodeploy11"  # TODO fill the value used to deploy the resources via ARM template
PROJECT_NAME = "feathr_getting_started"
LOCATION = "canadacentral"

# Currently support: 'azure_synapse', 'databricks', and 'local' 
SPARK_CLUSTER = "azure_synapse"

# TODO fill values to use databricks cluster:
DATABRICKS_CLUSTER_ID = None     # Set Databricks cluster id to use an existing cluster
DATABRICKS_URL = None   # Set Databricks workspace url to use databricks

# TODO fill values to use Azure Synapse cluster:
AZURE_SYNAPSE_SPARK_POOL = "spark31"  # Set Azure Synapse Spark pool name
AZURE_SYNAPSE_URL = f"https://web.azuresynapse.net?workspace=%2fsubscriptions%2f{os.environ['AZURE_SUBSCRIPTION_ID']}%2fresourceGroups%2f{RESOURCE_GROUP}%2fproviders%2fMicrosoft.Synapse%2fworkspaces%2f{RESOURCE_PREFIX}syws"  # Set Azure Synapse workspace url to use Azure Synapse

# Data store root path. Could be a local file system path, dbfs or Azure storage path like abfs or wasbs
DATA_STORE_PATH = TemporaryDirectory().name

# Feathr config file path to use an existing file
FEATHR_CONFIG_PATH = "./feathr_config.yaml"

# If set True, use an interactive browser authentication to get the redis password.
USE_CLI_AUTH = False

REGISTER_FEATURES = True

# (For the notebook test pipeline) If true, use ScrapBook package to collect the results.
SCRAP_RESULTS = False


# Create Feathr config file

feathr_config = f"""
# DO NOT MOVE OR DELETE THIS FILE

# This file contains the configurations that are used by Feathr
# All the configurations can be overwritten by environment variables with concatenation of `__` for different layers of this config file.
# For example, `feathr_runtime_location` for databricks can be overwritten by setting this environment variable:
# SPARK_CONFIG__DATABRICKS__FEATHR_RUNTIME_LOCATION
# Another example would be overwriting Redis host with this config: `ONLINE_STORE__REDIS__HOST`
# For example if you want to override this setting in a shell environment:
# export ONLINE_STORE__REDIS__HOST=feathrazure.redis.cache.windows.net

# version of API settings
api_version: 1
project_config:
  project_name: "{PROJECT_NAME}"
  # Information that are required to be set via environment variables.
  required_environment_variables:
    # the environemnt variables are required to run Feathr
    # Redis password for your online store
    - "REDIS_PASSWORD"
    # Client IDs and client Secret for the service principal. Read the getting started docs on how to get those information.
    - "AZURE_CLIENT_ID"
    - "AZURE_TENANT_ID"
    - "AZURE_CLIENT_SECRET"
  optional_environment_variables:
    # the environemnt variables are optional, however you will need them if you want to use some of the services:
    - ADLS_ACCOUNT
    - ADLS_KEY
    - BLOB_ACCOUNT
    - BLOB_KEY
    - S3_ACCESS_KEY
    - S3_SECRET_KEY
    - JDBC_TABLE
    - JDBC_USER
    - JDBC_PASSWORD
    - KAFKA_SASL_JAAS_CONFIG

offline_store:
  # paths starts with abfss:// or abfs://
  # ADLS_ACCOUNT and ADLS_KEY should be set in environment variable if this is set to true
  adls:
    adls_enabled: true

  # paths starts with wasb:// or wasbs://
  # BLOB_ACCOUNT and BLOB_KEY should be set in environment variable
  wasb:
    wasb_enabled: false

  # paths starts with s3a://
  # S3_ACCESS_KEY and S3_SECRET_KEY should be set in environment variable
  s3:
    s3_enabled: false
    # S3 endpoint. If you use S3 endpoint, then you need to provide access key and secret key in the environment variable as well.
    s3_endpoint: "s3.amazonaws.com"

  # snowflake endpoint
  # snowflake:
  #   url: "dqllago-ol19457.snowflakecomputing.com"
  #   user: "feathrintegration"
  #   role: "ACCOUNTADMIN"

  # jdbc endpoint
  # jdbc:
  #   jdbc_enabled: true
  #   jdbc_database: "feathrtestdb"
  #   jdbc_table: "feathrtesttable"


spark_config:
  # choice for spark runtime. Currently support: azure_synapse, databricks
  # The `databricks` configs will be ignored if `azure_synapse` is set and vice versa.
  spark_cluster: "azure_synapse"
  # configure number of parts for the spark output for feature generation job
  spark_result_output_parts: "1"

  azure_synapse:
    # dev URL to the synapse cluster. Usually it's `https://yourclustername.dev.azuresynapse.net`
    dev_url: "https://{RESOURCE_PREFIX}syws.dev.azuresynapse.net"
    # name of the sparkpool that you are going to use
    pool_name: "spark31"
    # workspace dir for storing all the required configuration files and the jar resources. All the feature definitions will be uploaded here
    workspace_dir: "abfss://{RESOURCE_PREFIX}fs@{RESOURCE_PREFIX}dls.dfs.core.windows.net/{PROJECT_NAME}"
    executor_size: "Small"
    executor_num: 1
    # This is the location of the runtime jar for Spark job submission. If you have compiled the runtime yourself, you need to specify this location.
    # Or use wasbs://public@azurefeathrstorage.blob.core.windows.net/feathr-assembly-LATEST.jar so you don't have to compile the runtime yourself
    # Local path, path starting with `http(s)://` or `wasbs://` are supported. If not specified, the latest jar from Maven would be used
    # feathr_runtime_location: "wasbs://public@azurefeathrstorage.blob.core.windows.net/feathr-assembly-LATEST.jar"

  databricks:
    # workspace instance
    workspace_instance_url: 'https://adb-6885802458123232.12.azuredatabricks.net/'
    # config string including run time information, spark version, machine size, etc.
    # the config follows the format in the databricks documentation: https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs#--request-structure-6
    # The fields marked as "FEATHR_FILL_IN" will be managed by Feathr. Other parameters can be customizable. For example, you can customize the node type, spark version, number of workers, instance pools, timeout, etc.
    config_template: '{{"run_name":"FEATHR_FILL_IN","new_cluster":{{"spark_version":"9.1.x-scala2.12","node_type_id":"Standard_D3_v2","num_workers":1,"spark_conf":{{"FEATHR_FILL_IN":"FEATHR_FILL_IN"}}}},"libraries":[{{"jar":"FEATHR_FILL_IN"}}],"spark_jar_task":{{"main_class_name":"FEATHR_FILL_IN","parameters":["FEATHR_FILL_IN"]}}}}'
    # workspace dir for storing all the required configuration files and the jar resources. All the feature definitions will be uploaded here
    work_dir: "dbfs:/{PROJECT_NAME}"
    # This is the location of the runtime jar for Spark job submission. If you have compiled the runtime yourself, you need to specify this location.
    # Or use https://azurefeathrstorage.blob.core.windows.net/public/feathr-assembly-LATEST.jar so you don't have to compile the runtime yourself
    # Local path, path starting with `http(s)://` or `dbfs://` are supported. If not specified, the latest jar from Maven would be used
    feathr_runtime_location: "https://azurefeathrstorage.blob.core.windows.net/public/feathr-assembly-LATEST.jar"

online_store:
  redis:
    # Redis configs to access Redis cluster
    host: "{RESOURCE_PREFIX}redis.redis.cache.windows.net"
    port: 6380
    ssl_enabled: True

feature_registry:
  api_endpoint: "https://{RESOURCE_PREFIX}webapp.azurewebsites.net/api/v1"
  # # Registry configs if use purview
  # purview:
  #   # configure the name of the purview endpoint
  #   purview_name: "{RESOURCE_PREFIX}purview"
  #   # delimiter indicates that how the project/workspace name, feature names etc. are delimited. By default it will be '__'
  #   # this is for global reference (mainly for feature sharing). For example, when we setup a project called foo, and we have an anchor called 'taxi_driver' and the feature name is called 'f_daily_trips'
  #   # the feature will have a globally unique name called 'foo__taxi_driver__f_daily_trips'
  #   delimiter: "__"
  #   # controls whether the type system will be initialized or not. Usually this is only required to be executed once.
  #   type_system_initialization: false


secrets:
  azure_key_vault:
    name: {RESOURCE_PREFIX}kv
"""

with open(FEATHR_CONFIG_PATH, "w") as file:
    file.write(feathr_config)

# Redis password
if 'REDIS_PASSWORD' not in os.environ:
    # Try to get all the required credentials from Azure Key Vault
    from azure.identity import AzureCliCredential, DefaultAzureCredential 
    from azure.keyvault.secrets import SecretClient

    vault_url = f"https://{RESOURCE_PREFIX}kv.vault.azure.net"
    if USE_CLI_AUTH:
        credential = AzureCliCredential(additionally_allowed_tenants=['*'],)
    else:
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=True,
            additionally_allowed_tenants=['*'],
        )
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    retrieved_secret = secret_client.get_secret('FEATHR-ONLINE-STORE-CONN').value
    os.environ['REDIS_PASSWORD'] = retrieved_secret.split(",")[1].split("password=", 1)[1]

if FEATHR_CONFIG_PATH:
    config_path = FEATHR_CONFIG_PATH
else:
    config_path = generate_config(
        resource_prefix=RESOURCE_PREFIX,
        project_name=PROJECT_NAME,
        spark_config__spark_cluster=SPARK_CLUSTER,
        spark_config__azure_synapse__dev_url=AZURE_SYNAPSE_URL,
        spark_config__azure_synapse__pool_name=AZURE_SYNAPSE_SPARK_POOL,
        spark_config__databricks__workspace_instance_url=DATABRICKS_URL,
        databricks_cluster_id=DATABRICKS_CLUSTER_ID,
    )


# Initialize the client

client = FeathrClient(config_path=config_path)

spark = (
        SparkSession
        .builder
        .appName("feathr")
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.3.0,io.delta:delta-core_2.12:2.1.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.ui.port", "8080")  # Set ui port other than the default one (4040) so that feathr spark job doesn't fail. 
        .getOrCreate()
    )


# Download the data and define features

DATA_FILE_PATH = str(Path(DATA_STORE_PATH, "nyc_taxi.csv"))

df_raw = nyc_taxi.get_spark_df(spark=spark, local_cache_path=DATA_FILE_PATH)
df_raw.limit(5).toPandas()

TIMESTAMP_COL = "lpep_dropoff_datetime"
TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"

# We define f_trip_distance and f_trip_time_duration features separately
# so that we can reuse them later for the derived features.
f_trip_distance = Feature(
    name="f_trip_distance",
    feature_type=FLOAT,
    transform="trip_distance",
)
f_trip_time_duration = Feature(
    name="f_trip_time_duration",
    feature_type=FLOAT,
    transform="cast_float((to_unix_timestamp(lpep_dropoff_datetime) - to_unix_timestamp(lpep_pickup_datetime)) / 60)",
)

features = [
    f_trip_distance,
    f_trip_time_duration,
    Feature(
        name="f_is_long_trip_distance",
        feature_type=BOOLEAN,
        transform="trip_distance > 30.0",
    ),
    Feature(
        name="f_day_of_week",
        feature_type=INT32,
        transform="dayofweek(lpep_dropoff_datetime)",
    ),
    Feature(
        name="f_day_of_month",
        feature_type=INT32,
        transform="dayofmonth(lpep_dropoff_datetime)",
    ),
    Feature(
        name="f_hour_of_day",
        feature_type=INT32,
        transform="hour(lpep_dropoff_datetime)",
    ),
]

# After you have defined features, bring them together to build the anchor to the source.
feature_anchor = FeatureAnchor(
    name="feature_anchor",
    source=INPUT_CONTEXT,  # Pass through source, i.e. observation data.
    features=features,
)

# Define data source path
if client.spark_runtime == "local" or (client.spark_runtime == "databricks" and is_databricks()):
    # In local mode, we can use the same data path as the source.
    # If the notebook is running on databricks, DATA_FILE_PATH should be already a dbfs path.
    data_source_path = DATA_FILE_PATH
else:
    # Otherwise, upload the local file to the cloud storage (either dbfs or adls).
    data_source_path = client.feathr_spark_launcher.upload_or_get_cloud_path(DATA_FILE_PATH)    


def preprocessing(df: DataFrame) -> DataFrame:
    import pyspark.sql.functions as F
    df = df.withColumn("fare_amount_cents", (F.col("fare_amount") * 100.0).cast("float"))
    return df

batch_source = HdfsSource(
    name="nycTaxiBatchSource",
    path=data_source_path,
    event_timestamp_column=TIMESTAMP_COL,
    preprocessing=preprocessing,
    timestamp_format=TIMESTAMP_FORMAT,
)

agg_key = TypedKey(
    key_column="DOLocationID",
    key_column_type=ValueType.INT32,
    description="location id in NYC",
    full_name="nyc_taxi.location_id",
)

agg_window = "90d"

# Anchored features with aggregations
agg_features = [
    Feature(
        name="f_location_avg_fare",
        key=agg_key,
        feature_type=FLOAT,
        transform=WindowAggTransformation(
            agg_expr="fare_amount_cents",
            agg_func="AVG",
            window=agg_window,
        ),
    ),
    Feature(
        name="f_location_max_fare",
        key=agg_key,
        feature_type=FLOAT,
        transform=WindowAggTransformation(
            agg_expr="fare_amount_cents",
            agg_func="MAX",
            window=agg_window,
        ),
    ),
]

agg_feature_anchor = FeatureAnchor(
    name="agg_feature_anchor",
    source=batch_source,  # External data source for feature. Typically a data table.
    features=agg_features,
)
derived_features = [
    DerivedFeature(
        name="f_trip_time_distance",
        feature_type=FLOAT,
        input_features=[
            f_trip_distance,
            f_trip_time_duration,
        ],
        transform="f_trip_distance / f_trip_time_duration",
    )
]

feature_names = [feature.name for feature in features + agg_features + derived_features]



client.build_features(
    anchor_list=[feature_anchor, agg_feature_anchor],
    derived_feature_list=derived_features,
)


# Create training data using point-in-time correct feature join

DATA_FORMAT = "parquet"
offline_features_path = str(Path(DATA_STORE_PATH, "feathr_output", f"features.{DATA_FORMAT}"))

# Features that we want to request. Can use a subset of features
query = FeatureQuery(
    feature_list=feature_names,
    key=agg_key,
)
settings = ObservationSettings(
    observation_path=data_source_path,
    event_timestamp_column=TIMESTAMP_COL,
    timestamp_format=TIMESTAMP_FORMAT,
)
client.get_offline_features(
    observation_settings=settings,
    feature_query=query,
    # For more details, see https://feathr-ai.github.io/feathr/how-to-guides/feathr-job-configuration.html
    execution_configurations=SparkExecutionConfiguration({
        "spark.feathr.outputFormat": DATA_FORMAT,
    }),
    output_path=offline_features_path,
)

print("DATA_STORE_PATH:", DATA_STORE_PATH)

client.wait_job_to_finish(timeout_sec=1000)


df = get_result_df(
    spark=spark,
    client=client,
    data_format=DATA_FORMAT,
    res_url=offline_features_path,
)


# Register features

if REGISTER_FEATURES:
    try:
        client.register_features()
    except KeyError:
        # TODO temporarily go around the "Already exists" error
        pass    
    print(client.list_registered_features(project_name=PROJECT_NAME))
    # You can get the actual features too by calling client.get_features_from_registry(PROJECT_NAME)


# Materialize features

# Get the last date from the dataset
backfill_timestamp = (
    df_raw
    .select(F.to_timestamp(F.col(TIMESTAMP_COL), TIMESTAMP_FORMAT).alias(TIMESTAMP_COL))
    .agg({TIMESTAMP_COL: "max"})
    .collect()[0][0]
)

FEATURE_TABLE_NAME = "nycTaxiDemoFeature"

# Time range to materialize
backfill_time = BackfillTime(
    start=backfill_timestamp,
    end=backfill_timestamp,
    step=timedelta(days=1),
)

# Destinations:
# For online store,
redis_sink = RedisSink(table_name=FEATURE_TABLE_NAME)

# For offline store,
# adls_sink = HdfsSink(output_path=)

settings = MaterializationSettings(
    name=FEATURE_TABLE_NAME + ".job",  # job name
    backfill_time=backfill_time,
    sinks=[redis_sink],  # or adls_sink
    feature_names=[feature.name for feature in agg_features],
)

client.materialize_features(
    settings=settings,
    execution_configurations={"spark.feathr.outputFormat": "parquet"},
)

client.wait_job_to_finish(timeout_sec=5000)


# Note, to get a single key, you may use client.get_online_features instead
materialized_feature_values = client.multi_get_online_features(
    feature_table=FEATURE_TABLE_NAME,
    keys=["239", "265"],
    feature_names=[feature.name for feature in agg_features],
)
materialized_feature_values


