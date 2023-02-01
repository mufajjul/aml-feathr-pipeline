from pyspark.sql import DataFrame
from feathr import INPUT_CONTEXT, HdfsSource
from feathr import BOOLEAN, FLOAT, INT32, ValueType
from feathr import Feature, DerivedFeature, FeatureAnchor
from feathr import WindowAggTransformation
from feathr import TypedKey
from pyspark.sql.functions import col
import utils

feathr_client = utils.get_feathr_client()


def feathr_udf_day_calc(df: DataFrame) -> DataFrame:
    df = df.withColumn("fare_amount_cents", col("fare_amount")*100)
    return df


# define the data source
batch_source = HdfsSource(name="nycTaxiBatchSource",
                          path="abfss://{container_name}@{storage_account}.dfs.core.windows.net/nyc_taxi.parquet".format(
                              storage_account=utils.fs_config.get("adls_account"), container_name=utils.fs_config.get("data_container_name")),
                          event_timestamp_column="lpep_dropoff_datetime",
                          preprocessing=feathr_udf_day_calc,
                          timestamp_format="yyyy-MM-dd HH:mm:ss")


# define anchor features
f_trip_distance = Feature(name="f_trip_distance",
                          feature_type=FLOAT, transform="trip_distance")
f_trip_time_duration = Feature(name="f_trip_time_duration",
                               feature_type=INT32,
                               transform="(to_unix_timestamp(lpep_dropoff_datetime) - to_unix_timestamp(lpep_pickup_datetime))/60")

features = [
    f_trip_distance,
    f_trip_time_duration,
    Feature(name="f_is_long_trip_distance",
            feature_type=BOOLEAN,
            transform="cast_float(trip_distance)>30"),
    Feature(name="f_day_of_week",
            feature_type=INT32,
            transform="dayofweek(lpep_dropoff_datetime)"),
]

request_anchor = FeatureAnchor(name="request_features",
                               source=INPUT_CONTEXT,
                               features=features)


# window aggregation features
location_id = TypedKey(key_column="DOLocationID",
                       key_column_type=ValueType.INT32,
                       description="location id in NYC",
                       full_name="nyc_taxi.location_id")

# calculate the average trip fare, maximum fare and total fare per location for 90 days
agg_features = [Feature(name="f_location_avg_fare",
                        key=location_id,
                        feature_type=FLOAT,
                        transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                          agg_func="AVG",
                                                          window="90d")),
                Feature(name="f_location_max_fare",
                        key=location_id,
                        feature_type=FLOAT,
                        transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                          agg_func="MAX",
                                                          window="90d")),
                Feature(name="f_location_total_fare_cents",
                        key=location_id,
                        feature_type=FLOAT,
                        transform=WindowAggTransformation(agg_expr="fare_amount_cents",
                                                          agg_func="SUM",
                                                          window="90d")),
                ]

agg_anchor = FeatureAnchor(name="aggregationFeatures",
                           source=batch_source,
                           features=agg_features)

# derived features
f_trip_time_distance = DerivedFeature(name="f_trip_time_distance",
                                      feature_type=FLOAT,
                                      input_features=[
                                          f_trip_distance, f_trip_time_duration],
                                      transform="f_trip_distance * f_trip_time_duration")

f_trip_time_rounded = DerivedFeature(name="f_trip_time_rounded",
                                     feature_type=INT32,
                                     input_features=[f_trip_time_duration],
                                     transform="f_trip_time_duration % 10")

feathr_client.build_features(anchor_list=[agg_anchor, request_anchor], derived_feature_list=[
    f_trip_time_distance, f_trip_time_rounded])


# register features
utils.logging.info("registering features")
feathr_client.register_features()
