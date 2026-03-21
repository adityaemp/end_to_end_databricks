import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType, TimestampType


@F.udf(returnType=StringType())
def _partition_id(dt):
    return f"{dt.year:04d}-{dt.month:02d}"


def _filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(F.col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(F.col(ts_column) < end_date)
    return df


def compute_features_fn(input_df, timestamp_column, start_date, end_date):
    """Compute medallion sensor features.

    input_df: bronze sensor stream
    timestamp_column: 'event_time'
    """
    df = _filter_df_by_ts(input_df, timestamp_column, start_date, end_date)

    # 1h and 15min sliding window from bronze readings
    sensor_stats = (
        df.groupBy("sensor_id", F.window(timestamp_column, "1 hour", "15 minutes"))
        .agg(
            F.mean("temperature").alias("mean_temp_1h"),
            F.stddev("temperature").alias("stddev_temp_1h"),
            F.mean("humidity").alias("mean_humidity_1h"),
            F.count("*").alias("event_count_1h"),
        )
        .select(
            F.col("sensor_id"),
            F.col("window.end").alias(timestamp_column),
            _partition_id(F.col("window.end")).alias("yyyy_mm"),
            F.col("mean_temp_1h").cast(FloatType()),
            F.col("stddev_temp_1h").cast(FloatType()),
            F.col("mean_humidity_1h").cast(FloatType()),
            F.col("event_count_1h").cast(IntegerType()),
        )
    )

    return sensor_stats
