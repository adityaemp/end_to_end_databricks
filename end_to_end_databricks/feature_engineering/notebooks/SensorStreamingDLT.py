# Databricks notebook source
##################################################################################
# Delta Live Tables Pipeline for Sensor Streaming Medallion Architecture
#
# Bronze: Raw streaming sensor data
# Silver: Cleaned and aggregated sensor data
# Gold: Feature table for ML training
##################################################################################

import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Bronze Layer: Raw sensor stream
@dlt.table(
    name="bronze_sensor_data",
    comment="Raw sensor streaming data with basic schema enforcement",
    table_properties={
        "quality": "bronze",
        "pipelines.reset.allowed": "true"
    }
)
def bronze_sensor_data():
    # Generate synthetic sensor data stream
    sensor_stream = (
        spark.readStream
        .format('rate')
        .option('rowsPerSecond', 20)
        .load()
        .withColumnRenamed('timestamp', 'event_time')
        .select(
            expr('CAST((value % 50) + 1 AS INT) AS sensor_id'),
            col('event_time'),
            (20.0 + 10.0 * sin(col('value') * 0.05) + (rand() - 0.5) * 3.0).alias('temperature'),
            (50.0 + 20.0 * cos(col('value') * 0.03) + (rand() - 0.5) * 5.0).alias('humidity'),
            (1000.0 + (rand() - 0.5) * 20.0).alias('pressure')
        )
    )

    # Schema enforcement
    schema = StructType([
        StructField("sensor_id", IntegerType(), False),
        StructField("event_time", TimestampType(), False),
        StructField("temperature", DoubleType(), False),
        StructField("humidity", DoubleType(), False),
        StructField("pressure", DoubleType(), False)
    ])

    return sensor_stream.withColumn("_rescued_data", col("_rescued_data")).select(
        col("sensor_id"),
        col("event_time"),
        col("temperature"),
        col("humidity"),
        col("pressure"),
        col("_rescued_data")
    )

# Silver Layer: Cleaned and aggregated data
@dlt.table(
    name="silver_sensor_data",
    comment="Cleaned sensor data with anomaly filtering and 5-minute aggregations",
    table_properties={
        "quality": "silver",
        "pipelines.reset.allowed": "true"
    }
)
@dlt.expect_or_drop("valid_temperature", "temperature BETWEEN -5 AND 80")
@dlt.expect_or_drop("valid_humidity", "humidity BETWEEN 0 AND 100")
def silver_sensor_data():
    return (
        dlt.read_stream("bronze_sensor_data")
        .filter((col('temperature') >= -5) & (col('temperature') <= 80) &
                (col('humidity') >= 0) & (col('humidity') <= 100))
        .groupBy('sensor_id', window('event_time', '5 minutes', '1 minute'))
        .agg(
            avg('temperature').alias('avg_temp_5m'),
            avg('humidity').alias('avg_humidity_5m'),
            min('pressure').alias('min_pressure_5m'),
            max('pressure').alias('max_pressure_5m'),
            count('*').alias('count_records_5m')
        )
        .select(
            'sensor_id',
            col('window.end').alias('event_time'),
            date_format(col('window.end'), 'yyyy-MM').alias('yyyy_mm'),
            'avg_temp_5m',
            'avg_humidity_5m',
            'min_pressure_5m',
            'max_pressure_5m',
            'count_records_5m'
        )
    )

# Gold Layer: Feature table for ML
@dlt.table(
    name="gold_sensor_features",
    comment="Hourly aggregated sensor features for ML training",
    table_properties={
        "quality": "gold",
        "pipelines.reset.allowed": "true"
    }
)
def gold_sensor_features():
    return (
        dlt.read("silver_sensor_data")
        .groupBy('sensor_id', window('event_time', '1 hour'))
        .agg(
            avg('avg_temp_5m').alias('mean_temp_1h'),
            stddev('avg_temp_5m').alias('std_temp_1h'),
            avg('avg_humidity_5m').alias('mean_humidity_1h'),
            last('max_pressure_5m').alias('last_pressure_1h'),
            sum('count_records_5m').alias('total_records_1h')
        )
        .select(
            'sensor_id',
            col('window.end').alias('event_time'),
            date_format(col('window.end'), 'yyyy-MM').alias('yyyy_mm'),
            'mean_temp_1h',
            'std_temp_1h',
            'mean_humidity_1h',
            'last_pressure_1h',
            'total_records_1h'
        )
    )