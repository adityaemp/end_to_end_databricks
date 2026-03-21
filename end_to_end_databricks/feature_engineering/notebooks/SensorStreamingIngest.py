# Databricks notebook source
##################################################################################
# Sensor Streaming Ingestion and Medallion ETL
#
# 1. Bronze: continuous ingestion of synthetic sensor events from rate stream
# 2. Silver: aggregate windows, clean anomalies, enrich
# 3. Gold: computed features for ML training
#
# Parameters:
# * bronze_path - DBFS path for bronze delta table
# * silver_path - DBFS path for silver delta table
# * gold_path   - DBFS path for gold delta table
# * checkpoint_base - DBFS path prefix for checkpoints
##################################################################################

# COMMAND ----------

# If running from the workspace where this notebook lives
import os
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

import logging
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SensorStreamingIngest')

bronze_path = dbutils.widgets.get('bronze_path') if dbutils.widgets.get('bronze_path') else '/tmp/end_to_end_databricks/bronze/sensor_data'
silver_path = dbutils.widgets.get('silver_path') if dbutils.widgets.get('silver_path') else '/tmp/end_to_end_databricks/silver/sensor_data'
gold_path = dbutils.widgets.get('gold_path') if dbutils.widgets.get('gold_path') else '/tmp/end_to_end_databricks/gold/sensor_data'
checkpoint_base = dbutils.widgets.get('checkpoint_base') if dbutils.widgets.get('checkpoint_base') else '/tmp/end_to_end_databricks/checkpoints/sensor_stream'

# COMMAND ----------

# Bronze Stream generation from synthetic sensor data
sensor_stream = (
    spark.readStream
        .format('rate')
        .option('rowsPerSecond', 20)
        .load()
        .withColumnRenamed('timestamp', 'event_time')
        .select(
            F.expr('CAST((value % 50) AS INT) + 1 AS sensor_id'),
            F.col('event_time'),
            (20.0 + 10.0 * F.sin(F.col('value') * 0.05) + (F.rand() - 0.5) * 3.0).alias('temperature'),
            (50.0 + 20.0 * F.cos(F.col('value') * 0.03) + (F.rand() - 0.5) * 5.0).alias('humidity'),
            (1000.0 + (F.rand() - 0.5) * 20.0).alias('pressure')
        )
)

bronze_query = (
    sensor_stream
    .writeStream
    .format('delta')
    .option('checkpointLocation', f'{checkpoint_base}/bronze')
    .outputMode('append')
    .trigger(processingTime='12 seconds')
    .start(bronze_path)
)

print('Started bronze stream query. Waiting 60 seconds to materialize sample data.')
bronze_query.awaitTermination(60)
bronze_query.stop()

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS sensor_streaming")

spark.sql(f"CREATE TABLE IF NOT EXISTS sensor_streaming.bronze_sensor_data USING DELTA LOCATION '{bronze_path}'")

# COMMAND ----------

# Silver: window aggregation and anomaly detection (filter unrealistic readings)
bronze_df = spark.read.format('delta').load(bronze_path)

processed_silver = (
    bronze_df.filter((F.col('temperature') >= -5) & (F.col('temperature') <= 80) & (F.col('humidity') >= 0) & (F.col('humidity') <= 100))
    .groupBy('sensor_id', F.window('event_time', '5 minutes', '1 minute'))
    .agg(
        F.avg('temperature').alias('avg_temp_5m'),
        F.avg('humidity').alias('avg_humidity_5m'),
        F.min('pressure').alias('min_pressure_5m'),
        F.max('pressure').alias('max_pressure_5m'),
        F.count('*').alias('count_records_5m')
    )
    .select(
        'sensor_id',
        F.col('window.end').alias('event_time'),
        F.expr('date_format(window.end, "yyyy-MM")').alias('yyyy_mm'),
        'avg_temp_5m',
        'avg_humidity_5m',
        'min_pressure_5m',
        'max_pressure_5m',
        'count_records_5m'
    )
)

(
    processed_silver
    .write
    .format('delta')
    .mode('overwrite')
    .option('mergeSchema', 'true')
    .save(silver_path)
)

spark.sql(f"CREATE TABLE IF NOT EXISTS sensor_streaming.silver_sensor_data USING DELTA LOCATION '{silver_path}'")

# COMMAND ----------

# Gold: feature generation for training (resample hourly and compute rolling stats)
silver_df = spark.read.format('delta').load(silver_path)

gold_df = (
    silver_df
    .withColumn('tmp', F.current_timestamp())
    .groupBy('sensor_id', F.window('event_time', '1 hour'))
    .agg(
        F.avg('avg_temp_5m').alias('mean_temp_1h'),
        F.stddev('avg_temp_5m').alias('std_temp_1h'),
        F.avg('avg_humidity_5m').alias('mean_humidity_1h'),
        F.last('max_pressure_5m').alias('last_pressure_1h'),
        F.sum('count_records_5m').alias('total_records_1h')
    )
    .select(
        'sensor_id',
        F.col('window.end').alias('event_time'),
        F.expr('date_format(window.end, "yyyy-MM")').alias('yyyy_mm'),
        'mean_temp_1h',
        'std_temp_1h',
        'mean_humidity_1h',
        'last_pressure_1h',
        'total_records_1h'
    )
)

(
    gold_df
    .write
    .format('delta')
    .mode('overwrite')
    .option('mergeSchema', 'true')
    .save(gold_path)
)

spark.sql(f"CREATE TABLE IF NOT EXISTS sensor_streaming.gold_sensor_features USING DELTA LOCATION '{gold_path}'")

# COMMAND ----------

print('Stream ingestion and medallion ETL complete.')

dbutils.notebook.exit(0)
