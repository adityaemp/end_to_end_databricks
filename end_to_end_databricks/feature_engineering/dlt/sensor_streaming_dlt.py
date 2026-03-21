import dlt
from pyspark.sql import functions as F


@dlt.table(
    name="bronze_sensor_data",
    comment="Bronze-layer raw sensor data from synthetic rate source",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_drop("valid_event_time", "event_time IS NOT NULL")
@dlt.expect_or_drop("valid_sensor_id", "sensor_id BETWEEN 1 AND 10000")
def bronze_sensor_data():
    # Synthetic sensor stream from rate source; in real deployment this is replaced with an external stream.
    return (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", 40)
        .load()
        .withColumn("event_time", F.col("timestamp"))
        .withColumn("sensor_id", (F.col("value") % 50 + 1).cast("int"))
        .withColumn("temperature", (20.0 + 10.0 * F.sin(F.col("value") * 0.05) + (F.rand() - 0.5) * 3.0).cast("double"))
        .withColumn("humidity", (50.0 + 20.0 * F.cos(F.col("value") * 0.03) + (F.rand() - 0.5) * 5.0).cast("double"))
        .withColumn("pressure", (1000.0 + (F.rand() - 0.5) * 20.0).cast("double"))
        .drop("timestamp", "value")
    )


@dlt.table(name="silver_sensor_data", comment="Silver-layer cleansed sensor aggregates", table_properties={"quality": "silver"})
@dlt.expect("not_null_temperature", "temperature IS NOT NULL")
@dlt.expect("temperature_safe_range", "temperature BETWEEN -20 AND 120")
def silver_sensor_data():
    bronze = dlt.read("bronze_sensor_data")
    cleaned = bronze.filter((F.col("temperature") >= -10) & (F.col("temperature") <= 80) & (F.col("humidity") >= 0) & (F.col("humidity") <= 100))

    # Aggregate in 5m tumbling windows
    return (
        cleaned
        .groupBy("sensor_id", F.window("event_time", "5 minutes"))
        .agg(
            F.avg("temperature").alias("avg_temp_5m"),
            F.avg("humidity").alias("avg_humidity_5m"),
            F.min("pressure").alias("min_pressure_5m"),
            F.max("pressure").alias("max_pressure_5m"),
            F.count("*").alias("count_5m"),
        )
        .select(
            "sensor_id",
            F.col("window.end").alias("event_time"),
            F.date_format(F.col("window.end"), "yyyy-MM").alias("yyyy_mm"),
            "avg_temp_5m",
            "avg_humidity_5m",
            "min_pressure_5m",
            "max_pressure_5m",
            "count_5m",
        )
    )


@dlt.table(name="gold_sensor_features", comment="Gold-layer ML features & labeling", table_properties={"quality": "gold"})
@dlt.expect_or_drop("non_null_avg_temp", "avg_temp_5m IS NOT NULL")
def gold_sensor_features():
    silver = dlt.read("silver_sensor_data")

    return (
        silver
        .groupBy("sensor_id", F.window("event_time", "1 hour"))
        .agg(
            F.avg("avg_temp_5m").alias("mean_temp_1h"),
            F.stddev("avg_temp_5m").alias("std_temp_1h"),
            F.avg("avg_humidity_5m").alias("mean_humidity_1h"),
            F.last("max_pressure_5m").alias("last_pressure_1h"),
            F.sum("count_5m").alias("total_records_1h"),
        )
        .select(
            "sensor_id",
            F.col("window.end").alias("event_time"),
            F.date_format(F.col("window.end"), "yyyy-MM").alias("yyyy_mm"),
            "mean_temp_1h",
            "std_temp_1h",
            "mean_humidity_1h",
            "last_pressure_1h",
            "total_records_1h",
            (F.col("mean_temp_1h") + F.unix_timestamp(F.col("event_time")) * 1.0).alias("forecast_feature"),
        )
    )
