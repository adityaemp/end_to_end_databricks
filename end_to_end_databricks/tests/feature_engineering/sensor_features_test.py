import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from end_to_end_databricks.feature_engineering.features.sensor_features import compute_features_fn


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder.master('local[2]').appName('sensor_features_test').getOrCreate()


def test_sensor_features_computation(spark):
    data = [
        (1, '2026-03-21 00:00:00', 20.0, 45.0, 1013.0),
        (1, '2026-03-21 00:05:00', 21.5, 47.0, 1012.5),
        (1, '2026-03-21 00:10:00', 19.0, 44.0, 1014.0),
        (2, '2026-03-21 00:03:00', 15.0, 61.0, 1015.0),
    ]
    df = spark.createDataFrame(data, schema=['sensor_id', 'event_time', 'temperature', 'humidity', 'pressure'])
    df = df.withColumn('event_time', col('event_time').cast('timestamp'))

    out = compute_features_fn(df, 'event_time', '2026-03-21 00:00:00', '2026-03-21 01:00:00')
    result = out.filter(col('sensor_id') == 1).collect()
    assert len(result) > 0
    row = result[0]
    assert row['mean_temp_1h'] is not None
    assert row['event_time'] is not None
