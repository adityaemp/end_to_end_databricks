-- Placeholder Delta tables for Lakehouse Monitoring and batch inference.
-- Run each block in a SQL editor or notebook with privileges on the catalog.
-- Column names match deployment/batch_inference/predict.py and monitoring-resource.yml
-- (model_id, prediction, fare_amount, timestamp).

-- dev
CREATE TABLE IF NOT EXISTS dev.end_to_end_databricks.predictions (
  model_id BIGINT,
  prediction DOUBLE,
  fare_amount DOUBLE,
  `timestamp` TIMESTAMP
) USING DELTA;

-- staging
CREATE TABLE IF NOT EXISTS staging.end_to_end_databricks.predictions (
  model_id BIGINT,
  prediction DOUBLE,
  fare_amount DOUBLE,
  `timestamp` TIMESTAMP
) USING DELTA;

-- prod
CREATE TABLE IF NOT EXISTS prod.end_to_end_databricks.predictions (
  model_id BIGINT,
  prediction DOUBLE,
  fare_amount DOUBLE,
  `timestamp` TIMESTAMP
) USING DELTA;

-- test
CREATE TABLE IF NOT EXISTS test.end_to_end_databricks.predictions (
  model_id BIGINT,
  prediction DOUBLE,
  fare_amount DOUBLE,
  `timestamp` TIMESTAMP
) USING DELTA;
