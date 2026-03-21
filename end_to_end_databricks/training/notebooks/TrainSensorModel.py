# Databricks notebook source
##################################################################################
# Sensor model training pipeline (MLflow + model registry + feature store)
# Parameters:
# * env
# * experiment_name
# * model_name
# * gold_feature_table
# * target_column
# * output_model_stage
##################################################################################

# COMMAND ----------

import os
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

from pyspark.sql import functions as F

# Inputs

dbutils.widgets.text('env', 'staging')

dbutils.widgets.text('experiment_name', '/Users/' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('user') + '/sensor_streaming_experiment')

dbutils.widgets.text('model_name', 'sensor_model')

dbutils.widgets.text('catalog_name', 'dev')

dbutils.widgets.text('schema_name', 'end_to_end_databricks')

dbutils.widgets.text('gold_feature_table', 'gold_sensor_features')

dbutils.widgets.text('target_column', 'mean_temp_1h')

dbutils.widgets.dropdown('output_model_stage', 'Staging', ['None', 'Staging', 'Production'])

experiment_name = dbutils.widgets.get('experiment_name')
model_name = dbutils.widgets.get('model_name')
catalog_name = dbutils.widgets.get('catalog_name')
schema_name = dbutils.widgets.get('schema_name')
gold_feature_table = dbutils.widgets.get('gold_feature_table')
full_table_name = f'{catalog_name}.{schema_name}.{gold_feature_table}'
full_model_name = f'{catalog_name}.{schema_name}.{model_name}'
target = dbutils.widgets.get('target_column')
output_stage = dbutils.widgets.get('output_model_stage')

# COMMAND ----------

import logging
import mlflow
from databricks.feature_store import FeatureStoreClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrainSensorModel')
logger.info('Starting TrainSensorModel notebook')

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

fs = FeatureStoreClient()

logger.info(f'Starting model training with table: {full_table_name}, target: {target}')

# COMMAND ----------

gold_df = spark.table(full_table_name).na.drop()

# We use feature store training set for repeatability and lineage
training_set = fs.create_training_set(
    gold_df,
    label=target,
    exclude_columns=['sensor_id', 'event_time', 'yyyy_mm']
)
train_df = training_set.load_df()

pandas_df = train_df.toPandas()

X = pandas_df.drop(target, axis=1)
y = pandas_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

with mlflow.start_run() as run:
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    logger.info(f'Model trained with RMSE: {rmse:.4f}, R2: {r2:.4f}')

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)
    mlflow.sklearn.log_model(
        model,
        artifact_path='sensor_rf_model',
        registered_model_name=full_model_name,
        signature=None,
        input_example=X_test.head(5)
    )

    if output_stage in ['Staging', 'Production']:
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(full_model_name, [output_stage])
        # transition the new version to requested stage
        new_version = client.get_model_version_by_label(full_model_name, 'None').version if latest else run.info.run_id
        # In the example we just register; stage transition in workflow orchestrator is preferred.

print(f'Model training complete with RMSE={rmse:.3f}, R2={r2:.3f}')
