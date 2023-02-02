
import argparse
import os
import glob

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from azure.identity import DefaultAzureCredential 
from azure.ai.ml.entities import AzureDataLakeGen2Datastore
from azure.ai.ml import MLClient

import mlflow
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to input data.")
    parser.add_argument("--registered_model_name", type=str, help="Model name.")
    parser.add_argument("--config_path", type=str, help="Path to AML config file.")
    parser.add_argument("--azure_tenant_id", type=str)
    parser.add_argument("--azure_client_id", type=str)
    parser.add_argument("--azure_client_secret", type=str)
    args = parser.parse_args()

    os.environ['AZURE_TENANT_ID'] = args.azure_tenant_id
    os.environ['AZURE_CLIENT_ID'] = args.azure_client_id
    os.environ['AZURE_CLIENT_SECRET'] = args.azure_client_secret
    print("ENV:")
    print(os.environ['AZURE_TENANT_ID'])
    print(os.environ['AZURE_CLIENT_ID'])

    ml_client = MLClient.from_config(DefaultAzureCredential(), path=args.config_path)
    ws = ml_client.workspaces.get(ml_client.workspace_name) 

    # df = pd.read_csv(args.data, header=1, index_col=0)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    X = df_new.loc[:, df_new.columns != "label"]
    y = df_new["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        # "n_estimators": 500,
        # "max_depth": 4,
        # "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }


    reg = ensemble.HistGradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    mse = mean_squared_error(y_test, reg.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    mlflow.log_metric("MSE", mse)

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=reg,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name
    )

    # # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=reg,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    ###########################
    #</save and register model>
    ###########################
    mlflow.end_run()

if __name__ == '__main__':
    main()
