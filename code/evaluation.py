import json
import logging
import pathlib
import joblib
import tarfile

import pandas as pd

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":

    logger.info("Starting evaluation.")

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.info("Loading RandomForest model.")
    
    model = joblib.load("model.joblib")

    logger.info("Reading test data.")

    test_path = "/opt/ml/processing/data/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].astype("int8").to_numpy()
    X_test = df.iloc[:, 1:].values

    logger.info("Performing predictions against test data.")

    predictions = model.predict(X_test)

    logger.debug("Calculating classification metrics.")

    f1  = f1_score(y_test, predictions, average="macro")
    cla_rep = classification_report(y_test, predictions)
    con_mat = confusion_matrix(y_test, predictions).tolist()
    
    report_dict = {
        "classification_metrics": {
                "test":{
                    "f1": f1,
                    "cr": cla_rep,
                    "cm": con_mat
                }
        }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report")

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as oJ:
        json.dump(report_dict, oJ, indent=4)