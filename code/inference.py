import os
import io
import joblib
import pandas as pd


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    transformer = joblib.load(os.path.join(model_dir, "transformer.joblib"))
    return {"model": model, "transformer": transformer}


def input_fn(input_data, content_type):
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(input_data))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_bundle):
    transformer = model_bundle["transformer"]
    model = model_bundle["model"]

    processed_data = transformer.transform(input_data)

    prediction = model.predict(processed_data)
    
    return prediction


def output_fn(prediction, accept):
    if accept == "text/csv":
        return ",".join(str(x) for x in prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


