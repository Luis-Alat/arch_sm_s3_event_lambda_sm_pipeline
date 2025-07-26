import argparse
import os
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def main():
    
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', None))

    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument("--class-weight", type=str, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)

    args = parser.parse_args()

    # Loading training data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # Training model
    class_weight = args.class_weight
    if (class_weight != "balanced") or (class_weight is not None):
        class_weight = json.loads(class_weight)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=class_weight,
        min_samples_split=args.min_samples_split
    )
    model.fit(X_train, y_train)

    # Evaluate model if validation data exist
    if args.validation and os.path.exists(os.path.join(args.validation, 'validation.csv')):
        
        val_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'), header=None)

        X_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1]

        predictions = model.predict(X_val)

        f1 = round(f1_score(y_val, predictions, average="macro"), 4)

        print(f"F1-score={f1}")

    # Saving model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == '__main__':
    main()