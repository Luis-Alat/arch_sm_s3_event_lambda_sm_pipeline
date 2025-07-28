import argparse
import os
import joblib
import logging

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def main():
    
    logger.info("Starting training process")

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

    logger.info("Loading training data...")

    # Loading training data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # Defining right format for parameter class_weight
    class_weight = args.class_weight
    if (class_weight != "balanced") or (class_weight is not None):
        class_weight = {
            int(pair_value.split(":")[0]):float(pair_value.split(":")[1])
            for pair_value in class_weight.split(",")
        }

    logger.info("\nTraining model\n")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=class_weight,
        min_samples_split=args.min_samples_split,
        verbose=1
    )
    model.fit(X_train, y_train)

    # Evaluate model if validation data exist
    logger.info("Using validation dataset...")
    if args.validation and os.path.exists(os.path.join(args.validation, 'validation.csv')):
        
        val_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'), header=None)

        X_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1]

        predictions = model.predict(X_val)

        f1 = round(f1_score(y_val, predictions, average="macro"), 4)

        logger.debug(f"F1-score={f1}")
        print(f"F1-score={f1}")

    # Saving model
    logger.info("Saving trained model")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    logger.info("All done!")

if __name__ == '__main__':
    main()