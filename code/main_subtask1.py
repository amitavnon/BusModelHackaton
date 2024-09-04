from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from hackathon_code.models.ridge import RidgeModel

import logging
from hackathon_code.create_new_table import *
import pandas as pd
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
"""
usage:
    python code/main_subtask1.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main_subtask1.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


# implement here your load, preprocess,train,predict,save functions (or any other design you choose)
def load_set(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="ISO-8859-8")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    df = load_set(args.training_set)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    processed_df = advanced_preprocess(df)
    X, y = processed_df.drop(['passengers_up'], axis=1), processed_df.passengers_up

    # 3. train a model
    logging.info("training...")
    model = RidgeModel(ignore_estimated=True).fit(X, y)

    # 4. load the test set (args.test_set)
    test_df = load_set(args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")
    processed_test_df = advanced_preprocess(test_df, can_drop_samples=False)

    # Identify columns missing in test_df
    missing_columns = set(X.columns) - set(processed_test_df.columns)
    # Create a DataFrame with missing columns filled with 0s
    # Add each missing column to test_df with 0 values
    for col in tqdm(missing_columns, desc='Adding missing columns'):
        processed_test_df[col] = 0

    # Reorder columns to match df
    processed_test_df = processed_test_df[X.columns]

    # Removing new categorical columns
    mismatched_columns = set(processed_test_df.columns) - set(X.columns)
    processed_test_df.drop(mismatched_columns, axis=1, inplace=True)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(processed_test_df)
    predictions = predictions * (predictions > 0)

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    final_df = pd.DataFrame(test_df.trip_id_unique_station)
    final_df['passengers_up'] = predictions
    final_df.to_csv(args.out, index=False, encoding='ISO-8859-8')
