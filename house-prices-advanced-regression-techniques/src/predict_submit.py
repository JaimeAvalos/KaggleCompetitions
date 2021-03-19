import os

import argparse
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import impute


import config


def predict_submit(trained_model, submit):

    df_test = pd.read_csv(config.TEST_FILE_CLEAN)

    #predict target
    df_test_array = df_test.values

    clf = joblib.load('../models/' + trained_model + '.bin')

    preds = clf.predict(df_test_array)# * 10e-8

    preds = pd.DataFrame(preds)

    preds['Id'] = preds.index + 1461

    preds = preds[['Id',0]]
    
    preds.to_csv('../predictions/' + trained_model + '.csv', index=False, header= ['Id','SalePrice'] )

    if submit == 'yes':

        predictions_csv = '../predictions/' + trained_model + '.csv'

        submission = 'kaggle competitions submit -f ' + predictions_csv + ' -c ' + config.KAGGLE_NAME +' -m "Submission of model ' + trained_model + '"'

        os.system(submission)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--trained_model",
    type=str
    )
    parser.add_argument(
    "--submit",
    type=str
    )
    args = parser.parse_args()

predict_submit(
submit=args.submit,
trained_model=args.trained_model
)
