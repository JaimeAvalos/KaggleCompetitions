import os
import argparse

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import numpy as np
#from sklearn.metrics import r2_score


#these two belong to the hyper parameter optimization part
from sklearn import ensemble
from sklearn import model_selection

import config
import model_dispatcher

optimize_param = False

def run(fold, model):

    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE_FOLDS)

    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    x_train = df_train.drop(["SalePrice","kfold"], axis=1).values
    y_train = df_train.SalePrice.values
    
    
    # similarly, for validation, we have
    x_valid = df_valid.drop(["SalePrice","kfold"] , axis=1).values
    y_valid = df_valid.SalePrice.values
    

    if optimize_param == True: #randomized search to look for the best parameters

        classifier = ensemble.RandomForestClassifier(n_jobs=-1)

        param_grid = {
            "n_estimators": np.arange(1300, 1700, 100),
            "max_depth": np.arange(25, 38),
            "criterion": ["gini"], #, "entropy"
            "min_samples_split": np.arange(8, 15),
            "min_samples_leaf": np.arange(1, 10)


        }

        clf = model_selection.RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_grid,
            n_iter=20,
            scoring="accuracy",
            verbose=10,
            n_jobs=1,
            cv=5
            )

        clf.fit(x_train, y_train)

        print("Best parameters set:")
        best_parameters = clf.best_estimator_.get_params()
        
        for param_name in sorted(param_grid.keys()):
            print(f"\t{param_name}: {best_parameters[param_name]}")

    else:
        clf = model_dispatcher.models[model]

        # fit the model on training data
        clf.fit(x_train, y_train)


    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy

    metrica = metrics.mean_squared_error(np.log(abs(y_valid)), np.log(abs(preds)), squared=False )
        
    print(f"Fold={fold}, Accuracy={metrica}")

    # save the model

    if metrica < 0.001:
        joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}_{metrica}.bin"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--fold",
    type=int
    )
    parser.add_argument(
    "--model",
    type=str
    )
    args = parser.parse_args()

run(
fold=args.fold,
model=args.model
)