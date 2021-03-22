import pandas as pd
from sklearn import model_selection
from sklearn import impute

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import config

#this is just a line to try to commit it


if __name__ == "__main__":
    
    # Training data is in a CSV file called train.csv
    train_data = pd.read_csv(config.TRAINING_FILE).drop("Id", axis = 1)

    test_data = pd.read_csv(config.TEST_FILE).drop("Id", axis = 1)

    save_prices = train_data["SalePrice"]

    #Drop columns present in train data but not in test data:
    drop_columns = train_data.columns.difference(test_data.columns).drop("SalePrice")

    train_data = train_data.drop(drop_columns, axis = 1)


    #enlist training object columns
    train_columns = train_data.dtypes.apply(lambda x: x.name).to_dict()

    train_object_columns = {key: value for key, value in train_columns.items() if value == "object"}

    train_object_columns = list(train_object_columns.keys())


    #convert to dummy data
    train_data = pd.get_dummies(train_data, columns = train_object_columns , drop_first = True)

    
    #Apply imputer to infere missing values. Probably the creation of specific models to infere each missing value would be better
    #imputer = IterativeImputer(max_iter=10, random_state=0)
    
    #imputer = impute.KNNImputer(n_neighbors=5)

    #imputed_data = imputer.fit_transform(train_data.values)

    #train_data = pd.DataFrame(imputed_data, columns = train_data.columns)

    #Now the same thing with the test data in order to get rid of columns not present in the testdata
    #enlist training object columns
    #test_columns = test_data.dtypes.apply(lambda x: x.name).to_dict()

    #test_object_columns = {key: value for key, value in test_columns.items() if value == "object"}

    #test_object_columns = list(test_object_columns.keys())


    #convert to dummy data
    test_data = pd.get_dummies(test_data, columns = train_object_columns , drop_first = True)


    full_data  = pd.concat([train_data,test_data]).drop("SalePrice", axis = 1)

    
    #Apply imputer to fill missing values:
    #imputer = IterativeImputer(max_iter=10, random_state=0)
    
    imputer = impute.KNNImputer(n_neighbors=5)

    #imputed_data = imputer.fit_transform(test_data.values)

    #test_data = pd.DataFrame(imputed_data, columns = test_data.columns)

    imputed_data = imputer.fit_transform(full_data.values)

    test_data = pd.DataFrame(imputed_data[(train_data.shape[0]):(imputed_data.shape[0]),:], columns = full_data.columns)

    train_data = pd.DataFrame(imputed_data[:train_data.shape[0]-1,:], columns = full_data.columns)

    train_data["SalePrice"] = save_prices# *10e8

    #Save cleant data

    train_data.to_csv(config.TRAINING_FILE_CLEAN, index=False)

    test_data.to_csv(config.TEST_FILE_CLEAN, index=False)