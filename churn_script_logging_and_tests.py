"""
This Module contains multiple tests for churn_library.py

This tests are:
    * test_import
    * test_eda
    * test_encoder_helper
    * test_perform_feature_engineering
    * test_train_models

Author: Ahiwe Onyebuchi Valentine
Date: July 2021
"""

import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Category columns in dataframe
CATEGORY_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def test_import(import_data):
    '''
    This tests the import_data function from churn_library

    input:
            import_data: import_data function from churn_library
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return dataframe


def test_eda(perform_eda):
    '''
    This tests the perform_eda function from churn_library

    input:
            perform_eda: perform_eda function from churn_library
    output:
            None
    '''
    try:
        dataframe = test_import(cls.import_data)
        perform_eda(dataframe)
        logging.info("Testing eda: SUCCESS")
    except BaseException:
        logging.error("Testing eda: EDA failed")
        raise


def test_encoder_helper(encoder_helper):
    '''
    This tests the encoder_helper function from churn_library

    input:
            encoder_helper: encoder_helper function from churn_library
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataframe = test_import(cls.import_data)
        encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
        logging.info("Testing encoder_helper: SUCCESS")
        return dataframe
    except BaseException:
        logging.error("Testing encoder_helper: encoder_helper failed")
        raise


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    This tests the perform_feature_engineering function from churn_library

    input:
            perform_feature_engineering: perform_feature_engineering function from churn_library
    output:
            x_train: x training data
            x_test: x testing data
            y_train: y training data
            y_test: y testing data
    '''
    try:
        dataframe = test_encoder_helper(cls.encoder_helper)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, "Churn")
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
        return x_train, x_test, y_train, y_test
    except BaseException:
        logging.error(
            "Testing test_perform_feature_engineering: test_perform_feature_engineering failed")
        raise


def test_train_models(train_models):
    '''
    This tests the train_models function from churn_library

    input:
            train_models: train_models function from churn_library
    output:
            None
    '''
    try:
        x_train, x_test, y_train, y_test = test_perform_feature_engineering(
            cls.perform_feature_engineering)
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing test_train_models: SUCCESS")
    except BaseException:
        logging.error(
            "Testing test_train_models: test_train_models failed")
        raise


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
