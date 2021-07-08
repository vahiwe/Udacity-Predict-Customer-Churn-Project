# library doc string
"""
This Module contains multiple functions used to accomplish common tasks
in data science

This file can also be imported as a module and contains the following
functions:

    * import_data - returns dataframe for the csv found at pth
    * perform_eda - perform eda on df and save figures to images folder
    * encoder_helper - helper function to turn each categorical column into a new column with
      propotion of churn for each category - associated with cell 15 from the notebook
    * perform_feature_engineering - performs feature engineering on the dataframe
    * classification_report_image - produces classification report for training and
      testing results and stores report as image in images folder
    * feature_importance_plot - creates and stores the feature importances in pth
    * train_models - train, store model results: images + scores, and store models

Author: Ahiwe Onyebuchi Valentine
Date: July 2021
"""

# import libraries
import logging
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

# Create an object that holds the arguments for classification_report_image to avoid
# R0913: Too many arguments from pylint


class ClassificationPredictedValue():
    '''
    Class to store predicted values for train and test
    '''

    def __init__(self,
                 y_train_preds_lr,
                 y_train_preds_rf,
                 y_test_preds_lr,
                 y_test_preds_rf):
        self.y_train_preds_lr = y_train_preds_lr
        self.y_train_preds_rf = y_train_preds_rf
        self.y_test_preds_lr = y_test_preds_lr
        self.y_test_preds_rf = y_test_preds_rf

    def get_y_train_preds_lr(self):
        '''
        returns y_train_preds_lr
        '''
        return self.y_train_preds_lr

    def get_y_train_preds_rf(self):
        '''
        returns y_train_preds_rf
        '''
        return self.y_train_preds_rf

    def get_y_test_preds_lr(self):
        '''
        returns y_test_preds_lr
        '''
        return self.y_test_preds_lr

    def get_y_test_preds_rf(self):
        '''
        returns y_test_preds_rf
        '''
        return self.y_test_preds_rf


# Storage Locations for Models
RANDOM_FOREST_MODEL = './models/rfc_model.pkl'
LOGISTIC_REGRESSION_MODEL = './models/logistic_model.pkl'

# Image Location
EDA_IMAGE_PATH = "images/eda/"
RESULT_IMAGE_PATH = "images/results/"

# Category columns in dataframe
CATEGORY_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

# Quantity columns in dataframe
QUANTITY_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

logging.basicConfig(
    filename='logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataframe = pd.DataFrame(pd.read_csv(pth))
        logging.info(
            "SUCCESS: Read file at %s with %s rows",
            pth,
            dataframe.shape[0])
    except FileNotFoundError as err:
        logging.error("ERROR: Failed to read file at %s", pth)
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("The file doesn't appear to have rows and columns")
        raise err

    # create churn column using lambda on Attrition_flag column
    try:
        assert 'Attrition_Flag' in dataframe.columns
        assert dataframe["Attrition_Flag"].shape[0] > 0

        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        return dataframe

    except AssertionError as err:
        logging.error("Creation of Churn Column Failed")
        raise err


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    try:
        # Specify histogram plots
        histogram_plot_columns = [
            "Churn",
            "Customer_Age",
            "Marital_Status",
            "Total_Trans_Ct",
            "Heat_Map"]

        # Create plot for the different columns
        for column in histogram_plot_columns:

            # Ignore Assertion for Heat Map
            if column != "Heat_Map":
                assert column in dataframe.columns
                assert dataframe[column].shape[0] > 0

            logging.info("Creating plot for %s", column)

            plt.figure(figsize=(20, 10))
            if column in CATEGORY_COLUMNS:
                dataframe[column].value_counts('normalize').plot(kind='bar')
            elif column == "Total_Trans_Ct":
                sns.distplot(dataframe['Total_Trans_Ct'])
            elif column == "Heat_Map":
                sns.heatmap(
                    dataframe.corr(),
                    annot=False,
                    cmap='Dark2_r',
                    linewidths=2)
            else:
                dataframe[column].hist()

            # Save plot
            plot_saved_location = f'{EDA_IMAGE_PATH}{column}_plot.png'
            logging.info(
                "Saving plot for %s in %s",
                column,
                plot_saved_location)
            plt.savefig(plot_saved_location)
            logging.info("SUCCESS: EDA images successfully Generated")
    except AssertionError as err:
        logging.error("ERROR: Creation of EDA images failed")
        raise err


def group_by_helper(dataframe, category, category_group):
    '''
      helper function to for encoder help
      This avoids the W0640: Cell variable category_group defined in loop from pylint

      input:
            dataframe: pandas dataframe
            category: Category
            category_group: category_group

      output:
              Numpy Series
    '''
    return dataframe[category].apply(
        lambda x: category_group.loc[x])


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
                      could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        for category in category_lst:
            assert category in dataframe.columns
            assert dataframe[category].shape[0] > 0

            logging.info(
                "Calculating churn proportion for %s column",
                category)
            category_group = dataframe.groupby(category).mean()[response]
            dataframe[f'{category}_{response}'] = group_by_helper(
                dataframe, category, category_group)

        logging.info("SUCCESS: Encoding of categorical data complete")
    except AssertionError as err:
        logging.error("ERROR: Encoding of categorical data failed")
        raise err


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument
                        that could be used for naming variables or index y column]

    output:
              x_train_df: x training data
              x_test_df: x testing data
              y_train_df: y training data
              y_test_df: y testing data
    '''
    # store the quantity columns
    keep_cols = QUANTITY_COLUMNS[:]

    # Extend with churn categorical columns
    keep_cols.extend([f'{column}_{response}' for column in CATEGORY_COLUMNS])

    try:
        assert response in dataframe.columns
        assert dataframe[response].shape[0] > 0

        y_df = dataframe[response]

        for column_name in keep_cols:
            assert column_name in dataframe.columns
            assert dataframe[column_name].shape[0] > 0

        x_df = pd.DataFrame()
        x_df[keep_cols] = dataframe[keep_cols]

        # train test split
        x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
            x_df, y_df, test_size=0.3, random_state=42)

        logging.info("SUCCESS: Feature engineering complete")
        return x_train_df, x_test_df, y_train_df, y_test_df
    except AssertionError as err:
        logging.error("ERROR: Feature engineering failed")
        raise err


def roc_image(lrc_model,
              cv_rfc_model,
              x_test_df,
              y_test_df):
    '''
    produces ROC image of the models
    input:
            lrc_model: Logistic Regression model
            cv_rfc_model: Random Forest model
            x_test_df: test data
            y_test_df: test response values
    output:
             None
    '''
    # Plot ROC for Logistic Regression
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lrc_model, x_test_df, y_test_df)

    # Plot ROC for Random forest
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc_model.best_estimator_,
        x_test_df,
        y_test_df,
        ax=ax,
        alpha=0.8)

    # Save plot
    plot_saved_location = f'{RESULT_IMAGE_PATH}ROC.png'
    logging.info(
        "SUCCESS: Saving plot for ROC in %s",
        plot_saved_location)
    plt.savefig(plot_saved_location)


def classification_report_image(y_train_dataframe,
                                y_test_dataframe,
                                predicted_values):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train_dataframe: training response values
            y_test_dataframe:  test response values
            predicted_values: An instance of ClassificationPredictedValue that contains
                              y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf

    output:
             None
    '''
    models = ["Random Forest", "Logistic Regression"]
    y_train_preds = [
        predicted_values.get_y_train_preds_rf(),
        predicted_values.get_y_train_preds_lr()]
    y_test_preds = [
        predicted_values.get_y_test_preds_rf(),
        predicted_values.get_y_test_preds_lr()]

    for (
            model,
            y_train_pred,
            y_test_pred) in zip(
            models,
            y_train_preds,
            y_test_preds):
        plt.figure(figsize=(10, 10))
        plt.rc('figure', figsize=(5, 5))

        plt.text(
            0.01, 1, str(f'{model} Train'), {
                'fontsize': 10}, fontproperties='monospace')

        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.4, str(
                classification_report(
                    y_test_dataframe, y_test_pred)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.6, str(f'{model} Test'), {
                'fontsize': 10}, fontproperties='monospace')

        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.8, str(
                classification_report(
                    y_train_dataframe, y_train_pred)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        # Save plot
        plot_saved_location = f'{RESULT_IMAGE_PATH}{model} Result.png'
        logging.info(
            "SUCCESS: Saving plot for %s in %s",
            model,
            plot_saved_location)
        plt.savefig(plot_saved_location)


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plot_saved_location = f'{output_pth}feature_importance_plot.png'

    # Save plot
    plt.savefig(plot_saved_location)

    logging.info(
        "SUCCESS: Saving plot for feature importance in %s",
        plot_saved_location)


def train_models(
        x_train_dataframe,
        x_test_dataframe,
        y_train_dataframe,
        y_test_dataframe):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train_dataframe: X training data
              x_test_dataframe: X testing data
              y_train_dataframe: y training data
              y_test_dataframe: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # params_grid for Random Forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Training Random Forest Model
    logging.info("Training Random Forest Model")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train_dataframe, y_train_dataframe)

    # Training Logistic Regression Model
    logging.info("Training Logistic Regression Model")
    lrc.fit(x_train_dataframe, y_train_dataframe)

    logging.info("SUCCESS: Done Training Models")

    # Prediction from Random Forest Model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train_dataframe)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test_dataframe)

    # Prediction from Logistic Regression Model
    y_train_preds_lr = lrc.predict(x_train_dataframe)
    y_test_preds_lr = lrc.predict(x_test_dataframe)

    logging.info("SUCCESS: Prediction Successful")

    # Create Image For Scores
    predicted_values = ClassificationPredictedValue(
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    classification_report_image(
        y_train_dataframe,
        y_test_dataframe,
        predicted_values)

    # Create feature importance image
    feature_importance_plot(cv_rfc, x_train_dataframe, RESULT_IMAGE_PATH)

    # Create ROC image
    roc_image(lrc, cv_rfc, x_test_dataframe, y_test_dataframe)

    # Save Random Forest Model
    logging.info("Saving Random Forest model at %s", RANDOM_FOREST_MODEL)
    joblib.dump(cv_rfc.best_estimator_, RANDOM_FOREST_MODEL)

    # Save Logistic Regression Model
    logging.info(
        "Saving Logistic Regression model at %s",
        LOGISTIC_REGRESSION_MODEL)
    joblib.dump(lrc, LOGISTIC_REGRESSION_MODEL)


if __name__ == "__main__":
    try:
        # Create dataframe from file
        DF = import_data("data/bank_data.csv")

        # Perform EDA on the Dataframe
        perform_eda(DF)

        # Encode Categorical Data
        encoder_helper(DF, CATEGORY_COLUMNS, "Churn")

        # Perform Feature engineering to split data
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            DF, "Churn")

        # Train models
        train_models(X_train, X_test, y_train, y_test)
    except BaseException:
        logging.error("Model Training Failed")
        raise
