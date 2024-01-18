from typing import Any

import mlflow
from pyspark.sql import DataFrame, SparkSession

from utils import Colors
from data_management_pipeline import data_management_pipeline
###############################################################################
#
# Run-Time Classifier Pipeline:
#
#   - Execute Data Management Pipeline:
#         - Read DW aircraftutilization table and filter the aircraftid and date given
#         - Read CSV files and filter the aircraftid and date given
#         - Join sensor data with aircraftutilization table from DW to get
#           the tuple to predict
#   - Get the best model trained in the data analysis pipeline
#   - Make the prediction
#   - Print the predicted value
#
###############################################################################




def classifier_prediction(tuple_to_predict: DataFrame, best_model: Any):
    """Compute the prediction value for the given aircraftid and date with the a given model"""

    predictions = best_model.transform(tuple_to_predict)
    prediction_value = predictions.select("prediction").collect()[0][0]

    return prediction_value


def output_prediction(
    prediction_value: bool,
    best_model_name: str,
    aircraftid: str,
    date: str,
    tuple_to_predict: DataFrame,
):
    """Print the predicted value for the given input."""

    print(
        f"{Colors.GREEN}Using {best_model_name} for predicting tuple [{aircraftid}, {date}, {tuple_to_predict.select('sensor_mean_value').collect()[0][0]}, {tuple_to_predict.select('flighthours').collect()[0][0]}, {tuple_to_predict.select('delayedminutes').collect()[0][0]}, {tuple_to_predict.select('flightcycles').collect()[0][0]}]{Colors.RESET}"
    )

    print(
        f"{Colors.GREEN}No maintenance predicted for the next 7 days{Colors.RESET}"
        if prediction_value == 0
        else f"{Colors.GREEN}Unscheduled maintenance predicted in the next 7 days{Colors.RESET}"
    )


#####################################
# Run-Time Classifier Main Function #
#####################################


def runtime_classifier_pipeline(
    spark: SparkSession,
    aircraftid: str,
    date: str,
    username: str,
    password: str,
):
    """Compute all the Run-Time Classifier Pipeline."""

    # Execute data management pipeline to get the tuple to predict
    data_management_pipeline(spark=spark, username=username, password=password, aircraftid=aircraftid, date=date)

    # Get the best model saved
    model_path = "./best_model"
    best_model = mlflow.spark.load_model(model_path)
    best_model_name = best_model.stages[-1].uid.split("_")[0]

    # Load the tuple 
    tuple_to_predict = spark.read.csv(
        "./tuple_to_predict", header=True, inferSchema=True
    )
    tuple_to_predict = tuple_to_predict.drop("date", "aircraftid")

    # Make the prediction
    prediction_value = classifier_prediction(tuple_to_predict, best_model)

    # Output the predicted value
    output_prediction(
        prediction_value, best_model_name, aircraftid, date, tuple_to_predict
    )
