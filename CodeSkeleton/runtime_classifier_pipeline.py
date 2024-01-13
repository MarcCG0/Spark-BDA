from typing import Any

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import mean, round

from utils import Colors, concatenate_csv_sensor_data, retrieve_dw_table

###############################################################################
#
# Run-Time Classifier Pipeline:
#
#   - Read DW aircraftutilization table
#   - Read CSV files with the given aircraftid and date
#   - Join sensor data with aircraftutilization table from DW to get
#     the tuple to predict
#   - Get the best model trained in the data analysis pipeline
#   - Make the prediction
#   - Print the predicted value
#
###############################################################################


def get_sensor_data(session: SparkSession, aircraft_id: str, date: str) -> DataFrame:
    """Returns a DataFrame that contains the mean values of the sensor for a given aircraftid and date."""

    df: DataFrame = concatenate_csv_sensor_data(session=session)

    df = df.where((df.aircraftid == aircraft_id) & (df.date == date))

    if df.isEmpty():
        raise ValueError(
            f"{Colors.RED}No tuple for the combination of date-aircraftid given by the user found in the database{Colors.RESET}"
        )

    df = df.groupBy("date", "aircraftid").agg(
        round(mean("value"), 2).alias("sensor_mean_value")
    )

    return df


def prepare_data_before_prediction(
    tuple: DataFrame,
    aircraft_utilization: DataFrame,
    aircraftid: str,
    day: str,
) -> DataFrame:
    """Join two DataFrames by 'date' and 'aircraftid' columns."""
    aircraft_utilization = aircraft_utilization.where(
        (aircraft_utilization.aircraftid == aircraftid)
        & (aircraft_utilization.timeid == day)
    )
    condition = (tuple.aircraftid == aircraft_utilization.aircraftid) & (
        tuple.date == aircraft_utilization.timeid
    )
    joined_df = tuple.join(aircraft_utilization, condition, "inner")
    joined_df = joined_df.drop("aircraftid", "timeid", "date")
    return joined_df


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

    columns = [
        "timeid",
        "aircraftid",
        "CAST(ROUND(flighthours, 2) AS DECIMAL(10,2)) as flighthours",
        "CAST(ROUND(delayedminutes, 2) AS DECIMAL(10,2)) as delayedminutes",
        "CAST(ROUND(flightcycles, 2) AS DECIMAL(10,2)) as flightcycles",
    ]
    table_instance = "public"

    table_name = "aircraftutilization"

    query = f"SELECT {', '.join(columns)} FROM {table_instance}.{table_name} WHERE aircraftid = '{aircraftid}' AND timeid = '{date}'"

    # Read the needed DW table and CSV files
    aircraft_utilization = retrieve_dw_table(
        db_name="DW",
        session=spark,
        username=username,
        password=password,
        query=query,
    )
    filtered_sensor_data = get_sensor_data(spark, aircraftid, date)

    # Get the filtered tuple
    tuple_to_predict = prepare_data_before_prediction(
        filtered_sensor_data, aircraft_utilization, aircraftid, date
    )
    # Get the best model saved
    model_path = "./best_model"
    best_model = mlflow.spark.load_model(model_path)
    best_model_name = best_model.stages[-1].uid.split("_")[0]

    # Make the prediction
    prediction_value = classifier_prediction(tuple_to_predict, best_model)

    # Output the predicted value
    output_prediction(
        prediction_value, best_model_name, aircraftid, date, tuple_to_predict
    )
