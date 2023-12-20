
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import input_file_name, regexp_extract, to_date, mean, datediff, col
from src.connection.connect import retrieve_dw_table

from typing import Optional

def get_sensor_data(
    session: SparkSession,
    single: bool,
    aircraft_id: Optional[str] = None,
    date: Optional[str] = None
) -> DataFrame:
    """Returns a DataFrame that contains the mean values of the sensor for a given aircraftid and date."""
    path_to_csv_files = "../raw_data/*.csv"  
    df = session.read.csv(path_to_csv_files, header=True, inferSchema=True, sep=";")
    pattern = r'.*\/([^\/]*)\.csv$'
    df = df.withColumn("date", to_date(df["date"]))
    df = df.withColumn("aircraftid", regexp_extract(input_file_name(), pattern, 1).substr(-6, 6))
    

    if single:
        if aircraft_id is None or date is None:
            raise ValueError("Both aircraft_id and date must be provided when single is True.")
        
        df = df.where((df.aircraftid == aircraft_id) & (df.date == date))
        df = df.groupBy("date", "aircraftid").agg(
        mean("value").alias("sensor_mean_value")
        )
    else:   

        df = df.groupBy("date", "aircraftid").agg(
        mean("value").alias("sensor_mean_value")
        )
    return df



def retrieve_needed_tables(spark: SparkSession, username: str, password: str, want_aircraft_utilization: bool = False, want_operation_interruption: bool = False) -> DataFrame:
    """Retrieves the needed tables for performing the data management
    to obtain our matrix for training.
    """

    if want_aircraft_utilization:
        df = retrieve_dw_table("DW", "public", spark, "aircraftutilization", ["timeid", "aircraftid", "flighthours", "delayedminutes", "flightcycles"], username, password)
        # Check if aircraft_utilization DataFrame is retrieved successfully
        if not df:
            raise ValueError("Failed to retrieve aircraft_utilization table.")

    if want_operation_interruption:
        df = retrieve_dw_table("AMOS", "oldinstance", spark, "operationinterruption", ["subsystem", "aircraftregistration", "starttime"], username, password)
        # Check if operation_interruption DataFrame is retrieved successfully
        if not df:
            raise ValueError("Failed to retrieve operation_interruption table.")

    return df
    

def get_training_data(sensor_data: DataFrame, operation_interruption: DataFrame, aircraft_utilization: DataFrame)-> DataFrame:
    """Performs the necessary transformations needed
      for obtaining the training data matrix
    """

    joined_df = sensor_data.join(
        aircraft_utilization,
        (sensor_data.date == aircraft_utilization.timeid) & (sensor_data.aircraftid == aircraft_utilization.aircraftid),
        how='inner'
    )
    joined_df = joined_df.drop(aircraft_utilization.aircraftid)

    operation_interruption = operation_interruption.withColumn("starttime", to_date(operation_interruption.starttime))
    operation_interruption = operation_interruption.filter(operation_interruption.subsystem == 3453)

    condition = (joined_df.aircraftid == operation_interruption.aircraftregistration) & \
            (datediff(operation_interruption.starttime, joined_df.date) >= 0) & \
            (datediff(operation_interruption.starttime, joined_df.date) <= 6)

    joined_df_with_interruption = joined_df.join(operation_interruption, condition, "left_outer")

    joined_df_with_interruption = joined_df_with_interruption.withColumn(
    "operation_interruption",
    col("starttime").isNotNull().cast("int")
)

    joined_df_with_interruption = joined_df_with_interruption.drop("starttime", "subsystem", "aircraftregistration", "timeid")

    return joined_df_with_interruption


def prepare_data_before_prediction(tuple: DataFrame, aircraft_utilization: DataFrame, aircraftid: str, day: str)-> DataFrame: 
    """Join two DataFrames by 'date' and 'aircraftid' columns.
    """
    aircraft_utilization = aircraft_utilization.where((aircraft_utilization.aircraftid == aircraftid) & (aircraft_utilization.timeid == day))
    condition = (tuple.aircraftid == aircraft_utilization.aircraftid) & (tuple.date == aircraft_utilization.timeid) 
    joined_df = tuple.join(aircraft_utilization, condition, 'inner')
    #joined_df = tuple.join(aircraft_utilization, on=['date', 'aircraftid'], how='inner')
    #joined_df.show()

    joined_df = joined_df.drop("aircraftid", "timeid", "date")

    return joined_df


    
    



