from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    datediff,
    input_file_name,
    mean,
    regexp_extract,
    to_date,
    round
)
from utils import Colors

###############################################################################
#
# Data Managemenet Pipeline:
#
#   - Read csv files to extract sensor data
#   - Read DW and AMOS needed tables
#   - Join sensor data with aircraftutilization table from DW
#   - Filter needed rows from operationinterruption table (
#       conditions:
#               - subsystem = 3453
#               - Maintenance is predicted in the next 7 days for that flight
#               - Maintenance kind = ['AircraftOnGround', 'Safety', 'Delay'] (unscheduled maintenance)
#       )
#   - Join previous resultant dataframe with filtered operationinterruption
#     rows and store it in a matrix
#   - Save the matrix in a csv file
#
###############################################################################


def retrieve_dw_table(
    db_name: str,
    table_instance: str,
    session: SparkSession,
    table_name: str,
    columns: List[str],
    username: str,
    password: str,
) -> DataFrame:
    """Given a database name, a table instance, a spark session,
    the name for the table you want to retrieve from the database
    and the columns you want to retrieve from that table, retrieves
    the those tables for performing the data management
    to obtain our matrix for training
    """

    db_properties = {
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://postgresfib.fib.upc.edu:6433/{db_name}?sslmode=require",
        "user": username,
        "password": password,
    }

    try:
        data = session.read.jdbc(
            url=db_properties["url"],
            table=table_instance + "." + table_name,
            properties=db_properties,
        )
    except ValueError:
        raise ValueError(
            f"{Colors.RED}Failed to retrieve {table_name} table.{Colors.RESET}"
        )

    df = data.select(columns)
    return df


def get_sensor_data(
    session: SparkSession,
) -> DataFrame:
    """Returns a DataFrame that contains the csv file with the mean values
    of the sensor, the aircraftid and the date."""

    path_to_csv_files = "./resources/trainingData/*.csv"
    df = session.read.csv(path_to_csv_files, header=True, inferSchema=True, sep=";")
    pattern = r".*\/([^\/]*)\.csv$"
    df = df.withColumn("date", to_date(df["date"]))
    df = df.withColumn(
        "aircraftid",
        regexp_extract(input_file_name(), pattern, 1).substr(-6, 6),
    )

    df = df.groupBy("date", "aircraftid").agg(mean("value").alias("sensor_mean_value"))

    return df



def get_training_data(
    sensor_data: DataFrame,
    operation_interruption: DataFrame,
    aircraft_utilization: DataFrame,
    aircraftid: Optional[str],
    date: Optional[str], 
) -> DataFrame:
    """Performs the necessary transformations needed
    for obtaining the training data matrix
    """

    # If we want only a single row
    if aircraftid is not None and date is not None: 
        sensor_data=sensor_data.where((sensor_data.date==date)&(sensor_data.aircraftid==aircraftid))
        aircraft_utilization=aircraft_utilization.where((aircraft_utilization.timeid==date)&(aircraft_utilization.aircraftid==aircraftid))

    # Join sensor data with aircraftutilization table from DW
    joined_df = sensor_data.join(
        aircraft_utilization,
        (sensor_data.date == aircraft_utilization.timeid)
        & (sensor_data.aircraftid == aircraft_utilization.aircraftid),
        how="inner",
    )
    joined_df = joined_df.drop(aircraft_utilization.aircraftid)

    # If we want to do the whole data matrix
    if aircraftid is None and date is None: 

        # Filter needed rows from operationinterruption table
        operation_interruption = operation_interruption.withColumn(
            "starttime", to_date(operation_interruption.starttime)
        )
        operation_interruption = operation_interruption.filter(
            (operation_interruption.subsystem == 3453)&
            (operation_interruption.kind.isin(['AircraftOnGround', 'Safety', 'Delay']))

        )

        condition = (
            (joined_df.aircraftid == operation_interruption.aircraftregistration)
            & (datediff(operation_interruption.starttime, joined_df.date) >= 0)
            & (datediff(operation_interruption.starttime, joined_df.date) <= 6)
        )

        # Join previous dataframes
        joined_df_with_interruption = joined_df.join(
            operation_interruption, condition, "left_outer"
        )

        joined_df_with_interruption = joined_df_with_interruption.withColumn(
            "label", col("starttime").isNotNull().cast("int")
        )

        joined_df_with_interruption = joined_df_with_interruption.drop(
            "starttime", "subsystem", "aircraftregistration", "timeid", "kind"
        )
        return joined_df_with_interruption
    
    else: 
        
        joined_df = joined_df.drop(
            "timeid"
        )
        return joined_df


#################################
# Data Management Main function #
#################################


def data_management_pipeline(spark: SparkSession, username: str, password: str, aircraftid: Optional[str] = None, date: Optional[str] = None):
    """Compute all the data management pipeline."""

    # Read csv files
    sensor_data: DataFrame = get_sensor_data(spark)

    # Read DW and AMOS needed tables
    aircraft_utilization = retrieve_dw_table(
        "DW",
        "public",
        spark,
        "aircraftutilization",
        [
            "timeid",
            "aircraftid",
            "flighthours",
            "delayedminutes",
            "flightcycles",
        ],
        username,
        password,
    )
    operation_interruption = retrieve_dw_table(
        "AMOS",
        "oldinstance",
        spark,
        "operationinterruption",
        ["subsystem", "aircraftregistration", "starttime", "kind"],
        username,
        password,
    )

    # Create a matrix with all needed data
    training_data: DataFrame = get_training_data(
        sensor_data, operation_interruption, aircraft_utilization, aircraftid, date
    )

    columns_to_round = ['sensor_mean_value', 'flighthours', 'delayedminutes', 'flightcycles']

    # Apply rounding to each column
    for column in columns_to_round:
        training_data = training_data.withColumn(column, round(training_data[column], 2))
    
    # Save the matrix in a csv file
    if aircraftid is None and date is None: 
        output_path = "./training_data"
    else: 
        output_path = "./tuple_to_predict"
    training_data.coalesce(1).write.csv(output_path, header=True, mode="overwrite")