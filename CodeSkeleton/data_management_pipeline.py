from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, datediff, mean, round, to_date

from utils import concatenate_csv_sensor_data, retrieve_dw_table

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
#       )
#   - Join previous resultant dataframe with filtered operationinterruption
#     rows and store it in a matrix
#   - Save the matrix in a csv file
#
###############################################################################


def get_sensor_data(
    session: SparkSession,
) -> DataFrame:
    """Returns a DataFrame that contains the csv file with the mean values
    of the sensor, the aircraftid and the date."""

    df: DataFrame = concatenate_csv_sensor_data(session=session)

    df = df.groupBy("date", "aircraftid").agg(
        round(mean("value"), 2).alias("sensor_mean_value")
    )

    return df


def get_training_data(
    sensor_data: DataFrame,
    operation_interruption: DataFrame,
    aircraft_utilization: DataFrame,
) -> DataFrame:
    """Performs the necessary transformations needed
    for obtaining the training data matrix
    """

    # Join sensor data with aircraftutilization table from DW
    joined_df = sensor_data.join(
        aircraft_utilization,
        (sensor_data.date == aircraft_utilization.timeid)
        & (sensor_data.aircraftid == aircraft_utilization.aircraftid),
        how="inner",
    )
    joined_df = joined_df.drop(aircraft_utilization.aircraftid)

    # Filter needed rows from operationinterruption table
    operation_interruption = operation_interruption.withColumn(
        "starttime", to_date(operation_interruption.starttime)
    )
    operation_interruption = operation_interruption.filter(
        operation_interruption.subsystem == 3453
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

    # Delete the unnecessary columns
    joined_df_with_interruption = joined_df_with_interruption.drop(
        "starttime", "subsystem", "aircraftregistration", "timeid"
    )

    return joined_df_with_interruption


#################################
# Data Management Main function #
#################################


def data_management_pipeline(spark: SparkSession, username: str, password: str):
    """Compute all the data management pipeline."""

    # Read csv files
    sensor_data: DataFrame = get_sensor_data(spark)

    columns = [
        "timeid",
        "aircraftid",
        "CAST(ROUND(flighthours, 2) AS DECIMAL(10,2)) as flighthours",
        "CAST(ROUND(delayedminutes, 2) AS DECIMAL(10,2)) as delayedminutes",
        "CAST(ROUND(flightcycles, 2) AS DECIMAL(10,2)) as flightcycles",
    ]
    table_instance = "public"
    table_name = "aircraftutilization"

    query = f"SELECT {', '.join(columns)} FROM {table_instance}.{table_name}"

    # Read DW and AMOS needed tables
    aircraft_utilization = retrieve_dw_table(
        db_name="DW",
        session=spark,
        username=username,
        password=password,
        query=query,
    )

    columns = ["subsystem", "aircraftregistration", "starttime"]
    table_instance = "oldinstance"
    table_name = "operationinterruption"

    query = f"SELECT {', '.join(columns)} FROM {table_instance}.{table_name}"

    operation_interruption = retrieve_dw_table(
        db_name="AMOS",
        session=spark,
        username=username,
        password=password,
        query=query,
    )

    # Create a matrix with all needed data
    training_data: DataFrame = get_training_data(
        sensor_data, operation_interruption, aircraft_utilization
    )

    # Save the matrix in a csv file
    output_path = "./results/training_data"
    training_data.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
