
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.functions import input_file_name, regexp_extract, to_date, mean, datediff, when, col, expr

from  connection.connect import retrieve_dw_table, create_session


def get_full_csv_concatenation(session: SparkSession) -> DataFrame:
    path_to_csv_files = "../raw_data/*.csv"  
    df = session.read.csv(path_to_csv_files, header=True, inferSchema=True, sep=";")
    pattern = r'.*\/([^\/]*)\.csv$'
    df = df.withColumn("date", to_date(df["date"]))
    df = df.withColumn("aircraftid", regexp_extract(input_file_name(), pattern, 1).substr(-6, 6))
    df_mean = df.groupBy("date", "aircraftid").agg(mean("value").alias("mean_value"))
    return df_mean


def retrieve_needed_tables(spark: SparkSession) -> tuple[DataFrame, DataFrame]:

    aircraft_utilization = retrieve_dw_table("DW", "public",spark,"aircraftutilization", ["timeid","aircraftid", "flighthours", "delayedminutes", "flightcycles"])
    operation_interruption = retrieve_dw_table("AMOS", "oldinstance", spark,"operationinterruption", ["subsystem","aircraftregistration", "starttime"])

    return aircraft_utilization, operation_interruption
    

def get_training_data(sensor_data: DataFrame, operation_interruption: DataFrame, aircraft_utilization: DataFrame)-> DataFrame:

    joined_df = sensor_data.join(
        aircraft_utilization,
        (sensor_data.date == aircraft_utilization.timeid) & (sensor_data.aircraftid == aircraft_utilization.aircraftid),
        how='inner'
    )
    joined_df = joined_df.drop(aircraft_utilization.aircraftid)

    operation_interruption = operation_interruption.withColumn("starttime", to_date(operation_interruption.starttime))
    operation_interruption = operation_interruption.filter(operation_interruption.subsystem == 3453)
    operation_interruption.show()

    condition = (joined_df.aircraftid == operation_interruption.aircraftregistration) & \
            (datediff(operation_interruption.starttime, joined_df.date) >= 0) & \
            (datediff(operation_interruption.starttime, joined_df.date) <= 7)

    joined_df_with_interruption = joined_df.join(operation_interruption, condition, "left_outer")

    joined_df_with_interruption = joined_df_with_interruption.withColumn("operation_interruption", (col("starttime").isNotNull()).cast("boolean"))

    joined_df_with_interruption = joined_df_with_interruption.drop("starttime", "subsystem", "aircraftregistration", "timeid")

    return joined_df_with_interruption
