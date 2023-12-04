from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.functions import input_file_name, regexp_extract, to_date, mean
#from  ..connection.connect import retrieve_dw_table



POSTGRESQL_DRIVER_PATH = "../postgresql-42.2.8.jar" 


def retrieve_dw_table(db_name: str, session : SparkSession,table_name:str, columns: list[str]) -> DataFrame: 
    # Configure the filepath for the PostgreSQL driver
    POSTGRESQL_DRIVER_PATH = "../postgresql-42.2.8.jar"  # Replace with your PostgreSQL driver path

    

    # Define the PostgreSQL database connection properties
    db_properties = {
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://postgresfib.fib.upc.edu:6433/{db_name}?sslmode=require",  # Replace with your database URL
        "user": "marc.camps.garreta",  # Replace with your database username
        "password": "DB180503"  # Replace with your database password
    }

    # Read data from the PostgreSQL database into a DataFrame
    data = session.read.jdbc(url=db_properties["url"],
                        table="public." + table_name,
                        properties=db_properties)

    # Data obtained from a database can be manipulated using SparkSQL’s operations
    df = data.select(columns)
    return df

def get_full_csv_concatenation(session: SparkSession) -> DataFrame:
    path_to_csv_files = "../data/*.csv"  
    df = session.read.csv(path_to_csv_files, header=True, inferSchema=True, sep=";")
    pattern = r'.*\/([^\/]*)\.csv$'
    df = df.withColumn("date", to_date(df["date"]))
    df = df.withColumn("aircraftid", regexp_extract(input_file_name(), pattern, 1).substr(-6, 6))
    df_mean = df.groupBy("date", "aircraftid").agg(mean("value").alias("mean_value"))
    return df_mean

#def get_aircraft_registration() -> DataFrame: 


def show_spark_dataframe(df: DataFrame): 
    unique_dates = df.select("").distinct()
    unique_dates.show(n=unique_dates.count(), truncate=False)


def main(): 
    conf = SparkConf().set("spark.master", "local").set("spark.app.name", "DBALab").set("spark.jars", POSTGRESQL_DRIVER_PATH)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sensor_data: DataFrame = get_full_csv_concatenation(spark)
    aircraft_utilization = retrieve_dw_table("DW",spark,"aircraftutilization", ["timeid","aircraftid", "flighthours", "delayedminutes", "flightcycles"])

    joined_df = sensor_data.join(
        aircraft_utilization,
        (sensor_data.date == aircraft_utilization.timeid) & (sensor_data.aircraftid == aircraft_utilization.aircraftid),
        how='inner'
    )
    # joined_df = joined_df.drop(aircraft_utilization.timeid)
    joined_df.show()

    operation_interruption = retrieve_dw_table("AMOS",spark,"operationinterruption", ["subsystem","aircraftregistration", "starttime"])
    operation_interruption = operation_interruption.withColumn("starttime", to_date(operation_interruption.starttime))
    operation_interruption = operation_interruption.filter(operation_interruption.subsystem == 3453)
    operation_interruption.show()



if __name__ == "__main__":
    main()
