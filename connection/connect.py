import os
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf

def retrieve_dw_table(session : SparkSession,table_name:str, columns: list[str], db_name: str) -> DataFrame: 
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


def main(): 
    df = retrieve_dw_table("aircraftutilization", ["aircraftid", "flighthours", "delayedminutes"])
    df.show()


if __name__ == "__main__":
    main()
