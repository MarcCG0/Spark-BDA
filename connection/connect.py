import os
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf

POSTGRESQL_DRIVER_PATH = "../postgresql-42.2.8.jar" 

def create_session() -> SparkSession: 
    
    conf = SparkConf().set("spark.master", "local").set("spark.app.name", "DBALab").set("spark.jars", POSTGRESQL_DRIVER_PATH)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark

def retrieve_dw_table(db_name: str, table_instance: str, session : SparkSession,table_name:str, columns: list[str]) -> DataFrame: 

    db_properties = {
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://postgresfib.fib.upc.edu:6433/{db_name}?sslmode=require",  # Replace with your database URL
        "user": "marc.camps.garreta",  # Replace with your database username
        "password": "DB180503"  # Replace with your database password
    }

    data = session.read.jdbc(url=db_properties["url"],
                        table= table_instance+"." + table_name,
                        properties=db_properties)

    df = data.select(columns)
    return df
