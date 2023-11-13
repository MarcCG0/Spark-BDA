import os
import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf


# Configure the filepath for the PostgreSQL driver
POSTGRESQL_DRIVER_PATH = "../postgresql-42.2.8.jar"  # Replace with your PostgreSQL driver path

# Create the Spark configuration
conf = SparkConf().set("spark.master", "local").set("spark.app.name", "DBALab").set("spark.jars", POSTGRESQL_DRIVER_PATH)

# Initialize the Spark Session
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Define the PostgreSQL database connection properties
db_properties = {
    "driver": "org.postgresql.Driver",
    "url": "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require",  # Replace with your database URL
    "user": "marc.camps.garreta",  # Replace with your database username
    "password": "DB180503"  # Replace with your database password
}

# Read data from the PostgreSQL database into a DataFrame
data = spark.read.jdbc(url=db_properties["url"],
                       table="public.aircraftutilization",
                       properties=db_properties)

# Data obtained from a database can be manipulated using SparkSQL’s operations
df = data.select("aircraftid", "flighthours", "delayedminutes")
df.show(30)

# Stop the Spark session when you’re done
spark.stop()
