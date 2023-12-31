from data_management_pipeline import data_management_pipeline
from data_analysis_pipeline import data_analysis_pipeline
from runtime_classifier_pipeline import runtime_classifier_pipeline

from pyspark.sql import DataFrame, SparkSession
from pyspark import SparkConf
import argparse
import os
import sys



HADOOP_HOME = "./Codeskeleton/resources/hadoop_home"
PYSPARK_PYTHON = "python3.9"
PYSPARK_DRIVER_PYTHON = "python3.9"
POSTGRESQL_DRIVER_PATH = "./Codeskeleton/resources/postgresql-42.2.8.jar"



def create_session() -> SparkSession: 
    """Initialize the SparkSession
    """
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf: SparkConf = SparkConf().set("spark.master", "local").set("spark.app.name", "DBALab").set("spark.jars", POSTGRESQL_DRIVER_PATH)
    spark: SparkSession = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark



def main():

    # Create spark session 
    spark: SparkSession = create_session()

    print("Choose the pipeline you want to execute:\n"
         " - 'Data Management' for Data Management Pipeline\n"
         " - 'Data Analysis' for Data Analysis Pipeline\n"
         " - 'Run-Time Classifier' for Run-Time Classifier Pipeline\n"
         " - 'All pipelines' for the whole process")

    

    print("Generating training data and modeling" if not only_predict else "Loading training data")

    # Execute Data Management Pipeline
    aircraft_utilization, sensor_data = data_management_pipeline(spark, username, password)

    # Execute Data Analysis Pipeline
    data_analysis_pipeline(spark)

    # executar run-time classifier
    runtime_classifier_pipeline(spark, aircraftid, date, username, password)


if __name__ == "__main__":
    main()