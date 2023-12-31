from data_management_pipeline import data_management_pipeline
from data_analysis_pipeline import data_analysis_pipeline
from runtime_classifier_pipeline import runtime_classifier_pipeline

from pyspark.sql import DataFrame, SparkSession
from pyspark import SparkConf
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
    
    print("Enter your user credentials")
    username = input("username: ")
    password = input("password: ")
    # Create spark session 
    spark: SparkSession = create_session()

    data_management_executed = False
    data_analysis_executed = False

    print("Choose the pipeline you want to execute:\n"
         " - Enter 'Data Management' for Data Management Pipeline\n"
         " - Enter 'Data Analysis' for Data Analysis Pipeline\n"
         " - Enter 'Run-Time Classifier' for Run-Time Classifier Pipeline\n"
         " - Enter 'All pipelines' for the whole process\n"
         "Enter your choice: ")

    execution = input()
    
    if execution == "Data Management":

        print("Generating training data...")
        data_management_pipeline(spark, username, password)
        data_management_executed = True
        
    elif execution == "Data Analysis":

        if not data_management_executed:
            raise ValueError("Data Analysis pipeline requires Data Management pipeline to be executed first")
        
        print("Training the models...")
        data_analysis_pipeline(spark)
        data_analysis_executed = True

    elif execution == "Run-Time Classifier":

        if not data_management_executed or not data_analysis_executed:
            raise ValueError("Run-Time Classifier pipeline requires Data Management and Data Analysis pipelines to be executed first")
         
        print("Enter de new record to make the prediction")
        aircraftid = input("aircraftid: ")
        date = input("date: ")
    
        print("Generating prediction...")
        runtime_classifier_pipeline(spark, aircraftid, date, username, password)

    elif execution == "All pipelines":

        print("Enter de new record to make the prediction")
        aircraftid = input("aircraftid: ")
        date = input("date: ")

        print("Generating training data...")
        data_management_pipeline(spark, username, password)
        data_management_executed = True

        print("Training the models...")
        data_analysis_pipeline(spark)
        data_analysis_executed = True

        print("Generating prediction...")
        runtime_classifier_pipeline(spark, aircraftid, date, username, password)

    else:
        raise ValueError("Invalida choice. Please select a valid pipeline option")
    
    

if __name__ == "__main__":
    main()