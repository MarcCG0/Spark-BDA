from connect import create_session
from data_management import retrieve_needed_tables, get_sensor_data, get_training_data
from train_models import train
from pyspark.ml import PipelineModel

from typing import Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

    
import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

import mlflow
import argparse

def compute_whole_process(spark: SparkSession) -> Tuple[DataFrame,PipelineModel | None, str]:
    """When only_predict set to False then compute the whole process: 
        1. Data management pipeline 
        2. Data analysis pipeline

        Otherwise, just load previously calculated values and use them to predict the input tuple.
    """
    aircraft_utilization, operation_interruption = retrieve_needed_tables(spark)
    sensor_data: DataFrame = get_sensor_data(spark)
    training_data: DataFrame = get_training_data(sensor_data, operation_interruption, aircraft_utilization).orderBy("aircraftid", "date")
    output_path = "../results/training_data.csv"
    training_data.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
    (train_data, validation_data, test_data) = training_data.randomSplit([0.6, 0.20, 0.20], seed=123)   #TODO: check if test data is needed
    with mlflow.start_run(run_name="Model Training Run"):
        best_model_name, _, best_model = train(train_data, validation_data)    
        mlflow.end_run()

    return training_data, best_model, best_model_name

def main():

    HADOOP_HOME = "./workspaces/Spark-BDA/execucio/CodeSkeleton/resources/hadoop_home"
    JDBC_JAR = "./workspaces/Spark-BDA/execucio/CodeSkeleton/resources/postgresql-42.2.8.jar"
    PYSPARK_PYTHON = "python3.6"
    PYSPARK_DRIVER_PYTHON = "python3.6"

    if(__name__== "__main__"):
        os.environ["HADOOP_HOME"] = HADOOP_HOME
        sys.path.append(HADOOP_HOME + "\\bin")
        os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
        os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

        conf = SparkConf()  # create the configuration
        conf.set("spark.master", "local")
        conf.set("spark.app.name","DBALab")
        conf.set("spark.jars", JDBC_JAR)

        # Initialize a Spark session
        spark = SparkSession.builder \
        .config(conf=conf).getOrCreate()

        #Create and point to your pipelines here




    spark: SparkSession = create_session()
    parser = argparse.ArgumentParser(description="Your script description")

    parser.add_argument("--aircraftid", type=str, required=True, help="Aircraft ID as a string eg. XY-BDC")
    parser.add_argument("--day", type=str, required=True, help="Day as a string in format yyyy-MM-dd")
    parser.add_argument("--only_predict", type=str, help= "If True, only predicts over the previously obtained models, otherwise it computes the whole process of data management pipeline and data analysis pipeline")

    args = parser.parse_args()

    aircraftid: str = args.aircraftid
    day: str = args.day

    only_predict: bool = True if args.only_predict == "True" else False

    print("Generating training data and modeling" if not only_predict else "Loading training data")

    if only_predict: 
        training_data = spark.read.csv("../results/training_data.csv", header=True, inferSchema=True)
        model_path = "../best_model"
        best_model = mlflow.spark.load_model(model_path)
        best_model_name = best_model.stages[-1].uid.split("_")[0]
    else: 
        training_data, best_model, best_model_name = compute_whole_process(spark)

    tuple_to_predict = training_data.filter((col("aircraftid") == aircraftid) & (col("date") == day))

    predictions = best_model.transform(tuple_to_predict)
    prediction_value = predictions.select('prediction').collect()[0][0]

    print(f"Using {best_model_name} for predicting tuple [{aircraftid}, {day}, {tuple_to_predict.select('sensor_mean_value').collect()[0][0]}, {tuple_to_predict.select('flighthours').collect()[0][0]}, {tuple_to_predict.select('delayedminutes').collect()[0][0]}, {tuple_to_predict.select('flightcycles').collect()[0][0]}]")


    print("No maintenance predicted for the next 7 days" if prediction_value == 0 else "Unscheduled maintenance predicted in the next 7 days")

if __name__ == "__main__":
    main()

