from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
import os, sys

HADOOP_HOME = "./Codeskeleton/resources/hadoop_home"
PYSPARK_PYTHON = "python3.9"
PYSPARK_DRIVER_PYTHON = "python3.9"
POSTGRESQL_DRIVER_PATH = "../../postgresql-42.2.8.jar"

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

def retrieve_dw_table(db_name: str, table_instance: str, session : SparkSession, table_name:str, columns: list[str], username:str, password:str) -> DataFrame: 
    """Given a database name, a table instance, a spark session,
      the name for the table you want to retrieve from the database
        and the columns you want to retrieve from that table
    """

    db_properties = {
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://postgresfib.fib.upc.edu:6433/{db_name}?sslmode=require",
        "user": username, 
        "password": password 
    }

    data = session.read.jdbc(url=db_properties["url"],
                        table= table_instance+"." + table_name,
                        properties=db_properties)

    df = data.select(columns)
    return df
