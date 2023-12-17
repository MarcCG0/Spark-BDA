from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf

POSTGRESQL_DRIVER_PATH = "../postgresql-42.2.8.jar" 

def create_session() -> SparkSession: 
    """Initialize the SparkSession
    """
    conf: SparkConf = SparkConf().set("spark.master", "local").set("spark.app.name", "DBALab").set("spark.jars", POSTGRESQL_DRIVER_PATH)
    spark: SparkSession = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark

def retrieve_dw_table(db_name: str, table_instance: str, session : SparkSession,table_name:str, columns: list[str]) -> DataFrame: 
    """Given a database name, a table instance, a spark session,
      the name for the table you want to retrieve from the database
        and the columns you want to retrieve from that table
    """

    db_properties = {
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://postgresfib.fib.upc.edu:6433/{db_name}?sslmode=require",
        "user": "marc.camps.garreta", 
        "password": "DB180503" 
    }

    data = session.read.jdbc(url=db_properties["url"],
                        table= table_instance+"." + table_name,
                        properties=db_properties)

    df = data.select(columns)
    return df
