import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DateType
from pyspark.sql import DataFrame
import datetime
import re
from typing import List, Optional
from ..connection.connect import retrieve_dw_table


data_dir: str = '../data'

file_names: List[str] = os.listdir(data_dir)

print(file_names)

spark: SparkSession = SparkSession.builder.appName("ExtractInfo").getOrCreate()

df: DataFrame = spark.createDataFrame([(file_name,) for file_name in file_names], ["file_name"])

@udf(returnType=DateType())
def extract_datetime(file_name: str) -> Optional[datetime.date]:
    match = re.match(r'(\d{6})-(.*?)\.csv', file_name)
    if match:
        datetime_part: str = match.group(1)
        return datetime.datetime.strptime(datetime_part, "%y%m%d").date()
    return None

@udf(returnType=StringType())
def extract_aircraft_id(file_name: str) -> Optional[str]:
    match = re.match(r'(\d{6})-(.*?)\.csv', file_name)
    if match:
        return match.group(2)
    return None

df_result: DataFrame = df.withColumn("datetime", extract_datetime("file_name")).withColumn("aircraft_id", extract_aircraft_id("file_name"))

# Show the result
df_result.show(truncate=False)

spark.stop()
