from pyspark.sql import SparkSession, Row
from datetime import date, datetime


# Initialize a SparkSession
spark = SparkSession.builder.appName("create_dataframe").getOrCreate()

# Create a DataFrame with the specified rows and schema
df = spark.createDataFrame([
    Row(a=1, b=2.0, c="string1", d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3.0, c="string2", d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5.0, c="string3", d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])

# Show the DataFrame
df.show(4, vertical=True)
# Stop the SparkSession
spark.stop()

