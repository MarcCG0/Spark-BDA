
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col
import os

# Initialize a Spark session
spark = SparkSession.builder.appName("ProcessFileNames").getOrCreate()

# Path to the 'data' directory
data_dir = '../data'  # Update this path as needed

# Use Hadoop FileSystem API to list files in the directory
path = os.path.join(data_dir, '*')
file_names_rdd = spark.sparkContext.wholeTextFiles(path).keys()

# Convert RDD to DataFrame
file_names_df = file_names_rdd.toDF(["file_path"])

# Extract file name from the full file path
file_names_df = file_names_df.withColumn("file_name", split(col("file_path"), "/").getItem(-1))

# Split the file name and extract the required parts
split_col = split(file_names_df['file_name'], '-')
file_names_df = file_names_df.withColumn('day', split_col.getItem(0))
file_names_df = file_names_df.withColumn('aircraft_id', split_col.getItem(4) + "-" + split_col.getItem(5))

# Select the relevant columns and show results
result_df = file_names_df.select('day', 'aircraft_id')
result_df.show()

# Stop the Spark session
spark.stop()
