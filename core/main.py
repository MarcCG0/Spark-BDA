from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import input_file_name, regexp_extract, to_date, mean


def get_full_csv_concatenation() -> DataFrame:
    spark = SparkSession.builder.appName("CSV Concatenation").getOrCreate()
    path_to_csv_files = "../data/*.csv"  
    df = spark.read.csv(path_to_csv_files, header=True, inferSchema=True, sep=";")
    pattern = r'.*\/([^\/]*)\.csv$'
    df = df.withColumn("date", to_date(df["date"]))
    df = df.withColumn("aircraftid", regexp_extract(input_file_name(), pattern, 1).substr(-6, 6))
    df_mean = df.groupBy("date", "aircraftid").agg(mean("value").alias("mean_value"))
    return df_mean

def show_spark_dataframe(df: DataFrame): 
    unique_dates = df.select("").distinct()
    unique_dates.show(n=unique_dates.count(), truncate=False)


def main(): 
    df: DataFrame = get_full_csv_concatenation()
    df.show()

if __name__ == "__main__":
    main()
