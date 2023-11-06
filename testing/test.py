import pyspark
spark = pyspark.sql.SparkSession.builder.appName('configurationtest').getOrCreate()
print(spark.range(5).collect())
