from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.functions import input_file_name, regexp_extract, to_date, mean, datediff, when, col, expr

from  connection.connect import retrieve_dw_table, create_session
from data_management import retrieve_needed_tables, get_full_csv_concatenation, get_training_data



def main(): 
    spark = create_session()
    aircraft_utilization, operation_interruption = retrieve_needed_tables(spark)
    sensor_data: DataFrame = get_full_csv_concatenation(spark)
    training_data: DataFrame = get_training_data(sensor_data, operation_interruption, aircraft_utilization)
    training_data.show()
    output_path = "../results/training_data.csv"
    training_data.coalesce(1).write.csv(output_path, header=True, mode="overwrite")





if __name__ == "__main__":
    main()
