from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, to_date

###############################################################################
#
# Utils:
#
#   - This file contains elements that are used all over the project, that is utils.
#
###############################################################################


class Colors:
    """
    Class that represents the colors for the logs
    of the project.
    """

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"


def retrieve_dw_table(
    db_name: str,
    session: SparkSession,
    username: str,
    password: str,
    query: str,
) -> DataFrame:
    """
    Given a database name, a table instance, a spark session, the name for the table you want to retrieve from the database,
    the columns you want to retrieve from that table, and an optional SQL query, retrieves the selected rows for performing the
    data management to obtain our matrix for training.
    """

    db_properties = {
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://postgresfib.fib.upc.edu:6433/{db_name}?sslmode=require",
        "user": username,
        "password": password,
    }

    try:
        table = f"({query}) AS temp_table"
        data = session.read.jdbc(
            url=db_properties["url"],
            table=table,
            properties=db_properties,
        )
    except Exception:
        error_message = f"{Colors.RED}Error occurred while executing JDBC query, check the error provided by Spark.{Colors.RESET}"
        raise Exception(error_message)

    return data


def concatenate_csv_sensor_data(session: SparkSession) -> DataFrame:
    """
    This function concatenates and parses the csv's containing the
    sensor data in order to proceed with averaging its values in
    posterior steps.
    Note: the value for aircraft id is parsed directly from the csv
    file name.
    """
    path_to_csv_files = "./resources/trainingData/*.csv"
    df = session.read.csv(path_to_csv_files, header=True, inferSchema=True, sep=";")
    pattern = r".*\/([^\/]*)\.csv$"
    df = df.withColumn("date", to_date(df["date"]))
    df = df.withColumn(
        "aircraftid",
        regexp_extract(input_file_name(), pattern, 1).substr(-6, 6),
    )
    return df
