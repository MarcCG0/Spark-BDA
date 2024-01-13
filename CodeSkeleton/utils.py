from typing import List 
from pyspark.sql import DataFrame, SparkSession

###############################################################################
#
# Utils:
#
#   - This file contains elements that are used all over the project, that is utils. 
#
###############################################################################

class Colors:
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



