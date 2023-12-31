from data_management_pipeline import data_management_pipeline
from data_analysis_pipeline import data_analysis_pipeline
from runtime_classifier_pipeline import runtime_classifier_pipeline

from pyspark.sql import DataFrame, SparkSession
from pyspark import SparkConf
import os
from utils import Colors



# HADOOP_HOME = "./resources/hadoop_home"
# PYSPARK_PYTHON = "python3.9"
# PYSPARK_DRIVER_PYTHON = "python3.9"
POSTGRESQL_DRIVER_PATH = "./resources/postgresql-42.2.8.jar"



def create_session() -> SparkSession: 
    """Initialize the SparkSession
    """
    # os.environ["HADOOP_HOME"] = HADOOP_HOME
    # sys.path.append(HADOOP_HOME + "\\bin")
    # os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    # os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf: SparkConf = SparkConf().set("spark.master", "local").set("spark.app.name", "DBALab").set("spark.jars", POSTGRESQL_DRIVER_PATH)
    spark: SparkSession = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark

def set_env_variable(variable_name: str, variable_value:str):
    # Determine the shell profile file based on the user's shell
    shell = os.path.basename(os.environ.get('SHELL'))
    if shell == 'zsh':
        profile_path = os.path.expanduser("~/.zshrc")
    else:
        profile_path = os.path.expanduser("~/.bash_profile")

    # Append the export command to the profile file
    with open(profile_path, "a") as profile_file:
        profile_file.write(f"\nexport {variable_name}='{variable_value}'\n")


def main():

    variable_value = os.environ.get("USERNAME_BDA")

    if variable_value is None: 
        raise ValueError("USERNAME_BDA and PASSWORD_BDA environment variables must be specified before running the program.\n"
                         "Please modify the set.sh file with your credentials and run the 'source ./set.sh' command to set them automatically.")        

    username = os.environ['USERNAME_BDA']
    password = os.environ['PASSWORD_BDA']

    print(f"Running the program with username: {username} & password: {password}")

    # Create spark session 
    spark: SparkSession = create_session()

    data_path = "./results/training_data.csv"
    best_model_path = "./best_model"

    

    while True: 
        data_management_executed = os.path.exists(data_path)
        data_analysis_executed = os.path.exists(best_model_path)

        print(f"{Colors.BLUE}Choose the pipeline you want to execute:\n"
            " - Enter 'Data Management' for Data Management Pipeline\n"
            " - Enter 'Data Analysis' for Data Analysis Pipeline\n"
            " - Enter 'Run-Time Classifier' for Run-Time Classifier Pipeline\n"
            " - Enter 'All pipelines' for the whole process\n"
            " - Enter 'exit' to exit the program\n"
            f"Enter your choice: {Colors.RESET}")

        execution = input()
        
        if execution == "Data Management":

            print(f"{Colors.GREEN}Generating training data...{Colors.RESET}")
            data_management_pipeline(spark, username, password)
            print(f"{Colors.GREEN}Finished with the Data Analysis execution{Colors.RESET}")
            
        elif execution == "Data Analysis":

            if not data_management_executed:
                raise ValueError(f"{Colors.RED}Data Analysis pipeline requires Data Management pipeline to be executed first{Colors.RESET}")
            
            print(f"{Colors.GREEN}Training the models...{Colors.RESET}")
            data_analysis_pipeline(spark)
            print(f"{Colors.GREEN}Finished with the Data Analysis execution{Colors.RESET}")

        elif execution == "Run-Time Classifier":

            if not data_management_executed or not data_analysis_executed:
                raise ValueError(f"{Colors.RED}Run-Time Classifier pipeline requires Data Management and Data Analysis pipelines to be executed first{Colors.RESET}")
            
            print(f"{Colors.GREEN}Enter de new record to make the prediction{Colors.RESET}")
            aircraftid = input("aircraftid: ")
            date = input("date: ")
        
            print(f"{Colors.GREEN}Generating prediction...{Colors.RESET}")
            runtime_classifier_pipeline(spark, aircraftid, date, username, password)

            print(f"{Colors.GREEN}Finished with the Run-Time Classifier execution{Colors.RESET}")

        elif execution == "All pipelines":

            print(f"{Colors.GREEN}Enter de new record to make the prediction{Colors.RESET}")
            aircraftid = input("aircraftid: ")
            date = input("date: ")

            print(f"{Colors.GREEN}Generating training data...{Colors.RESET}")
            data_management_pipeline(spark, username, password)
            data_management_executed = True

            print(f"{Colors.GREEN}Training the models...{Colors.RESET}")
            data_analysis_pipeline(spark)
            data_analysis_executed = True

            print(f"{Colors.GREEN}Generating prediction...{Colors.RESET}")
            runtime_classifier_pipeline(spark, aircraftid, date, username, password)

            print(f"{Colors.GREEN}Finished with the whole execution{Colors.RESET}")
        elif execution == "exit":
            print(f"{Colors.GREEN}Program exited successfully{Colors.RESET}")
            break

        else:
            raise ValueError(f"{Colors.RED}Invalid choice. Please select a valid pipeline option{Colors.RESET}")
    
    

if __name__ == "__main__":
    main()