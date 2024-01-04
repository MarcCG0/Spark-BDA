from typing import Any, List, Optional, Tuple

import mlflow
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from utils import Colors

###############################################################################
#
# Data Analysis Pipeline:
#
#   - Read matrix created in data management pipeline
#   - Prepare data to train the models. That includes:
#           - Split data into train and validation sets
#           - Define classifier models and evalutation function
#           - Index the labels and assemble the feataures
#   - Train the models. That includes, for each model:
#           - Fit the model
#           - Evaluate predictions
#           - Store the model (name, parameters and metrics)
#   - Rank all the models by F1-Score
#   - Save the best model
#
###############################################################################


def def_classifiers_and_evaluator() -> (
    Tuple[List[Tuple[Any, Any]], MulticlassClassificationEvaluator]
):
    """Initialize the classifiers and evaluation function that will be used."""

    # Define and initialize classifiers
    classifiers = [
        (
            "Decision Tree",
            DecisionTreeClassifier(labelCol="label", featuresCol="features"),
            {
                "maxDepth": [5, 10, 20],
                "maxBins": [20, 40, 60, 120]
            }
        ),
        (
            "Random Forest",
            RandomForestClassifier(labelCol="label", featuresCol="features"),
            {
                "numTrees": [10, 20, 30],
                "maxDepth": [5, 10, 15]
            }
        ),
        (
            "Logistic Regression",
            LogisticRegression(labelCol="label", featuresCol="features"),
            {
                "maxIter": [10, 50, 100],
                "regParam": [0.01, 0.1, 0.5]
            }
        ),
        (
            "Gradient-Boosted Trees",
            GBTClassifier(labelCol="label", featuresCol="features"),
            {
                "maxIter": [10, 20, 30],
                "maxDepth": [3, 5, 7]
            }
        ),
        (
            "Linear SVC", 
            LinearSVC(labelCol="label", featuresCol="features"),
            {
                "maxIter": [10, 50, 100],
                "regParam": [0.01, 0.1, 0.5]
            }
        ),
        (
            "Naive Bayes",
            NaiveBayes(labelCol="label", featuresCol="features"),
            {
                "smoothing": [0.0, 1.0, 2.0]
            }
        )
    ]

    # Define and initialize evulation function
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    return classifiers, evaluator


def train(
    spark: SparkSession,
    train_data: DataFrame,
    validation_data: DataFrame,
    classifiers: List[Tuple[Any, Any]],
    evaluator: MulticlassClassificationEvaluator,
) -> Tuple[DataFrame, Optional[PipelineModel]]:
    """Train different classification models to find the one that better fits the matrix data."""

    schema = StructType([
                        StructField("Model", StringType(), True),
                        StructField("F1-Score", DoubleType(), True)
                        ])

    models: DataFrame = spark.createDataFrame([], schema=schema)

    best_f1_score = 0.0
    best_model = None

    # Define indexer and assembler
    assembler = VectorAssembler(
        inputCols=[
            "sensor_mean_value",
            "flighthours",
            "delayedminutes",
            "flightcycles",
        ],
        outputCol="features",
    )

    # Train and evaluate models
    for name, classifier, param_grid in classifiers:
        # Index the labels and assemble the features (convert them into a single vector)
        pipeline = Pipeline(stages=[assembler, classifier])

        # Define a parameter grid to search over
        grid = ParamGridBuilder()
        for param, values in param_grid.items():
            grid.addGrid(classifier.getParam(param), values)

        grid = grid.build()
        # Cross validator
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=grid,
                                  evaluator=evaluator,
                                  numFolds=3) 

        # Run cross-validation, and choose the best set of parameters
        cvmodel = crossval.fit(train_data)
        model = cvmodel.bestModel

        # Evaluate the model with validation data
        predictions = model.transform(validation_data)
        f1_score = evaluator.evaluate(predictions)
        # accuracy = evaluator.evaluate(predictions, {evaluator.metricName:"accuracy"})

        print(f"{Colors.GREEN}{name} trained{Colors.RESET}")

        # Store the model and the scores in a dataframe
        models = models.union(spark.createDataFrame(
            [(name, f1_score)], schema=schema))

        # Log parameters, metrics, and model
        mlflow.log_param(f"{name} parameters", str(
            model.stages[-1].extractParamMap()))
        mlflow.log_metric(f"{name} F1-Score", f1_score)
        mlflow.spark.log_model(model, f"{name} model")

        mlflow.spark.save_model(model, f"./models/{name}")

        # Update best model if necessary
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model = model

    return models, best_model


###############################
# Data Analysis Main Function #
###############################


def data_analysis_pipeline(spark: SparkSession):
    """Compute all the data analysis pipeline."""

    # Read the matrix computed in the data management pipeline
    training_data = spark.read.csv(
        "./results/training_data.csv", header=True, inferSchema=True
    )
    training_data = training_data.drop("date", "aircraftid")

    # Split data from the matrix into train and validation sets
    (train_data, validation_data) = training_data.randomSplit(
        [0.7, 0.3], seed=123)

    # Define classification models and evaluation function that will be used
    classifiers, evaluator = def_classifiers_and_evaluator()

    # Train different models to find the best one and store them in a folder
    with mlflow.start_run(run_name="Model Training Run"):
        models, best_model = train(
            spark, train_data, validation_data, classifiers, evaluator
        )
        mlflow.end_run()

    # Rank the models by f1-score
    models = models.orderBy("F1-Score", ascending=False)
    models.show()

    # Save the best model
    mlflow.spark.save_model(best_model, "./best_model")
