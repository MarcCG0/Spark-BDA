from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import DataFrame
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, GBTClassifier, LinearSVC, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
from typing import Tuple

def train(train_data: DataFrame, validation_data: DataFrame) -> Tuple[str, float, PipelineModel | None]:
    best_model_name = ""
    best_f1_score = 0.0
    best_model = None

    indexer = StringIndexer(inputCol="operation_interruption", outputCol="label")
    assembler = VectorAssembler(inputCols=["sensor_mean_value", "flighthours", "delayedminutes", "flightcycles"], outputCol="features")
    
    # Initialize classifiers
    classifiers = [
        ("Decision Tree", DecisionTreeClassifier(labelCol="label", featuresCol="features")),
        ("Random Forest", RandomForestClassifier(labelCol="label", featuresCol="features")),
        ("Logistic Regression", LogisticRegression(labelCol="label", featuresCol="features")),
        ("Gradient-Boosted Trees", GBTClassifier(labelCol="label", featuresCol="features")),
        ("Linear SVC", LinearSVC(labelCol="label", featuresCol="features")),
        ("Naive Bayes", NaiveBayes(labelCol="label", featuresCol="features"))
    ]

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    # Train and evaluate models
    for name, classifier in classifiers:
        pipeline = Pipeline(stages=[indexer, assembler, classifier])
        model = pipeline.fit(train_data)
        predictions = model.transform(validation_data)
        f1_score = evaluator.evaluate(predictions)
        
        print(f"{name} F1-Score: {f1_score}")

        # Log parameters, metrics, and model
        mlflow.log_param(f"{name} parameters", str(model.stages[-1].extractParamMap()))
        mlflow.log_metric(f"{name} F1-Score", f1_score)
        mlflow.spark.log_model(model, f"{name} model")

        mlflow.spark.save_model(model, f"../models/{name}")

        # Update best model if necessary
        if f1_score > best_f1_score:
            best_model_name = name
            best_f1_score = f1_score
            best_model = model

    mlflow.spark.save_model(best_model, f"../best_model")

    return best_model_name, best_f1_score, best_model
