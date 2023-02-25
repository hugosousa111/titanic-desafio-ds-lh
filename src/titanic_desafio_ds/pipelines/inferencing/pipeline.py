"""
This is a boilerplate pipeline 'inferencing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_prediction,
            inputs={
                "predictor": "titanic_classifier",
                "prediction_dataset": "test_dataset"
            },
            outputs="dataset_with_predictions",
        )
    ])
