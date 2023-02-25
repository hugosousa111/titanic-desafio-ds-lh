"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_classification_model, 
    generate_training_metrics
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_classification_model,
            inputs={
                "train_df": "train_dataset",
                "target": "params:target",
                "random_state": "params:random_state"
            },
            outputs="classification_model",
            name="train_classification_model"
        ),
        node(
            func=generate_training_metrics,
            inputs={
                "classification_model": "classification_model",
                "validation_df": "validation_dataset",
                "target": "params:target"
            },
            outputs="classification_metrics",
            name="generate_training_metrics"
        )
    ])
