"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from titanic_desafio_ds.pipelines import (
    feature_engineering as fe,
    training,
    serving,
    inferencing
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    feature_engineering_pipeline = fe.create_pipeline()
    training_pipeline = training.create_pipeline()
    serving_pipeline = serving.create_pipeline()
    inferencing_pipeline = inferencing.create_pipeline()

    return {
        "feature_engineer": feature_engineering_pipeline,
        "train": training_pipeline,
        "serve": serving_pipeline,
        "inference": inferencing_pipeline, 
        "__default__": (
            feature_engineering_pipeline
            + training_pipeline
            + serving_pipeline
            + inferencing_pipeline
        )
    }
