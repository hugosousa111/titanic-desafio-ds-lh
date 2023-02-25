"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    fill_nan_values,
    ordinal_encoder_categorical_variables,
    feature_and_target_selection,
    create_feature_category,
    data_split,
    feature_engineering_dataset_test
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fill_nan_values,
            inputs={
                "df": "titanic_train_dataset",
                "features": "params:features"
            },
            outputs="df_filled",
            name="fill_nan_values",
        ),
        node(
            func=ordinal_encoder_categorical_variables,
            inputs={
                "df": "df_filled",
                "categorical_variables": "params:categorical_variables",
            },
            outputs=[
                "df_transformed_categorical_variables",
                "ordinal_encoder_transformer",
            ],
            name="ordinal_encoder_categorical_variables",
        ),
        node(
            func=feature_and_target_selection,
            inputs={
                "df": "df_transformed_categorical_variables",
                "target": "params:target", 
                "features": "params:features"
            },
            outputs="df_select",
            name="feature_and_target_selection"
        ),
        node(
            func=create_feature_category,
            inputs={
                "df": "df_select",
                "target": "params:target", 
                "features": "params:features"
            },
            outputs="X_y_df",
            name="create_feature_category"
        ),
        node(
            func=data_split,
            inputs={
                "df": "X_y_df",
                "split_proportion": "params:validation_split_proportion",
                "random_state": "params:validation_random_state"
            },
            outputs=[
                "train_dataset",
                "validation_dataset"
            ]
        ), 
        node(
            func=feature_engineering_dataset_test,
            inputs={
                "df": "titanic_train_dataset",
                "df_test": "titanic_test_dataset",
                "features": "params:features",
                "oe": "ordinal_encoder_transformer",
                "categorical_variables": "params:categorical_variables",
            },
            outputs="test_dataset",
            name="feature_engineering_dataset_test",
        )
    ])
