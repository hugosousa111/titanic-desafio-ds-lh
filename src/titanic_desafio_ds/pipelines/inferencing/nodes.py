"""
This is a boilerplate pipeline 'inferencing'
generated using Kedro 0.18.3
"""

import pandas as pd

def generate_prediction(
    predictor, 
    prediction_dataset: pd.DataFrame
):
    """ Node generates a prediction on a dataset 
    
    Args:
        predictor: Pickle model of the encapsulated serving pipeline.
        prediction_dataset: Dataset with the same structure as the one passed by kedro fast-api

    Returns: 
        pd.DataFrame: prediction dataset
    """

    predictions = predictor.predict(
        args_API=prediction_dataset,
        context=None # Context is only necessary for predictions generated through API requests
    )

    prediction_dataset["prediction"] = predictions["prediction"]

    return prediction_dataset
