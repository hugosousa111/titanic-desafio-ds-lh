import pandas as pd


class MLPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, args_API: pd.DataFrame, context):
        df_args = args_API
        prediction = self.model.predict(df_args)
        prediction = prediction.tolist()
        return {"prediction": prediction}


def save_predictor(model):
    predictor = MLPredictor(model)
    return predictor
