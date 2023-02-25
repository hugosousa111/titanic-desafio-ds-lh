"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.3
"""

import pandas as pd

from sklearn.metrics import classification_report
from sklearn import svm 


def train_classification_model(
    train_df:pd.DataFrame, target: str, random_state
):
    """Train model

    Args:
        train_df (pd.DataFrame): train dataset
        target (str): target column 
        random_state (_type_): random state training

    Returns:
        model: classification model trained
    """

    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]

    classification_model = svm.SVC(random_state=random_state)
    classification_model.fit(X_train, y_train)

    return classification_model

def generate_training_metrics(
    classification_model, validation_df: pd.DataFrame, target: str
):
    """Generate training metrics

    Args:
        classification_model (model): model trained
        validation_df (pd.DataFrame): validation dataset
        target (str): target column

    Returns:
        Dict: Dict with metrics to mlflow
    """
    X_test = validation_df.drop(target, axis=1)
    y_test = validation_df[target]

    y_pred = classification_model.predict(X_test)

    classification_metrics = classification_report(y_test, y_pred, output_dict=True)
    
    return {"accuracy": {"value": classification_metrics['accuracy'], "step": 1},
            "precision": {"value": classification_metrics['macro avg']['precision'], "step": 1},
            "recall": {"value": classification_metrics['macro avg']['recall'], "step": 1},
            "f1_score": {"value": classification_metrics['macro avg']['f1-score'], "step": 1},
            "support": {"value": classification_metrics['macro avg']['support'], "step": 1}
            }