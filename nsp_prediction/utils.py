#modules to use for nsp prediction
from typing import List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, fbeta_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder


f2_score = make_scorer(fbeta_score, beta=1.1, pos_label=1)


def preprocess_features(X_df: pd.DataFrame):
    """Pre-rpoceamiento a usar para entrenamiento de modelos"""

    num_cols = X_df.select_dtypes(include=np.number).columns
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns

    transformer_numerico = StandardScaler()
    transformer_categorico = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformer_numerico, num_cols),
            ('cat', transformer_categorico, cat_cols)
        ])
    
    return preprocessor.fit_transform(X_df)


class BaseClassificationModel:
    """Clase base definida para entrenamiento, ajuste y predicción
        de modelos a usar"""

    def __init__(self, model_def):
        self.model = model_def

    def train_adjust(self, X, Y, grid_params):
        """
        entrenamiento y ajuste de modelo. Utiliza GridSearchCV
        para ajuste de modelos, según diccionario de
        parámetros entregado en grid_params
        """

        random_model =  GridSearchCV(
            self.model, grid_params, scoring=f2_score, n_jobs=-1, cv=2, verbose=2
        )

        random_model.fit(X, Y)
        self.model = random_model.best_estimator_

    def predict(self, features):
        """predicción según features entregadas"""

        pred = self.model.predict(features)
        return pred


def weigthed_score(pred, true_val):
    """promedio ponderado entre f-bta y ROC AUC"""

    f_beta = fbeta_score(true_val, pred, beta=1.1, pos_label=1)
    roc_auc = roc_auc_score(true_val, pred)

    score = (f_beta*0.6) + (roc_auc*0.4)

    return score


def select_best_model(models_list, features, true_val):
    """selección de mejor modelo según mayor weighted_socre"""

    best_score = -1
    best_model = None

    for model in models_list:

        pred = model.predict(features)
        score = weigthed_score(pred, true_val)

        if score > best_score:

            best_score = score
            best_model = model

    return [best_model, best_score]
