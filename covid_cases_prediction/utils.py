#modules to use for prediction and evaluation
import random

from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
 
def train_and_adjust_prophet_model(data, grid_params):
    """
    entrenamiento y ajuste de modelo prophet. Utiliza gri search
    para ajuste de modelos, según diccionario de
    parámetros entregado en grid_params
    """

    #definición de grid en base a parámetros entregados
    grid = ParameterGrid(grid_params)
    best_model = None
    mape_best_model = 10000000000

    for p in grid:

        #definición de modelo base y entrenamiento
        random.seed(0)
        train_model =Prophet(
            n_changepoints = p['n_changepoints'],
            changepoint_prior_scale = p['changepoint_prior_scale'],
            interval_width=0.95,
            growth = 'logistic'
        )
        train_model.fit(data)

        #predicción para evaluación de modelo
        future = train_model.make_future_dataframe(0)
        future['cap'] = data['y'].max()
        future['floor'] = 0
        forecast_df = train_model.predict(future)

        #obtención de métrica de evaluación y selección de mejor modelo
        MAPE = mean_absolute_percentage_error(data['y'],abs(forecast_df['yhat']))
        if MAPE < mape_best_model:
           best_model = train_model
           mape_best_model = MAPE

    return (best_model, mape_best_model)
