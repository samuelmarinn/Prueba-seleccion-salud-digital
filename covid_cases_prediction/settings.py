#constants definitions

#urls de datos de casos covid
COVID_CASES_BASE_URL = 'https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto5'
NATIONAL_TOTALS_T_URL  = f'{COVID_CASES_BASE_URL}/TotalesNacionales_T.csv?raw=true'

#diccionario de parámetros de modelo a usar
PROPHET_PARAMS_GRID = {
    'changepoint_prior_scale':[0.1, 0.2, 0.3, 0.4],
    'n_changepoints' : [100, 150, 200, 250]
}