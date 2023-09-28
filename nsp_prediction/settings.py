#constants and models settings definitions

NSP_DATA_URL = 'https://dcc.uchile.cl/~fvillena/files/nsp.xlsx'
TEST_SIZE = 0.2

LOG_REG_SPACE = {
    #'solver': ['sag', 'saga', 'liblinear'],
    'penalty': ['l1', 'l2'],
    'C': [ 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'class_weight': [{0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}]
}
