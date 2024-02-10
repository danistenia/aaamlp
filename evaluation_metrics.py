''' Implementation of evaluation metrics for ML models - Recreating what scikit-learn does. '''

import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(y_true: list, y_pred: list) -> float:
    '''
    Función creada a mano para calcular el accuracy.

    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve el calculo de la métrica de accuracy
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"
    return (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)

if __name__ == '__main__':

    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 1]
    print('Accuracy calculado a mano:', accuracy(y_true=y_true, y_pred=y_pred))
    print('Accuracy calculado con scikit learn:', accuracy(y_true=y_true, y_pred=y_pred))
