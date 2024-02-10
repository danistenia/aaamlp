''' Implementation of evaluation metrics for ML models - Recreating what scikit-learn does. '''

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

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

def true_positives(y_true: list, y_pred: list) -> int:
    '''
    Función para contar los true positives. Recordar que los TP son las veces en que tu modelo dice que es un 1 y efectivamente es un 1.

    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad de true positives.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"

    return np.array([x == y for x, y in zip(y_pred, y_true) if x == 1]).sum()

def true_negatives(y_true: list, y_pred: list) -> int:
    '''
    Función para contar los true negatives. Recordar que los TN son las veces en que tu modelo dice que es un 0 y efectivamente es un 0.

    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad de true negatives.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"

    return np.array([x == y for x, y in zip(y_pred, y_true) if x == 0]).sum()


def false_positives(y_true: list, y_pred: list) -> int:
    '''
    Función para contar los false positives. Recordar que los FP son las veces en que tu modelo dice que es un 1 y no es un 1, es un 0.

    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad de false positives.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"

    return np.array([x != y for x, y in zip(y_pred, y_true) if x == 1]).sum()


def false_negatives(y_true: list, y_pred: list) -> int:
    '''
    Función para contar los false negatives. Recordar que los FN son las veces es que tu modelo dice que es un 0 y es un 1.

    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad de false negatives.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"

    return np.array([x != y for x, y in zip(y_pred, y_true) if x == 0]).sum()

def precision(y_true: list, y_pred: list) -> int:
    '''
    Función que calcula el precision. Recordar que el precision es el porcentaje de veces que tu modelo dice que es un 1 y efectivamente es un 1.
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad la precision.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"
    precision = true_positives(y_true=y_true, y_pred=y_pred) / (true_positives(y_true=y_true, y_pred=y_pred) + false_positives(y_true=y_true, y_pred=y_pred))
    return precision

def recall(y_true: list, y_pred: list) -> int:
    '''
    Función que calcula el recall. Recordar que el recall es el porcentaje de aciertos de 1's sobre todo el universo de 1's. 
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad la precision.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"
    precision = true_positives(y_true=y_true, y_pred=y_pred) / (true_positives(y_true=y_true, y_pred=y_pred) + false_negatives(y_true=y_true, y_pred=y_pred))
    return precision


if __name__ == '__main__':

    y_true = [1, 1, 0, 0, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 0]
    print('Accuracy calculado a mano:', accuracy(y_true=y_true, y_pred=y_pred))
    print('Accuracy calculado con scikit learn:', accuracy_score(y_true=y_true, y_pred=y_pred))
    print('Precision calculada a mano:', precision(y_true=y_true, y_pred=y_pred))
    print('Precision calculado con scikit learn:', precision_score(y_true=y_true, y_pred=y_pred))
    print('Recall calculado a mano:', recall(y_true=y_true, y_pred=y_pred))
    print('Recall calculado con scikit learn:', recall_score(y_true=y_true, y_pred=y_pred))
