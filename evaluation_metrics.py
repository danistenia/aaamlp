''' Implementation of evaluation metrics for ML models - Recreating what scikit-learn does. '''

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt

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

def prc(y_true: list, y_pred: list, display=False):
    '''
    Función que calcula el precision y recall curve.
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve un grafico con la curva de precision recall.
    '''
    x_recall = []
    y_precision = []

    thresholds = [0, 0.1, 0.15, 0.22, 0.29, 0.33, 0.45, 0.51, 0.55, 0.8 , 0.85, 0.9]
    
    for t in thresholds:
        temp_pred = [x>=t for x in y_pred]
        x_recall.append(recall(y_true=y_true, y_pred=temp_pred))
        y_precision.append(precision(y_true=y_true, y_pred=temp_pred))
    print(x_recall)
    print(y_precision)
    if display:
        plt.plot(x_recall, y_precision)
        plt.show()
    

if __name__ == '__main__':

    y_true = [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
    y_pred_floats = [0.9, 0.8, 0.85, 0.55, 0, 0, 0.45, 0, 0, 0.33, 0, 0, 0.22, 0, 0.29, 0, 0.51, 0.10, 0, 0, 0.15]
    #size = 1_500
    #y_true = np.random.randint(2, size=size)
    #y_pred = np.random.randint(2, size=size)
    print('Accuracy calculado a mano:', accuracy(y_true=y_true, y_pred=y_pred))
    print('Accuracy calculado con scikit learn:', accuracy_score(y_true=y_true, y_pred=y_pred))
    print('Precision calculada a mano:', precision(y_true=y_true, y_pred=y_pred))
    print('Precision calculado con scikit learn:', precision_score(y_true=y_true, y_pred=y_pred))
    print('Recall calculado a mano:', recall(y_true=y_true, y_pred=y_pred))
    print('Recall calculado con scikit learn:', recall_score(y_true=y_true, y_pred=y_pred))
    prc(y_true=y_true, y_pred=y_pred_floats, display=True)
    