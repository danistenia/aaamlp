''' Implementation of evaluation metrics for ML models - Recreating what scikit-learn does. '''

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import pandas as pd

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

def precision(y_true: list, y_pred: list) -> float:
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

def recall(y_true: list, y_pred: list) -> float:
    '''
    Función que calcula el recall. Recordar que el recall es el porcentaje de aciertos de 1's sobre todo el universo de 1's. 
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve la cantidad la precision.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"
    recall = true_positives(y_true=y_true, y_pred=y_pred) / (true_positives(y_true=y_true, y_pred=y_pred) + false_negatives(y_true=y_true, y_pred=y_pred))
    return recall

def f1(y_true: list, y_pred: list) -> float:
    '''
    Función que calcula el f1 score que es la armonic mean entre prec y recall.
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve el f1.
    '''
    assert len(y_true) == len(y_pred), "El largo de predicciones debe ser igual al largo ground truths"
    f1 = 2 * recall(y_true, y_pred) * precision(y_true, y_pred) / (recall(y_true, y_pred) + precision(y_true, y_pred))
    return f1

def prc(y_true: list, y_pred: list, display=False):
    '''
    Función que calcula el precision y recall curve. Es decir, precision y recall para distintos cortes de probabilidad.
    Los cortes de probabilidad son estáticos, pero no sé si se definen segun algo o algun criterio.
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
    
    if display:
        plt.plot(x_recall, y_precision)
        plt.show()

def prc_df(y_true: list, y_pred: list):
    '''
    Función que calcula la precision y recall para distintos cortes.
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve un dataframe con precision y recall para distintos cortes de probabilidad
    '''
    x_recall = []
    y_precision = []

    thresholds = [0, 0.1, 0.15, 0.22, 0.29, 0.33, 0.45, 0.51, 0.55, 0.8 , 0.85, 0.9]
    
    for t in thresholds:
        temp_pred = [x>=t for x in y_pred]
        x_recall.append(recall(y_true=y_true, y_pred=temp_pred))
        y_precision.append(precision(y_true=y_true, y_pred=temp_pred))
    
    df = pd.DataFrame(data=
                      {
                          'thrs': thresholds,
                          'recall': x_recall,
                          'precision': y_precision
                      }
                      )
    print(df.to_string())

def trp(y_true: list, y_pred: list):
    '''
    Función que calcula el true positive rate que en realidad es nada más que el recall. También conocido como la sensitivity.
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve el TPR.
    '''

    return recall(y_true, y_pred)

def fpr(y_true: list, y_pred: list):
    '''
    Función que calcula el false positive rate.
    Args:
        y_true: Contiene la lista de valores reales o ground truth.
        y_pred: Contiene los valores predichos por el modelo.
    Returns:
        Devuelve el FPR.
    '''

    fpr = false_positives(y_true, y_pred) / (false_positives(y_true, y_pred) + true_negatives(y_true, y_pred))

    return fpr

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
    #prc(y_true=y_true, y_pred=y_pred_floats, display=True)
    #prc_df(y_true=y_true, y_pred=y_pred_floats)
    print('F1 score calculado a mano:', f1(y_true, y_pred))
    print('F1 score calculado con scikit learn:', f1_score(y_true, y_pred))
    print('tpr', trp(y_true, y_pred))
    print('fpr', fpr(y_true, y_pred))

    