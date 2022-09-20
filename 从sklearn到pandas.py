from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


iris = datasets.load_iris()
dataset = sklearn_to_df(iris)
