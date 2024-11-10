# Работа с табличными данными
import pandas as pd
import numpy as np

# Пайплайн
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Преобразование признаков
from sklearn.preprocessing import MinMaxScaler

class OMGReader(BaseEstimator, TransformerMixin):
    pass