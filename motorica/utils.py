import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from typing import List

N_OMG_SENSORS = 50
N_GYR_SENSORS = 3
N_ACC_SENSORS = 3
cols_omg = list(map(str, range(N_OMG_SENSORS)))
cols_gyr = [f'GYR{i}' for i in range(N_GYR_SENSORS)]
cols_acc = [f'ACC{i}' for i in range(N_ACC_SENSORS)]

NOGO = 0


def read_meta_info(
    filepath: str, 
    cols_with_lists: List[str] = ['mark_sensors', 'hi_val_sensors']
) -> pd.DataFrame:
    '''
    Читает в датафрейм метаданные из файла, путь к которому передан в аргументе **filepath**.
    Столбцы, названия которых переданы в списке **cols_with_lists**, преобразуются к типу *list*
    '''
    meta_info = pd.read_csv(filepath, index_col=0)
    # После чтения файла заново - столбцы со списками, стали обычными строками. 
    # Нужно их преобразовать обратно в списки
    for col in cols_with_lists:
        if col in meta_info:
            meta_info[col] = meta_info[col].apply(
                lambda x: x.strip('[]').replace("'", ' ').replace(' ', '').split(',')
            )
    return meta_info


def mark_montage(
    data: pd.DataFrame,
    omg_cols: List[str],
    sync_col: str = 'SYNC',
    label_col: str = 'label',
    pron_col: str = 'Pronation',
    window: int = 0,
    scale: bool = True,
    grad1_spacing: int = 5,
    grad2_spacing: int = 5
) -> np.ndarray[int]:
    '''
    Осуществляет поиск границ фактически выполняемых жестов по локальным максимумам второго градиента измерений *omg*-датчиков.
    После определения границ производит разметку:
    - `act_label` метка жеста (0 - 'NOGO', 1 - 'Thumb', 2 - 'Grab', 3 - 'Open', 4 - 'OK', 5 - 'Pistol')
    - `act_pronation` метка пронации (0, 1, 2)
    - `sample` порядковый номер жеста в монтаже

    ### Параметры
    **data**: *pd.DataFrame*<br>данные, включающие временные ряды показаний omg-датчиков

    **omg_cols**: *list*<br>список названий столбцов, соответствующих датчикам, по которым будут определены границы выполняемых жестов

    **sync_col**: *str, default="SYNC"*<br>Название признака, отвечающего за синхронизацию с протоколом

    **sync_shift**: *int, default=0*<br>Общий сдвиг разметки синхронизации

    **window**: *int, default=10*<br>ширина окна для предварительного сглаживания показаний датчиков по медиане; `0` - без сглаживания

    **scale**: *bool, default=True*<br>выполнять ли предварительное приведение показаний omg-датчиков к единому масштабу

    **grad1_spacing**: *int, default=5*<br>параметр `spacing` для функции `numpy.gradient()` для вычисления **первого** градиента

    **grad2_spacing**: *int, default=5*<br>параметр `spacing` для функции `numpy.gradient()` для вычисления **второго** градиента

    ### Возвращаемый результат

    Кортеж (**data_copy**, **bounds**, **grad2**)
    
    **data_copy**: *pandas.DataFrame* Размеченная копия данных

    **bounds**: *numpy.ndarray[int]*<br>Номера строк (индексов), соответствующих границе выполняемых жестов

    **grtad2**: *numpy.ndarray[float]* Второй градиент
    '''

    omg = np.array(data[omg_cols])

    # Сглаживание
    if window:
        omg = pd.DataFrame(omg).rolling(window, center=True).median()
    
    # Приведение к единому масштабу
    if scale:
        omg = MinMaxScaler((0, 1000)).fit_transform(omg)

    # Вычисление градиентов:
    # 1) Первый – сумма абсолютных градиентов по модулю
    grad1 = np.sum(np.abs(np.gradient(omg, grad1_spacing, axis=0)), axis=1)
    # не забудем заполнить образовавшиеся "дырки" из NaN
    grad1 = np.nan_to_num(grad1) 

    # 2) Второй – градиент первого градиента,
    grad2 = np.gradient(grad1, grad2_spacing)
    grad2 = np.nan_to_num(grad2)
    # усиленный возведением в квадрат, но с сохранением знака
    grad2 *= np.abs(grad2)

    # Искать локальный максимум градиентов будем только среди "пиков"
    peaks = pd.Series(grad2)
    peaks[grad2 < 0] = 0
    mask = (peaks.shift(-1) < peaks) & (peaks.shift(1) < peaks)
    peaks[~mask] = 0

    # Искать локальные максимумы второго градиента будем внутри отрезков, 
    # определяемых по признаку синхронизации
    sync_mask = data[sync_col] != data[sync_col].shift(-1)
    sync_index = data[sync_mask].index

    res = []
    for l, r in zip(sync_index, sync_index[1:]):
        try:
            max_i = np.argmax(peaks[l: r])
        except ValueError:
            break
        res.append(
            (l + max_i,                      # индекс начала жеста (начало смены жеста)
             data.loc[l + max_i, label_col], # метка жеста
             data.loc[l + max_i, pron_col])  # метка пронации
        )

    # Отдельно сохраним индексы границ жестов (мы их возвращаем в кортеже результата функции)
    bounds = np.array(res)[:, 0]

    # Теперь разметим каждое измерение в наборе фактическими метками, 
    data_copy = data.copy()
    data_copy['act_label'] = 0
    data_copy['act_pronation'] = 0
    # а также добавим порядковый номер жеста – sample
    data_copy['sample'] = 0

    for i, lr in enumerate(zip(res, res[1:] + [(data_copy.index[-1] + 1, 0, 0)])):
        l, r = lr
        # l[0], r[0] - индексы начала текущего и следующего жестов соответственно
        data_copy.loc[l[0]: r[0], 'act_label'] = l[1]     # l[1] - метка жеста
        data_copy.loc[l[0]: r[0], 'act_pronation'] = l[2] # l[2] - метка пронации
        data_copy.loc[l[0]: r[0], 'sample'] = i + 1       # порядковый номер жеста в монтаже

    return data_copy, bounds, grad2


def read_train_and_test(
        montage: str,
        features: List[str], 
        target_col: str = 'act_label',
        subdir: str = 'marked/'
) -> List:
    
    data_train = pd.read_csv(subdir + montage + ".train", index_col=0)
    data_test = pd.read_csv(subdir + montage + ".test", index_col=0)
    X_train = data_train.drop(target_col, axis=1)[features]
    y_train = data_train[target_col]
    X_test = data_test.drop(target_col, axis=1)[features]
    y_test = data_test[target_col]
    return X_train, X_test, y_train, y_test