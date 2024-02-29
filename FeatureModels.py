import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def Metrics(y, y_pred):
    '''Вычисление метрик качества прогнозов
    '''
    
    mse = round(mean_squared_error(y, y_pred), 4)
    mae = round(mean_absolute_error(y, y_pred), 4)
    mape = round(mean_absolute_percentage_error(y, y_pred), 4)
    R2 = round(r2_score(y, y_pred), 4)
    
    return [mse, mae, mape, R2]


def MetricsDF(model, x_train, x_valid, y_train, y_valid, tableName = 'Errors', metricsCV = [], testUsed = False, x_test = [], y_test = [], plot = False, savePath = '', sheet_name = 'Results'):
    '''Формирование таблицы метрик на тренировочной и тестовой выборке
    '''
    
    
    resDf = pd.DataFrame(index=['MSE', 'MAE', 'MAPE', 'R2'], data = 
                             {'Train': Metrics(y_train, model.predict(x_train)),
                             'Valid': Metrics(y_valid, model.predict(x_valid))})
    
    # Усреднение по валиду и кросс валидации
    if len(metricsCV):
        resDf['Valid + CV'] = (resDf['Valid'] + metricsCV) / 2
        resDf.drop(['Valid'], axis = 1, inplace = True)
    
    if testUsed:
        resDf['Test'] = Metrics(y_test, model.predict(x_test))
                        
    resDf.index.name = tableName
    
    
    display(resDf)
    
    
    if plot:
        x_valid.sort_index(inplace = True)
        y_valid.sort_index(inplace = True)
        CompareGraph(y_valid.index, y_valid, model.predict(x_valid), 'Истинные значения', 'Прогнозируемые значения', 'Сравнение прогноза с истинными данными. Valid')
        if testUsed:
            x_test.sort_index(inplace = True)
            y_test.sort_index(inplace = True)
            CompareGraph(y_test.index, y_test, model.predict(x_test), 'Истинные значения', 'Прогнозируемые значения', 'Сравнение прогноза с истинными данными. Test')
    
    # Сохранение в файл
    if savePath:
        
        # Номер последней строки в файле
        numRow = 0
        wb = load_workbook(savePath)
        if sheet_name in wb.sheetnames:
            numRow = wb[sheet_name].max_row + 2
        
        with pd.ExcelWriter(savePath, engine="openpyxl", mode = 'a', if_sheet_exists='overlay') as writer:
            resDf.to_excel(writer, sheet_name = sheet_name, startrow = numRow)


def PlotGraph(x, y, xlabel, ylabel, title):
    '''Отрисовка графика функции
    '''
    
    plt.figure(figsize = (8, 3))
    plt.plot(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid()
    plt.show()
            
        
def CompareGraph(x, y1, y2, l1, l2, title):
    '''Сравнение графиков двух списков значений
    '''
    
    plt.figure(figsize = (10, 4))
    plt.plot(x, y1, label = l1)
    plt.plot(x, y2, label = l2)

    plt.xlabel('time')
    plt.ylabel('PM 2.5, мкг/м³')
    plt.title(title)
    
    plt.legend()
    plt.grid()
    
    plt.show()
    
    
def splitDataBySeason(df, season):
    '''Разделение данных по сезонам
    '''
    
    # Зима
    if season == 'winters':
        win19 = df[:'2019-03-01 00:00:00']
        win19_20 = df['2019-11-27 00:00:00':'2020-02-27 00:00:00']
        win20_21 = df['2020-11-27 00:00:00':'2021-02-25 00:00:00']
        win21_22 = df['2021-12-15 00:00:00':'2022-02-22 00:00:00']
        win22_23 = df['2022-12-13 00:00:00' : '2023-02-22 00:00:00']

        return pd.concat([win19, win19_20, win20_21, win21_22, win22_23])
    
    # Весна
    if season == 'springs':
        spr19 = pd.concat([df['2019-03-10' : '2019-05-01'], df['2019-11-01' : '2019-11-22']])
        spr20 = pd.concat([df['2020-03-01' : '2020-05-01'], df['2020-11-01' : '2020-11-10']])
        spr21 = pd.concat([df['2021-03-10' : '2021-05-01'], df['2021-11-01' : '2021-11-25']])
        spr22 = pd.concat([df['2022-03-01' : '2022-05-01'], df['2022-11-01' : '2022-11-25']])
        spr23 = df['2023-03-01' : ]

        return pd.concat([spr19, spr20, spr21, spr22, spr23])

    # Лето
    if season == 'summers':
        sum19 = df['2019-05-01' : '2019-07-12']
        sum20 = df['2020-05-01' : '2020-08-01']
        sum21 = df['2021-05-01' : '2021-08-01']
        sum22 = df['2022-05-01' : '2022-08-01']

        return pd.concat([sum19, sum20, sum21, sum22])
    
    # Осень
    if season == 'autumns':
        aut19 = df['2019-08-21' : '2019-10-16']
        aut20 = df['2020-08-17' : '2020-10-30']
        aut21 = df['2021-08-15' : '2021-11-01']
        aut22 = df['2022-08-01' : '2022-10-14']

        return pd.concat([aut19, aut20, aut21, aut22])
    

def ModelProcessing(model, sensor, fill, district, begin, end, seasonModel = '', valid_size = 0.3, season_test = '', params = None, gridSearch = True, paramDependencies = [], plotRes = False, featImp = False,
                       path = '../data/', savePath = ''):
    '''Обучение, подбор параметров, валидация, прогнозирование для выбранного класса моделей
    '''
    
    # Данные pm 
    df = pd.read_csv(f"{path}pm25_{sensor}{fill}.csv", sep = ';', index_col = ['Date'], parse_dates = ['Date'], usecols = ['Date', district])
    df.rename(columns = {district : 'pm'}, inplace = True)

    # Исключение выборки прогнозирования
    df.drop(index = [f'2023-02-{i}' for i in range(18, 23)] + [f'2023-03-{i}' for i in range(23, 28)] +
                    [f'2022-07-{i}' for i in range(28, 32)] + [f'2022-08-01'] +
                     [f'2022-10-{i}' for i in range(10, 15)], inplace = True)
        
    
    features = ['Pressure', 'Temperature', 'Wet', 'Wind_dir', 'Wind_speed']
    for feat in features:
        df[feat] = pd.read_csv(f"{path}Features{fill}/{feat}_{sensor}.csv", sep = ';', index_col = ['Date'], parse_dates = ['Date'], usecols = ['Date', district])

        # Проверка на пропуски
        nans = pd.isnull(df[begin : end].values).sum()
             
        if nans > 0:
            print(f'Пропуски в {feat} :', nans)
 
    
    # Добавление инверсий
    inversions = pd.read_csv(f"{path}Inversions.csv", sep = ';', index_col = ['Date'], parse_dates = ['Date'])
        
    df = inversions.join(df)    
    
    df.dropna(inplace = True)
    
    
    testUsed = False
    
    test_index = pd.read_csv(f"../data/test_index.csv", sep = ';', dayfirst = True, parse_dates = [0, 1, 2, 3])
    for col in test_index.columns:
        idx = list(set(test_index[col].dropna()) & set(df.index))
        
        if col == season_test:
            testUsed = True
            x_test = df.loc[idx].drop(['pm'], axis = 1)
            y_test = df.loc[idx]['pm']
       
        df.drop(idx, inplace = True)
    
    df = df[begin : end]
    
    
    tableName = f'{min(df.index.date)} — {max(df.index.date)}'
    
    
    # Срез для выбранного сезона
    if seasonModel:
        df = splitDataBySeason(df, seasonModel)
    
    # Разбиение данных
    x = df.drop(['pm'], axis = 1)
    y = df['pm']

    # train : valid выборки 1-valid_size : valid_size
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = valid_size, random_state=1)
    
    # Поиск по сетке
    if gridSearch:
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'r2' ]
        model = GridSearchCV(estimator = model, param_grid = params, scoring = scoring, refit = 'r2', n_jobs = -1)
    else:
        model.set_params(**params)
    
    # Обучаем
    model.fit(x_train, y_train)
    
    # Допольнительная колонка ошибок
    metricsCV = []
    # Параметры лучшей модели
    if gridSearch:
        param_grid = params
        params = model.best_params_
        
        # Качество по кросс валидации
        scorings = pd.DataFrame(model.cv_results_)[list(map(lambda x : 'mean_test_' + x, scoring))]
        metricsCV = scorings.iloc[np.where(scorings == model.best_score_)[0][0]].values
        metricsCV[:3] = abs(metricsCV[:3])
        
        model = model.best_estimator_
        
        
    
    tableName += f" {params}"
    
    if testUsed:
        MetricsDF(model, x_train, x_valid, y_train, y_valid, tableName, metricsCV, testUsed, x_test, y_test, plotRes, savePath)
    else:
        MetricsDF(model, x_train, x_valid, y_train, y_valid, tableName, metricsCV, testUsed, plotRes, savePath)
    
    # Определение важности признаков
    if featImp: FeatureImportance(model)
        
        
    # Зависимость качества от параметров
    if len(paramDependencies):
        # Название параметра
        for paramName in paramDependencies:
            loss_mse = []
            for p in tqdm(param_grid[paramName]):
                model.set_params(**{paramName: p})

                model.fit(x_train, y_train)
                loss_mse.append(mean_squared_error(model.predict(x_valid), y_valid))
        
            PlotGraph(param_grid[paramName], loss_mse, paramName, 'MSE', f"Зависимость MSE от {paramName}")
    

def FeatureImportance(model):
    '''Вычисление и отрисовка важности признаков
    '''
    importances = dict(sorted(zip(model.feature_names_in_, model.feature_importances_), key = lambda x: x[1], reverse = True))
    
    plt.figure(figsize = (10, 4))
    plt.bar(importances.keys(), importances.values())
    
    plt.title('Важность признаков')
    plt.xlabel('Признаки')
    plt.ylabel('Важность')
    
    plt.grid()
    plt.show()

