### Методы для построения регрессий
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from tqdm import tqdm

from xgboost import XGBRegressor

from openpyxl import load_workbook

# Срез количества дней для тестовой выборки
maxForecastDays = 5


def CompareGraph(x, y1, y2, l1, l2, title):
    ''' Сравнение графиков двух списков значений
    '''
    
    plt.figure(figsize = (9, 4))
    plt.plot(x, y1, label = l1)
    plt.plot(x, y2, label = l2)

    plt.xlabel('time')
    plt.ylabel('PM 2.5, мкг/м³')
    plt.title(title)
    
    plt.legend()
    plt.grid()
    
    
def PredictAndMetrics(model, X, y, rnd = 2, plot = False, modelName = ''):
    ''' Прогнозы линейной регрессии и вычисление ошибок
    '''
    
    y_pred = model.predict(X)
    
    if plot:
        CompareGraph(y.index, y, y_pred, 'Истинные значения', 'Пронозируемые значения', f'Сравнение прогноза {modelName} с истинными данными')
    return list(map(lambda i: round(i, rnd), [mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred), mean_absolute_percentage_error(y, y_pred), r2_score(y, y_pred), 1 - (1-r2_score(y, y_pred))*(X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)]))

 
def splitDataBySeason(df, season):
    ''' Разделение данных по сезонам
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


def Regression(X, y, ts, vs = 0, modelType = 'linear', param_dependences = [], alpha = 1.0):
    '''Построение модели регрессии, подбор гиперпараметров в случае бустинга регрессий
    '''
    
    # В случае валидационной выборки, разделить в соотношении (1-vs-ts) : vs : ts
    if vs:
        # (1-vs-ts) : vs + ts
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = ts + vs, random_state = 100)
        
        # vs : ts
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size = ts / (vs + ts), random_state = 100) 
        
    
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = ts, random_state = 100)
    
    #display(X.shape, x_train.shape, x_valid.shape, x_test.shape )
    
    # Создание и обучение модели
    if modelType == 'linear':
        model = LinearRegression()
    
    if modelType == 'lasso':
        model = Lasso(alpha = alpha)
        
    if modelType == 'ridge':
        model = Ridge(alpha = alpha)
    
    if modelType == 'xgbReg':

        params = {
            'n_estimators' : list(range(100, 10000, 1000)),
            'learning_rate' : [0.05, 0.1, 0.2, 0.3, 0.5]}
        
        model = XGBRegressor(booster = 'gblinear',  random_state = 100)
        
        model = GridSearchCV(estimator = model, param_grid = params)
        
        
    model.fit(x_train, y_train)
    
    # Зависимость качества от параметров
    if modelType == 'xgbReg' and len(param_dependences):
        # Название параметра
        for paramName in param_dependences:
            xgb_top = model.best_estimator_
            loss_mse = []
            for p in tqdm(params[paramName]):
                xgb_top.set_params(**{paramName: p})

                xgb_top.fit(x_train, y_train)
                loss_mse.append(mean_squared_error(xgb_top.predict(x_valid), y_valid))

            plt.figure(figsize = (8, 3))
            plt.plot(params[paramName], loss_mse)
            
            plt.title(f"Зависимость MSE от {paramName}")
            plt.xlabel(paramName)
            plt.ylabel('MSE')
            
            plt.grid()
            plt.show()
    
    # Список ошибок
    metr = []
    
    # Тренировочная выборка 
    metr.append(PredictAndMetrics(model, x_train, y_train))
    
    # Валидационная выборка
    if vs:
        metr.append(PredictAndMetrics(model, x_valid, y_valid))
    
    # Тестовая выборка
    metr.append(PredictAndMetrics(model, x_test, y_test))
    
    return model, metr


def PolynomFeat(X, degree):
    ''' Возведение признаков в степень
    '''
    
    # Возведение признаков в степень
    pol = PolynomialFeatures(degree = degree)
    return pol.fit_transform(X)


def BuildModels(data, modelType = 'linReg', param_dependences = False, forecastDays = 0, plot_forecast = False):
    ''' Построение различных моделей регрессии
    '''
    
    vs = 0.2
    ts = 0.1
    
    # Срез данных для прогнозирования
    if forecastDays:
        X_forecast = data.iloc[-maxForecastDays: data.shape[0] - maxForecastDays + forecastDays].drop(['pm'], axis = 1)
        y_forecast = data.iloc[-maxForecastDays: data.shape[0] - maxForecastDays + forecastDays]['pm']
    
    data = data.iloc[:-maxForecastDays]
    
    # Целевая переменная и признаки
    y = data['pm']
    x = data.drop(['pm'], axis = 1)
    
    
    metricNames = ['MSE', 'MAE', 'MAPE', 'R2', 'R2_adj']

    # Бустинг регрессий
    if modelType == 'xgbReg':
        xgbRes = pd.DataFrame(index = metricNames)
        
        # Построение моделей и вычисление ошибок
        if ts:
            model, metr = Regression(x, y, ts = ts, vs = vs, modelType = modelType, param_dependences = param_dependences)
        else:
            model, metr = Regression(x, y, ts = vs, modelType = modelType, param_dependences = param_dependences)
        
        pd.DataFrame(index = ['1',' 2', '3'], data = {'Train' : [4, 5, 6]})
        xgbRes['Train'] = metr[0]
        xgbRes['Valid'] = metr[1]
        
        if ts: xgbRes['Test'] = metr[2]
        
        if forecastDays:
            xgbRes['Forecast'] = PredictAndMetrics(model, X_forecast, y_forecast, plot = plot_forecast, modelName = mod)
        
        xgbRes.index.name = str(model.best_params_)
        
        display(xgbRes)
        
        return [xgbRes]

    
    linTypes = ['linear', 'lasso', 'ridge']
    polDegr = [2, 3]
    
    train_res = pd.DataFrame({'Train ' : metricNames})

    valid_res = pd.DataFrame({'Valid ' :  metricNames}) 
    
    # Используется тестовая выборка
    if ts: test_res = pd.DataFrame({'Test ' : metricNames})
    
    if forecastDays: forecast_res = pd.DataFrame({'Forecast ' : metricNames})
    

    # Линейные регрессии
    for mod in linTypes:

        # Построение моделей и вычисление ошибок
        if ts:
            model, metr = Regression(x, y, ts = ts, vs = vs, modelType = mod)
        else:
            model, metr = Regression(x, y, ts = vs, modelType = mod)

        train_res[mod] = metr[0]
        valid_res[mod] = metr[1]
        
        if ts: test_res[mod] = metr[2]
        
        if forecastDays:
            forecast_res[mod] = PredictAndMetrics(model, X_forecast, y_forecast, plot = plot_forecast, modelName = mod)
        
    # Полиномиальные регрессии    
    for degr in polDegr:

        # Построение моделей и вычисление ошибок
        if ts:
            model, metr = Regression(PolynomFeat(x, degr), y, ts = ts, vs = vs)
        else:
            model, metr = Regression(PolynomFeat(x, degr), y, ts = vs)
        
        polName = f'Полиномиальная {degr} степени'
        
        train_res[polName] = metr[0]
        valid_res[polName] = metr[1]
        
        if ts: test_res[polName] = metr[2]
            
        if forecastDays:
            forecast_res[polName] = PredictAndMetrics(model, PolynomFeat(X_forecast, degr), y_forecast)
            
    #print('Тренировочная выборка')
    display(train_res)
    
    #print('Валидационная выборка')
    display(valid_res)
    
    results = [train_res, valid_res]
    
    if ts:
        display(test_res)
        results.append(test_res)
    
    #print('Выборка прогнозирования')
    if forecastDays:
        display(forecast_res)
        results.append(forecast_res)
        
    return results


def featureSelect(matr, features, corrPMvalue = 0.2, corrFeatValue = 0.6):
    '''  Отбор признаков, наиболее коррелирующих с PM, исключая мультиколлинеарность
    '''
    
    # Признаки, коррелирующие с другими
    corrFeatures = []
    for feat in matr.apply(abs).sort_values(by='pm', ascending=False)[1:].index:
        # Признаки, которых нет в списке коррелирующих и которые коррелируют с PM со значением >= 0.2
        if feat not in features and feat not in corrFeatures and abs(matr[feat]['pm']) >= corrPMvalue:
            features.append(feat)
            # Исключить признаки коррелирующие с текущим
            for fcor, cor in zip(matr[feat].index[:-1], matr[feat].values[:-1]):
                
                # Если корреляция Спирмена составляет больше 0.6
                if fcor != feat and abs(cor) >= corrFeatValue:
                    corrFeatures.append(fcor)
                    

def SaveResults(savePath, fileName, df, sheetName = 'results', features = [], corr = None):
    ''' Сохранение результатов в файл
    '''
    
    # Извлечение номера последней строки в файле
    numRow = 0
    wb = load_workbook(savePath + fileName)
    
    if sheetName in wb.sheetnames:
        numRow = wb[sheetName].max_row + 4
    
    
    
    with pd.ExcelWriter(savePath + fileName, engine="openpyxl", mode = 'a', if_sheet_exists='overlay') as writer:
        # Запись признаков в файл
        if len(features):
            pd.DataFrame({'Признаки' : features, 'Корреляция Спирмена с pm' : corr}).to_excel(writer, sheet_name = sheetName, startrow = numRow, startcol = df[0].shape[1] + 1, index = False)

        for resDf in df:
            resDf.to_excel(writer, sheet_name = sheetName, startrow = numRow, index = True)

            numRow += resDf.shape[0] + 3
    
    