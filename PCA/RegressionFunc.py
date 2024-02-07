### Методы для построения регрессий

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from openpyxl import load_workbook

# Срез количества дней для тестовой выборки
maxForecastDays = 5

# Сравнение графиков двух списков значений
def CompareGraph(x, y1, y2, l1, l2, title):
    plt.figure(figsize = (9, 4))
    plt.plot(x, y1, label = l1)
    plt.plot(x, y2, label = l2)

    plt.xlabel('time')
    plt.ylabel('PM 2.5, мкг/м³')
    plt.title(title)
    
    plt.legend()
    plt.grid()
    
# Прогнозы линейной регрессии и вычисление ошибок
def PredictAndMetrics(model, X, y, rnd = 2, plot = False, modelName = ''):
    y_pred = model.predict(X)
    
    if plot:
        CompareGraph(y.index, y, y_pred, 'Истинные значения', 'Пронозируемые значения', f'Сравнение прогноза {modelName} с истинными данными')
    return list(map(lambda i: round(i, rnd), [mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred), mean_absolute_percentage_error(y, y_pred), r2_score(y, y_pred), 1 - (1-r2_score(y, y_pred))*(X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)]))


# Применение линейной регрессии с train_test_split
def Regression(X, y, ts, vs = 0, modelType = 'linear', alpha = 1.0):
    # В случае валидационной выборки, разделить в соотношении (1-vs-ts) : vs : ts
    if vs:
        # (1-vs-ts) : vs + ts
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = ts + vs, random_state = 100)
        
        # vs : ts
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size = ts / (vs + ts), random_state = 100) 
        
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts, random_state = 100)
    
    #display(X.shape, X_train.shape,X_valid.shape, X_test.shape )
    
    # Создание и обучение модели
    if modelType == 'linear':
        model = LinearRegression()
    
    if modelType == 'lasso':
        model = Lasso(alpha = alpha)
        
    if modelType == 'ridge':
        model = Ridge(alpha = alpha)
    
    model.fit(X_train, y_train)
    
    # Список ошибок
    metr = []
    
    # Тренировочная выборка 
    metr.append(PredictAndMetrics(model, X_train, y_train))
    
    # Валидационная выборка
    if vs:
        metr.append(PredictAndMetrics(model, X_valid, y_valid))
    
    # Тестовая выборка
    metr.append(PredictAndMetrics(model, X_test, y_test))
    
    return model, metr

# Возведение признаков в степень
def PolynomFeat(X, degree):
    # Возведение признаков в степень
    pol = PolynomialFeatures(degree = degree)
    return pol.fit_transform(X)

# Построение различных моделей регрессии
def BuildModels(data, forecastDays = 0, plot_forecast = False):
    
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
    
    
    
    linTypes = ['linear', 'lasso', 'ridge']
    polDegr = [2, 3]

    metricNames = ['MSE', 'MAE', 'MAPE', 'R2', 'R2_adj']

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

# Отбор признаков, наиболее коррелирующих с PM, исключая мультиколлинеарность
def featureSelect(matr, features, corrPMvalue = 0.2, corrFeatValue = 0.6):
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

# Сохранение результатов в файл
def SaveResults(savePath, fileName, df, sheetName = 'results', features = [], corr = None):
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
            resDf.to_excel(writer, sheet_name = sheetName, startrow = numRow, index = False)

            numRow += resDf.shape[0] + 3
    
    