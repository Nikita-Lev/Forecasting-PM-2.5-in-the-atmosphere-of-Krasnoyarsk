#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox, pearsonr, spearmanr
from datetime import datetime, timedelta

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

from itertools import product
from tqdm.notebook import tqdm
import seaborn as sns

import os 
import warnings

# ### Функции дифференцирования ряда и подбора гиперпараметров ARIMA


def PlotSerie(y, l):
    '''Построение графика временного ряда
    '''
    
    plt.figure(figsize=(11, 5))
    
    plt.plot(y)
    plt.title(l)
    plt.xlabel('time')
    plt.ylabel('PM 2.5')
    plt.grid()
    plt.show()


def CompareGraphDifPlot(y1, y2, l1, l2):
    '''Построение двух графиков временных рядов, вычисление критерия Дики-Фуллера
    '''
    
    PlotSerie(y1, l1)
    
    # Критерий Дики-Фуллера
    print('Критерий Дики-Фулера для исходного ряда:', round(adfuller(y1)[1], 4))
    
    PlotSerie(y2, l2)
        
    # Критерий Дики-Фуллера
    print('Критерий Дики-Фулера для преобразованного ряда:', round(adfuller(y2)[1], 4))



def plotCF(data, lags):
    '''Построение автокорреляционной и частичной автокорреляционной функций
    '''
    
    # График автокорреляции
    fig, ax = plt.subplots(figsize = (9, 3)) 

    sm.graphics.tsa.plot_acf(data.values, ax=ax, lags = lags)
    plt.grid()
    plt.show()
    
    # График частичной автокорреляции
    fig, ax = plt.subplots(figsize = (9, 3))

    sm.graphics.tsa.plot_pacf(data.values, ax=ax, method='ywm', lags = lags)
    plt.grid()
    plt.show()
    
    
def SeriesDiff(dat, d):
    '''Дифференцирование ряда
    '''
    
    dShift = dat - dat.shift(d)
    
    CompareGraphDifPlot(dat, dShift.iloc[d:], 'Исходный временной ряд', 'Временной ряд после дифференцирования')


def get_paramS(p, d, q, P, D, Q):
    '''Получение списка параметров, включая сезонные
    '''
    
    parameters = product(range(p + 1), range(d + 1), range(q + 1), range(P + 1), range(D + 1), range(Q + 1))
    return list(parameters)

def get_param(p, d, q):
    '''Получение списка параметров, не включая сезонные
    '''
    
    parameters = product(range(p + 1), range(d + 1), range(q + 1))
    return list(parameters)


def optimize_SARIMA(data, p, d, q, P=0, D=0, Q=0, s=0, exog = None):
    """
        Подбор оптимальных гиперпараметров поиском по сетке
    
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    if s:
        parameters_list = get_paramS(p, d, q, P, D, Q)
    else:
        parameters_list = get_param(p, d, q)

                    
    results = []
    best_aic = float('inf')
                    
    
    warnings.filterwarnings("ignore")
    for param in tqdm(parameters_list):
        try:
            # SARIMA
            if s:
                model = sm.tsa.statespace.SARIMAX(data, order=(param[0], param[1], param[2]),
                                              seasonal_order=(param[3], param[4], param[5], s), exog = exog).fit(disp=-1)
            # ARIMA
            else:
                model = sm.tsa.statespace.SARIMAX(data, order=(param[0], param[1], param[2]), exog = exog).fit(disp=-1)
        
        except ValueError:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_aic = aic
            best_param = param
    
    print('Оптимальные гиперпараметры', best_param)
    return best_param


def getModel(data, param, s, exog):
    '''Построение модели и диагностика
    '''
    
    if s:
        model = sm.tsa.statespace.SARIMAX(data, order = (param[0], param[1], param[2]),
                                      seasonal_order=(param[3], param[4], param[5], s), exog = exog).fit(disp=-1)
    else:
        model = sm.tsa.statespace.SARIMAX(data, order = (param[0], param[1], param[2]), exog = exog).fit(disp=-1)
    
    display(model.summary().tables[1]) # Таблица коэффициентов
    
    model.plot_diagnostics(figsize=(10, 7)) # Диагностические графики
    plt.show()
    
    return model


def invboxcox(y, lmbda):
    '''Обратное преобразование Бокса-Кокса
    '''
    
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))
    
def BoxCox(data):
    '''Преобразование Бокса-Кокса
    '''
    
    # Преобразование бокса-кокса
    ValBC, lmbda = boxcox(data.values)
    print('lambda =', lmbda)
    return ValBC, lmbda


def ReIndex(data, h):
    ''' Переиндексация (необходим сдвиг значений временного ряда)
    '''
    
    indexes = data.index[:-h] # Срез индексов
    data = data[h:] # Срез данных
    data.index = indexes # Переиндексация
    return data


def Metrics(y, y_pred):
    '''Вычисление метрик качества прогнозов
    '''


    mse = round(mean_squared_error(y, y_pred), 2)
    mae = round(mean_absolute_error(y, y_pred), 2)
    mape = round(mean_absolute_percentage_error(y, y_pred), 2)
    R2 = round(r2_score(y, y_pred), 2)
    #wape = round(100 * np.sum(np.abs(y - y_pred)) / np.sum(np.abs(y)), 2)
    
    return(pd.DataFrame(index = ['MSE', 'MAE', 'MAPE', '$R^2$'], data = {'Значение': [mse, mae, mape, R2]}))
    

def GetPred(model, start, end, exog = None, lmbda = False):
    '''Построение прогнозов до определённой даты
    '''
    
    pred = model.predict(start = pd.to_datetime(start), end = pd.to_datetime(end), dynamic=False, exog = exog)
    if lmbda != False:
        pred = invboxcox(pred, lmbda)
    return pred
    
    
class Forecaster:
    '''Класс прогнозирования
    '''

    def __init__(self, sensor, district, begin, start, end, exog_features = ['Temperature', 'Wet', 'Pressure', 'Wind_speed', 'Wind_dir'], fill = '', lags = None):
        self.district = district
        # Срез данных от begin до end
        self.begin = begin
        self.start = start # Начало прогнозирования
        self.end = end
        
        # Датчики (КНЦ или министерские)
        self.sensor = sensor
        
    
        # Данные pm 2.5
        self.df = pd.read_csv(f'../data/pm25_{self.sensor}{fill}.csv', sep = ';', index_col = ['Date'], parse_dates = ['Date'], usecols = ['Date', district])
        
        # Проверка на пропуски
        nans = pd.isnull(self.df[begin : end].values).sum()
        if nans > 0:
            print('Пропуски в PM :', nans, '\n')
        
        # Тренировочная и тестовая выборки
        self.train = self.df[begin : start]
        self.test = self.df[start : end]
        
        # Экзогенные переменные
        self.exog_features = exog_features
        
        self.exogTrain = None
        self.exogTest = None
        if exog_features:
            self.exogTrain = pd.DataFrame()
            self.exogTest = pd.DataFrame()
            # Цикл по экзогенным переменным
            for feature in exog_features:
                
                feat_df = pd.read_csv(f'../data/Features{fill}/{feature}_{self.sensor}.csv', sep = ';', index_col = ['Date'], parse_dates = ['Date'], usecols = ['Date', district])
                
                # Проверка на пропуски
                nans = pd.isnull(feat_df[begin : end].values).sum()
                if nans > 0:
                    print(f'Пропуски в {feature} :', nans, '\n')
                
                
                # Корреляции с pm
                #print('Корреляция Пирсона:', round(pearsonr(self.train, feat_df[begin : start].values)[0], 3),'\nСпирмана:',  round(spearmanr(self.train, feat_df[begin : start].values)[0], 3), '\n')#
                
                
                self.exogTrain[feature] = feat_df[begin : start][district].values
                self.exogTest[feature] = feat_df[start : end][district].values[1:]
            
           
            plt.figure(figsize = (11, 3))
             # Тепловая карта корреляций
            self.exogTrain['PM'] = self.train.values
            
            plt.subplot(1, 2, 1)
            sns.heatmap(data = self.exogTrain.corr(), annot=True)
            plt.title('Корреляции Пирсона:')
                       
                
            plt.subplot(1, 2, 2)
            sns.heatmap(data = self.exogTrain.corr(method='spearman'), annot=True)
            plt.title('Корреляции Спирмана:')
            
            self.exogTrain.drop(['PM'], axis = 1, inplace = True)
            plt.show()
            
            
        
        # Отображение ACF, PACF
        plotCF(self.train, lags)

      
    def CompareGraph(self, y1, y2, l1, l2, title):
        '''Сравнение двух функций на одном макете
        '''
    
    
        plt.figure(figsize = (11, 4))
        
        plt.plot(y1, label = l1)
        plt.plot(y2, label = l2)

        plt.title(f'Сравнение истинных и прогнозируемых значений {title}. ARIMA{self.optParam}')
        
        plt.xlabel('Время (YYYY-MM-DD)')
        plt.ylabel('PM 2.5, мкг/м³')
        plt.legend()
        plt.grid()
        
        # Вычисление метрик
        metrics = Metrics(y1, y2)
        
        
        '''table = plt.table(cellText = metrics.values, colLabels = metrics.columns, loc = 'right', colWidths =[0.12, 0.12])
        table.set_fontsize(20)
        table.scale(1, 1.5)
        '''
        if self.save == True:
            
            self.path = f'Results/{self.district}_{self.sensor}/{self.begin} — {self.end}'
            
            if not os.path.exists(self.path):
                os.mkdir(self.path)
                
            # Название графика
            figName = title
            # Сезонная модель
            if self.s:
                figName += '_S'
            # Экзогенные переменные
            
            # С экзогенными переменными
            if self.exog_features:
                for feat in self.exog_features:
                    figName += '_' + feat
            
            plt.savefig(f'{self.path}/{figName}.png', bbox_inches='tight')
        
        plt.show()
        
        display(metrics)

     
    def TrainCompare(self, lmbda = False):
        '''Сравнение значений, полученных моделью с истинными (обучающая выборка)
        '''
        
        self.modVal = ReIndex(self.model.fittedvalues, 1)

        # Если применено преобразование Бокса-Кокса, обратить
        if lmbda != False:
            self.modVal = invboxcox(self.modVal, lmbda)
        
        self.CompareGraph(self.train[:-1], self.modVal, 'Исходные данные', 'Данные, описываемые моделью', 'train')

    
    def TestCompare(self):
        '''Сравнение предсказаний с истинными значениями (тестовая выборка)
        '''
        
        self.CompareGraph(self.test, self.predictions, 'Истинные данные', 'Прогнозируемые значения', 'test')
        
        
    def getModel(self, p, d, q, P = 0, D = 0, Q = 0, s = 0, exog_features = None, use_optimal = False, save = False):
        '''Подбор гиперпараметров, построение модели
        '''
        
        self.save = save
        
        self.s = s
        
        
        # Использование заданных экзогенных признаков  
        self.exog_features = exog_features 
        if exog_features:
            # Переход от df к значениям
            self.exogTrainUsed = self.exogTrain[exog_features].values
            self.exogTestUsed = self.exogTest[exog_features].values
        else:
            self.exogTrainUsed = self.exogTestUsed = None
        
        # Поиск оптимальных гиперпараметров
        if use_optimal == True:
            
            self.optParam = optimize_SARIMA(self.train, p = p , d = d, q = q, P = P, D = D, Q = Q, s = s, exog = self.exogTrainUsed)
            self.d = self.optParam[1]
        else:
            if self.s:
                self.optParam = (p, d, q, P, D, Q)
            else:
                self.optParam = (p, d, q)
            self.d = d
          
        # Обучение модели и диагностика
        self.model = getModel(self.train, self.optParam, self.s, self.exogTrainUsed)

        # Модель на обучающей выборке
        self.TrainCompare()

        # Получение прогнозов
        self.predictions = GetPred(self.model, self.start, self.end, self.exogTestUsed)

        # Модель на тестовой выборке
        self.TestCompare()
        
        if self.save == True:
            # Тип модели S + ARIMA + X
            modelType = 'ARIMA'
            
            # Сезонная
            if self.s:
                modelType = 'S' + modelType
            # С экзогенными переменными
            if self.exog_features:
                modelType += 'X'
                for feat in self.exog_features:
                    modelType += '_' + feat
               
            self.model.save(f'{self.path}/{modelType}.pkl')
    