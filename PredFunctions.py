#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from datetime import datetime, timedelta

from sklearn.metrics import r2_score, mean_absolute_percentage_error as smape

from itertools import product
from tqdm.notebook import tqdm

import os 


# ### Функции дифференцирования ряда и подбора гиперпараметров ARIMA

# In[7]:


# Построение графика временного ряда
def PlotSerie(y, l):
    plt.figure(figsize=(11, 5))
    
    plt.plot(y)
    plt.title(l)
    plt.grid()
    plt.show()


# In[8]:


# Сравнение двух графиков отдельно
def CompareGraphDifPlot(y1, y2, l1, l2):
    
    PlotSerie(y1, l1)
    
    # Критерий Дики-Фуллера
    print('Критерий Дики-Фулера для исходного ряда:', round(adfuller(y1)[1], 4))
    
    PlotSerie(y2, l2)
        
    # Критерий Дики-Фуллера
    print('Критерий Дики-Фулера для преобразованного ряда:', round(adfuller(y2)[1], 4))


# In[9]:


# Построение автокорреляционной и частичной автокорреляционной функций
def plotCF(data):
    # График автокорреляции
    fig, ax = plt.subplots(figsize = (11, 4)) 

    sm.graphics.tsa.plot_acf(data.values, ax=ax)
    plt.grid()
    plt.show()
    
    # График частичной автокорреляции
    fig, ax = plt.subplots(figsize = (11, 4))

    sm.graphics.tsa.plot_pacf(data.values, ax=ax, method='ywm')
    plt.grid()
    plt.show()


# In[10]:


# Дифференцирование ряда
def SeriesDiff(dat, d):
    dShift = dat - dat.shift(d)
    
    CompareGraphDifPlot(dat, dShift.iloc[d:], 'Исходный временной ряд', 'Временной ряд после дифференцирования')


# In[11]:


# Получение списка параметров
def get_paramS(p, d, q, P, D, Q):
    parameters = product(range(p + 1), range(d + 1), range(q + 1), range(P + 1), range(D + 1), range(Q + 1))
    return list(parameters)

def get_param(p, d, q):
    parameters = product(range(p + 1), range(d + 1), range(q + 1))
    return list(parameters)


# Подбор оптимальных гиперпараметров поиском по сетке
import warnings
def optimize_SARIMA(data, p, d, q, P=0, D=0, Q=0, s=0):
    """
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
                                              seasonal_order=(param[3], param[4], param[5], s)).fit(disp=-1)
            # ARIMA
            else:
                model = sm.tsa.statespace.SARIMAX(data, order=(param[0], param[1], param[2])).fit(disp=-1)
        
        except ValueError:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_aic = aic
            best_param = param
    
    print('Оптимальные гиперпараметры', best_param)
    return best_param


# In[12]:


# Построение модели и диагностика
def getModel(data, param, s):
    if s:
        model = sm.tsa.statespace.SARIMAX(data, order = (param[0], param[1], param[2]),
                                      seasonal_order=(param[3], param[4], param[5], s)).fit(disp=-1)
    else:
        model = sm.tsa.statespace.SARIMAX(data, order = (param[0], param[1], param[2])).fit(disp=-1)
    
    display(model.summary().tables[1]) # Таблица коэффициентов
    
    model.plot_diagnostics(figsize=(14, 12)) # ,Диагностические графики
    plt.show()
    
    return model


# In[13]:


# Обратное преобразование Бокса-Кокса
def invboxcox(y, lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))
    
# Преобразование Бокса-Кокса
def BoxCox(data):
    # Преобразование бокса-кокса
    ValBC, lmbda = stats.boxcox(data.values)
    print('lambda =', lmbda)
    return ValBC, lmbda


# In[14]:


# In[15]:


# Переиндексация (необходим сдвиг значений)
def ReIndex(data, h):
    indexes = data.index[:-h] # Срез индексов
    data = data[h:] # Срез данных
    data.index = indexes # Переиндексация
    return data


# Вычисление метрик качества прогнозов
def Metrics(y, y_pred):
    
    mae = round(np.mean(np.abs(y - y_pred)), 4)
    R2 = round(r2_score(y, y_pred), 4)
    sMAPE = round(100 * smape(y, y_pred), 2)
    wape = round(100 * np.sum(np.abs(y - y_pred)) / np.sum(np.abs(y)), 2)
    
    return(pd.DataFrame({'Метрика' : ['MAE', '$R^2$', 'SMAPE', 'WAPE'], 'Значение': [mae, R2 , f'{sMAPE} %' , f'{wape} %']}))
    



# In[16]:


# Построение прогнозов до определённой даты
def GetPred(model, start, end, lmbda = False):
    pred = model.predict(start = pd.to_datetime(start), end = pd.to_datetime(end), dynamic=False)
    if lmbda != False:
        pred = invboxcox(pred, lmbda)
    return pred
    
    
    
# Класс прогнозирования
class Forecaster:
    
    def __init__(self, sensor, district, begin, start, end):
        self.district = district
        # Срез данных от begin до end
        self.begin = begin
        self.start = start
        # Начало прогнозирования
        self.end = end
        
        # Датчики (КНЦ или министерские)
        self.sensor = sensor
        
        # Данные
        self.df = pd.read_csv(f'pm25_{self.sensor}.csv', sep = ';', index_col = ['Date'], parse_dates = ['Date'])[district]
        
        # Тренировочная и тестовая выборки
        self.train = self.df[begin : start]
        self.test = self.df[start : end]
        
        # Отображение ACF, PACF
        plotCF(self.train)

      
    # Сравнение двух функций на одном макете
    def CompareGraph(self, y1, y2, l1, l2, title):
        plt.figure(figsize = (13, 6))
        plt.plot(y1, label = l1)
        plt.plot(y2, label = l2)

        plt.title(f'Сравнение истинных и прогнозируемых значений {title}. ARIMA{self.optParam}')
        plt.legend()
        plt.grid()
        
        # Вычисление метрик
        metrics = Metrics(y1, y2)
        
        plt.table(cellText = metrics.values, colLabels = metrics.columns, loc='bottom')
        plt.tick_params(axis='x', pad=-15)
        plt.subplots_adjust(bottom=0.15)
        
        if self.save == True:
            
            self.path = f'{self.district}_{self.sensor}/{self.begin} — {self.end}'
            
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            plt.savefig(f'{self.path}/{title}.png')   
        
        plt.show()

       
    # Сравнение значений, полученных моделью с истинными (обучающая выборка)
    def TrainCompare(self, lmbda = False):
        self.modVal = ReIndex(self.model.fittedvalues, 1)

        # Если применено преобразование Бокса-Кокса, обратить
        if lmbda != False:
            self.modVal = invboxcox(self.modVal, lmbda)

        self.CompareGraph(self.train[:-1], self.modVal, 'Исходные данные', 'Данные, описываемые моделью', 'train')

    

    # Сравнение предсказаний с истинными значениями (тестовая выборка)
    def TestCompare(self):
        self.CompareGraph(self.test, self.predictions, 'Истинные данные', 'Прогнозируемые данные', 'test')
        
        
    # Подбор гиперпараметров, построение модели
    def getModel(self, p, d, q, use_optimal = True, save = True):
        self.save = save
        # Поиск оптимальных гиперпараметров
        if use_optimal == True:
            self.optParam = optimize_SARIMA(self.train, p = p , d = d, q = q)
            self.d = self.optParam[1]
        else:
            self.optParam = (p, d, q)
            self.d = d
          
        # Обучение модели и диагностика
        self.model = getModel(self.train, self.optParam, s = 0)

        # Модель на обучающей выборке
        self.TrainCompare()

        # Получение прогнозов
        self.predictions = GetPred(self.model, self.start, self.end)

        # Модель на тестовой выборке
        self.TestCompare()
        
        if self.save == True:
            self.model.save(f'{self.path}/model.pkl')
    
