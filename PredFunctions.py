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

from sklearn.metrics import r2_score

from itertools import product
from tqdm.notebook import tqdm


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
            # Сезонная модель
            if s:
                model = sm.tsa.statespace.SARIMAX(data, order=(param[0], param[1], param[2]),
                                              seasonal_order=(param[3], param[4], param[5], s),
                                             enforce_stationarity=False).fit(disp=-1)
            else:
                model = sm.tsa.statespace.SARIMAX(data, order=(param[0], param[1], param[2]),
                                             enforce_stationarity=False).fit(disp=-1)
        
        except ValueError:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    #Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table['parameters'][0]


# In[12]:


# Построение модели и диагностика
def getModel(data, param, s):
    if s:
        model = sm.tsa.statespace.SARIMAX(data, order = (param[0], param[1], param[2]),
                                      seasonal_order=(param[3], param[4], param[5], s)).fit(disp=-1)
    else:
        model = sm.tsa.statespace.SARIMAX(data, order = (param[0], param[1], param[2])).fit(disp=-1)
    
    display(model.summary().tables[1]) # Таблица коэффициентов
    
    model.plot_diagnostics(figsize=(14, 12)) # Характеристика модели
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


# Сравнение двух графиков вместе
def CompareGraph(y1, y2, l1, l2, title):
    plt.figure(figsize = (13, 6))
    plt.plot(y1, label = l1)
    plt.plot(y2, label = l2)

    plt.gca().set(title = title)
    plt.legend()
    plt.grid()
    plt.show()


# In[15]:


# Переиндексация (необходим сдвиг значений)
def ReIndex(data, h):
    indexes = data.index[:-h] # Срез индексов
    data = data[h:] # Срез данных
    data.index = indexes # Переиндексация
    return data


# Сравнение значений, полученных моделью с истинными
def CompareModel(realData, model,lmbda = False):
    modVal = ReIndex(model.fittedvalues, 1)
    
    # Если применено преобразование Бокса-Кокса, обратить
    if lmbda != False:
        realData = invboxcox(realData, lmbda)
        modVal = invboxcox(modVal, lmbda)
    
    CompareGraph(realData, modVal, 'Исходные данные', 'Данные, описываемые моделью', 'Сравнение истинных значений с полученными моделью')
    
    mae = np.mean(np.abs(realData - modVal))
    R2 = r2_score(realData[:-1], modVal)
    print('MAE:', mae)
    print('R2:', R2)

# Сравнение предсказаний с истинными значениями
def ComparePred(realData, pred):
    CompareGraph(realData, pred, 'Истинные данные', 'Прогнозируемые данные', f'Сравнение прогнозов с истинными данными')
    
    mae = np.mean(np.abs(realData - pred))
    R2 = r2_score(realData, pred)
    print('MAE:', mae)
    print('R2:', R2)

# In[16]:


# Построение прогнозов до определённой даты
def GetPred(model, start, end, lmbda = False):
    pred = model.predict(start = pd.to_datetime(start), end = pd.to_datetime(end), dynamic=False)
    if lmbda != False:
        pred = invboxcox(pred, lmbda)
    return pred

