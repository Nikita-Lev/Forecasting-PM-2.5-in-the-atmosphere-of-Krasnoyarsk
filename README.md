# Прогнозирование уровня концентрации PM 2.5 в атмосфере г. Красноярска  
Красноярск является одним из нескольких городов России с самым грязным воздухом. В связи с этим возникает необходимость прогнозирования уровня концентрации загрязняющих веществ в атмосфере (частиц PM 2.5).  
В данной работе используются среднесуточные данные концентрации PM 2.5, прогнозирование выполнялось при помощи модели ARIMA.  
+ PredFunctions.py содержит класс прогнозирования и вспомогательные функции — поиск по сетке; построение графиков ACF и PACF; сравнение исходного временного ряда с дифференцированным или после преобразования Бокса-Кокса, сами преобразования, а также сравнение истинных значений ряда с прогнозируемыми (на графике, с вычислением метрик качества MAE и $R^2$)
+ Основной файл Forecasting.ipynb последовательно демонстрирует процесс прогнозирования: срез данных -> подбор гиперпараметров p, q и выбор экзогенных переменных -> построение модели и диагностика -> оценка качества и сравнительные графики


На данный момент модель показывает хорошее качество прогнозирования в определённые временные периоды. На данных за зиму 2019-2020 года значения MAE = 15; $R^2$ = 0.82 на обучающей выборке. На тестовой выборке MAE = 7.13, $R^2$ = 0.83.  
В случаях плохого качества прогнозирования добавлялись экзогенные переменные, что позволило улучшить качество до приемлемого. В этом можно убедиться, посмотрев результаты прогнозирования в папке Res - Mean_s.


![](https://github.com/Nikita-Lev/Forecasting-PM-2.5-in-the-atmosphere-of-Krasnoyarsk/blob/main/prediction_demo.gif)  
