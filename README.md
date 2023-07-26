# Прогнозирование уровня концентрации PM 2.5 в атмосфере г. Красноярска  
Красноярск является одним из нескольких городов России с самым грязным воздухом. В связи с этим возникает необходимость прогнозирования уровня концентрации загрязняющих веществ в атмосфере (частиц PM 2.5).  
В данной работе используются среднесуточные данные концентрации PM 2.5, прогнозирование выполнялось при помощи моделей ARIMAX, Random Forest.
+ PredFunctions.py содержит класс прогнозирования и вспомогательные функции — поиск по сетке; построение графиков ACF и PACF; сравнение исходного временного ряда с дифференцированным или после преобразования Бокса-Кокса, сами преобразования, а также сравнение истинных значений ряда с прогнозируемыми (на графике, с вычислением метрик качества MAE и $R^2$)
+ Основной файл Forecasting.ipynb последовательно демонстрирует процесс прогнозирования моделью ARIMAX: срез данных -> подбор гиперпараметров p, q и выбор экзогенных переменных на основе корреляций -> построение модели и диагностика -> оценка качества и сравнительные графики
+ Random Forest.ipynb демонстрирует процесс прогнозирования при помощи случайного леса. Наилучшим оказался случайный лес из 50 деревьев с минимальным количеством объектов для разделения узла = 3 (min_samples_split). В файле Results RF.ipynb приведены таблицы значений ошибок для различных случайных лесов. А для лучшего случайного леса (n_estimators = 50, min_samples_split = 3), добавлены температурные инверсии, как хорошо коррелирующие с PM 2.5 признаки.

На данный момент модели ARIMAX демонстрируют лучшее качество прогнозирования по сравнению со случайным лесом.
В случаях плохого качества прогнозирования добавлялись экзогенные переменные, что позволило улучшить качество до приемлемого: найдётся такая комбинация экзогенных переменных, что качество прогнозирования окажется на приемлемом уровне.


![](https://github.com/Nikita-Lev/Forecasting-PM-2.5-in-the-atmosphere-of-Krasnoyarsk/blob/main/prediction_demo.gif)  
