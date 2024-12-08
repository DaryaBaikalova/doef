from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math


def optimal_func(t_start, t_end, y_start, y_end, f, y_0=0.5, num_of_fractions=51):
    t = np.linspace(t_start, t_end, num_of_fractions)   # Дискретная шкала времени
    dt = t[1] - t[0]

    y0 = np.full(num_of_fractions, y_0)    # Начальное значение y

    # Задаем ограничения на y
    bounds = np.full((num_of_fractions, 2), (None, None)) # Границы для ограничения снизу и сверху соответственно
    bounds[0], bounds[-1] = (y_start, y_start), (y_end, y_end)

    res = minimize(lambda y: f(y, t, dt), y0, method='l-bfgs-b', bounds=bounds)

    return t, res.x
    
    


