import pandas as pd
import numpy as np

avg_final_profit_normal = [
    540131500,
    558786700,
    489545500,
    298821700,
    573589000,
    578604700,
    512332100,
    309286300
]

std_final_profit_normal = [
    3978059,
    1908504,
    4219279,
    2127035,
    4485011,
    2305987,
    4884570,
    2299039
]

t_value = 2.145

results = []

for i in range(len(avg_final_profit_normal)):
    avg = avg_final_profit_normal[i]
    std = std_final_profit_normal[i]
    n = 15

    h = t_value * (std / np.sqrt(n))

    percentage_of_average = 0.05 * avg

    # Compare
    print("Percent of average", percentage_of_average)
    print("H value", h)
    print()
    if percentage_of_average < h:
        print("!!!!Higher for index", i)
    
