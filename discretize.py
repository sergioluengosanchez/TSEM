
"""
Discretize the continuous variables using the equal frequency intervals 
"""
import pandas
import numpy as np
from utils import get_unique_rows, get_nan_rows, complete_df_random, complete_df_mode
import data_type


import pandas.core.algorithms as algos
from copy import deepcopy
from collections import Counter

def test_disc():
    from discretize import fixed_freq_disc, fixed_freq_disc_test
    path = "/media/marcobb8/linux_hdd/Dropbox/Phd/Epilepsy/code_py/dat_train.csv"
    data_train = pandas.read_csv(path)
    path = "/media/marcobb8/linux_hdd/Dropbox/Phd/Epilepsy/code_py/dat_test.csv"
    data_test = pandas.read_csv(path)
    q_vars = [10,11,12,26,27,29,30,46,47]
    freq = 30
    data_train_disc,  cut_list   =  fixed_freq_disc(data_train , q_vars, freq = 30)
    data_test_disc = fixed_freq_disc_test(data_test , q_vars, cut_list)




def get_eq_freq_int(values_in, cutpoints):
    values = deepcopy(values_in)
    for j in range(len(values)):
        if not pandas.isnull(values[j]):
            cat = 0
            while cat<len(cutpoints) and values[j] > cutpoints[cat]:
                cat = cat + 1
            values[j] = cat
    return values
    

# Discretizes the quantitative variables using the fixed frequency approach
def fixed_freq_disc(data , q_vars, freq = 30):
    
    data_disc = deepcopy(data)
    cut_list = []
    for i in range(len(q_vars)):
        ncuts = np.floor((data.shape[0] - sum(pandas.isnull(data.iloc[:,q_vars[i]]))) / freq)
#        bins = np.unique(algos.quantile(data.iloc[:,q_vars[i]], np.linspace(0, 1,int(ncuts)+2)))
#        result = pandas.tools.tile._bins_to_cuts(ser, bins, include_lowest=True)         
#        
#        
#        discrete, cutoffs = discretize(data.iloc[:,q_vars[i]],int(ncuts))
        values = deepcopy(data.iloc[:,q_vars[i]])
#        
        disc=pandas.qcut(values,int(ncuts), duplicates='drop')
        cutpoints = [x.right for x in disc.cat.categories][:-1]
        cut_list.append(cutpoints)
        values = get_eq_freq_int(values,cutpoints)
        data_disc.iloc[:,q_vars[i]] = values
    return data_disc, cut_list
    
    
def fixed_freq_disc_test(data , q_vars, cut_list):
    data_disc = deepcopy(data)
    for i in range(len(q_vars)):
        values = get_eq_freq_int(data.iloc[:,q_vars[i]],cut_list[i])
        data_disc.iloc[:,q_vars[i]] = values
    return data_disc
        
        
    

