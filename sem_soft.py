#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:03:38 2018

@author: marcobb8
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:32:35 2017

@author: Sergio Luengo and Marco Benjumeda
"""

# -------------------------------------------------------#
# Learn structure of elimination tree
# -------------------------------------------------------#

from copy import deepcopy
import sys
from time import time
import data_type
from scoring_functions import score_function
from elimination_tree import ElimTree
import multiprocessing
import numpy
from tree_width import greedy_tree_width
from export import get_adjacency_matrix_from_et
from var_elim import PyFactorTree
from utils import get_unique_rows, get_nan_rows, complete_df_random, complete_df_mode
from gs_structure import best_pred_cache, search_first_tractable_structure
from var_elim_data import Indicators, PyFactorTree_data, Indicators_vector


# Tractable SEM
def tsem_cache(data_frame, et0=None, u=5, metric='bic', tw_bound_type='b', tw_bound=5,  cores=multiprocessing.cpu_count(), forbidden_parent=None, add_only = False, custom_classes=None, constraint = True, complete_prior = 'mode', alpha = 1):
    """Learns a multi-dimensional Bayesian network classifier

    Args:
        data_frame (pandas.DataFrame): Input data
        et0 (elimination_tree.ElimTree): Initial elimination tree (optional)
        u (int): maximum number of parents allowed
        metric (str): scoring functions
        tw_bound_type (str): 'b' bound, 'n' none
        tw_bound (float): tree-width bound
        cores (int): Number of cores
        forbidden_parent (list): blacklist with forbidden parents
        add_only (bool): If true, allow only arc additions
        custom_classes: If not None, the classes of each variable are set to custom_classes
        constraint: If true, the additions and reversals that exceed the treewidth bound are stored in a blacklist
        complete_prior (str): complete with mode or at random
        alpha: Dirichlet prior for Bayesian parameter estimation

    Returns:
        elimination_tree.ElimTree Learned elimination tree
    """
    
    count_i = 0
    if custom_classes is None:
        data = data_type.data(data_frame)
    else:
        data = data_type.data(data_frame, classes= custom_classes)
    nan_rows = get_nan_rows(data_frame)
    
    if et0 is None:
        et = ElimTree('et', data.col_names, data.classes)
    else:
        et = et0.copyInfo()
    

    num_nds = et.nodes.num_nds
    
    # Initialize complete dataset
    if complete_prior == 'mode':
        data_frame_complete = complete_df_mode(data_frame)
    else:
        data_frame_complete = complete_df_random(data_frame, data.classes) # Initialize data with random values
    data_complete = data_type.data(data_frame_complete, classes= data.classes)
    
    ftd = PyFactorTree_data([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data.classes])
    ftd.learn_parameters(data_complete, alpha = 0.0)
    ok_to_proceed_em = True
    score_best_ll = 0    
    tla = time()    
    
    num_splits = numpy.ceil(data.nrows / (1000.0*8.0))*8
    indv = Indicators_vector(data_frame, data.classes, num_splits)        


    round_em = -1
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(data.ncols)]
    else:
        forbidden_parents = forbidden_parent
        
        
    num_changes = 0
    num_param_cal = 0
    num_improve_exp = 0
    its_em = 1
    while ok_to_proceed_em:   
        round_em += 1
        ok_to_proceed_em = False
        ok_to_proceed_hc = True
        # Run EM to compute parameters and compute score

        ftd_old = PyFactorTree_data([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data.classes])
        ftd.em_parallel(indv, its_em-1, 0.0)
        ftd_old.copy_parameters(ftd, range(num_nds))
        score_obs = ftd.em_parallel(indv, 1, 0.0)
        num_param_cal += 1

        ftd_mpe = PyFactorTree_data([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data.classes])
        ftd_mpe.copy_parameters(ftd_old, range(num_nds))        
        ftd_mpe.em_parallel(indv, 1, alpha)
        
        # Compute score of the model for the observed data
        for i in range(num_nds):
            score_obs[i] += score_function(data, i, et.nodes[i].parents.display(), metric, [],ll_in = 0)     
        score_best_ll = score_obs
        
        # Impute dataset
        params = ftd_mpe.get_parameters()
        fiti = PyFactorTree([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data.classes])
        fiti.set_parameters([i for i in range(num_nds)],params)
        fiti.mpe_data(data_complete, data, nan_rows)        
        
        score_best = numpy.array([score_function(data_complete, i, et.nodes[i].parents.display(), metric=metric) for i in range(num_nds)],dtype = numpy.double)
        # Cache
        cache = numpy.full([et.nodes.num_nds,et.nodes.num_nds],1000000,dtype=numpy.double)
        #loop hill-climbing
        while ok_to_proceed_hc:
            print "iteration ", count_i
            count_i += 1
            ftd_old = PyFactorTree_data([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data.classes])
            ftd_old.copy_parameters(ftd, range(num_nds)) 
            ta = time()
            score_best_ll = ftd.em_parallel(indv, 1, 0.0)
            num_param_cal += 1

            tb = time()
            # Compute real score
            score_best_ll_real = ftd.log_likelihood_parallel(indv)
            for i in range(num_nds):
                score_best_ll[i] += score_function(data, i, et.nodes[i].parents.display(), metric, [],ll_in = 0)   
                score_best_ll_real += score_function(data, i, et.nodes[i].parents.display(), metric, [],ll_in = 0)  
            # Input, Output and new
            lop_o, op_names_o, score_difs_o, cache =  best_pred_cache(data_complete, et, metric, score_best, forbidden_parents, cache, filter_l0 = True, add_only = add_only)
            
            ok_to_proceed_hc = False
            while len(lop_o) > 0:
                if tw_bound_type=='b':
                    change_idx, et_new = search_first_tractable_structure(et, lop_o, op_names_o, tw_bound, forbidden_parents, constraint)
                else:
                    change_idx, et_new = search_first_tractable_structure(et, lop_o, op_names_o, 10000)   
                if change_idx != -1:
                    best_lop = lop_o[change_idx]
                    xout = best_lop[0]
                    xin = best_lop[1]
                    best_op_names = op_names_o[change_idx]
                    best_score_difs = score_difs_o[change_idx]
                    
                                       
                    #Compute parameters of the new ET
                    ftd_aux = PyFactorTree_data([et_new.nodes[i].parent_et for i in range(et_new.nodes.num_nds)], [et_new.nodes[i].nFactor for i in range(et_new.nodes.num_nds)], [[i]+et_new.nodes[i].parents.display() for i in range(et_new.nodes.num_nds)], [len(c) for c in data.classes])
            
                    if best_op_names == 2:
                        nodes_change = best_lop.tolist()
                    else:
                        nodes_change = [best_lop[1]]

                    nodes_copy = range(num_nds)
                    for xi in nodes_change:
                        nodes_copy.remove(xi)
                    tc = time()
                    score_obs_new = list(score_best_ll)
                    ftd_aux.copy_parameters(ftd, nodes_copy)
                    for xi in nodes_change:
                        score_obs_new[xi] = ftd_aux.em_parameters_parallel(xi, ftd_old, indv, 0.0)
                        num_param_cal += 1
                        score_obs_new[xi] += score_function(data, xi, et_new.nodes[xi].parents.display(), metric, [],ll_in = 0)  
                     
                    # Compute real score
                    score_obs_new_real = ftd_aux.log_likelihood_parallel(indv)
                    for i in range(num_nds):
                        score_obs_new_real += score_function(data, i, et_new.nodes[i].parents.display(), metric, [],ll_in = 0) 
                    
                    td = time()
                    if score_obs_new_real > score_best_ll_real:
                        ok_to_proceed_hc = True
                        ok_to_proceed_em = True
                        ftd_best = None 
                        ftd_best = PyFactorTree_data([et_new.nodes[i].parent_et for i in range(et_new.nodes.num_nds)], [et_new.nodes[i].nFactor for i in range(et_new.nodes.num_nds)], [[i]+et_new.nodes[i].parents.display() for i in range(et_new.nodes.num_nds)], [len(c) for c in data.classes])
                        ftd_best.copy_parameters(ftd_aux, range(num_nds))
                        if sum(score_obs_new) > sum(score_best_ll):
                            num_improve_exp +=1
                                                
                        
                        score_best_ll = score_obs_new
                        et_best = et_new
                        # Update score and cache
                        if best_op_names == 2:
                            score_best[xin] += cache[xout, xin]
                            score_best[xout] += cache[xin, xout]
                            cache[:,xout]=1000000
                        else:
                            score_best[xin] += best_score_difs
                        cache[:,xin]=1000000
                        te = time()
                        lop_o = []
                        num_changes += 1
                        print "iteration ", count_i, ', round: ', round_em, ', change: ', [best_op_names, xout, xin], ', tw: ', et_best.tw()
                    else:
                        if best_op_names == 0:
                            forbidden_parents[best_lop[1]].append(best_lop[0])
                        elif best_op_names == 2:
                            forbidden_parents[best_lop[0]].append(best_lop[1])
                        
                    lop_o = lop_o[(change_idx+1):] 
                    op_names_o = op_names_o[(change_idx+1):] 
                else:
                    ok_to_proceed_hc = False
                    lop_o = []
            if ok_to_proceed_hc:
                ftd = ftd_best
                et = et_best
    
    ftd.em_parallel(indv, 1, alpha)
    tlb = time()            
    # Complete dataset with MPE
    params = ftd.get_parameters()
    fiti = PyFactorTree([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data.classes])
    fiti.set_parameters([i for i in range(num_nds)],params)
    fiti.mpe_data(data_complete, data, nan_rows)
    df_complete = data_complete.to_DataFrame()
    return et, tlb-tla, fiti, df_complete, [num_changes, num_param_cal, float(num_improve_exp)/float(num_param_cal)]
    


def update_forbidden_parents(forbidden_parents, score, scores_new, changes):
    fp = deepcopy(forbidden_parents)
    for score_new, change in zip(scores_new, changes):
        if change[0] == 'add':
            if score_new < sum(score):
                fp[change[2]].append(change[1])

        elif change[0] == 'remove':
            if score_new > sum(score):
                fp[change[2]].append(change[1])
        elif change[0] == 'reverse':
            if score_new > sum(score):
                fp[change[2]].append(change[1])
        else:
            raise ValueError('Undefined change: ' + change[0])
    return fp




