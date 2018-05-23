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
import sem

# Hill climbing algorithm for ets
#



def sem_soft_tractable(data_frame, et0=None, u=5, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1, cores=multiprocessing.cpu_count(), forbidden_parent=None, complete_prior = 'mode', add_only = False, custom_classes=None,alpha=0.01):
    """Soft tractable SEM (TSEM) 

    Args:
        data_frame (pandas.DataFrame): Input data
        et0 (elimination_tree.ElimTree): Initial elimination tree (optional)
        u (int): maximum number of parents
        metric (str): scoring functions
        metric_params (list): Parameters for the scoring function
        chc (bool): If truth efficient learning (constrained hill-climbing) is performed
        tw_bound_type (str): 'b' bound, 'n' none
        tw_bound (float): tree-width bound
        optimization_type (str): 'tw' tries to reduce tree-width, 'size' tries to reduce size. Only used if tw_bound_type!='n'
        k_complex (float): complexity penalization
        cores (int): Number of cores
        forbidden_parent (list): blacklist with forbidden parents
        complete_prior (str): complete with mode or at random
        add_only (bool): If true, allow only arc additions
        custom_classes: If not None, the classes of each variable are set to custom_classes
        alpha: Dirichlet prior for Bayesian parameter estimation
    Returns:
        elimination_tree.ElimTree Learned elimination tree
        float Learning time
        PyFactorTree C++ elimination tree for inference
        df_complete Dataframe completed with hard assignments
        score_best Best score
    """
    
    #Initialize variables
    [df_unique,weights] = get_unique_rows(data_frame)
    
    k = k_complex
    count_i = 0
    nan_rows = get_nan_rows(df_unique)
    if custom_classes is None:
        data_missing = data_type.data(df_unique)
    else:
        data_missing = data_type.data(df_unique, classes= custom_classes)
    rows = range(data_missing.nrows)
        
    if custom_classes is None:
        data_missing_repeat = data_type.data(data_frame)
    else:
        data_missing_repeat = data_type.data(data_frame, classes= custom_classes)
    
    if complete_prior == 'mode':
        data_frame_complete = complete_df_mode(data_frame)
    else:
        print 'init at random'
        data_frame_complete = complete_df_random(data_frame, data_missing.classes) # Initialize data with random values
        print data_missing.classes
        print data_frame_complete
    data_complete = data_type.data(data_frame_complete, classes= data_missing.classes)
    
    if et0 is None:
        et = ElimTree('et', data_missing.col_names, data_missing.classes)
    else:
        et = et0.copyInfo()
    len_nodes = data_complete.ncols
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(data_complete.ncols)]
    else:
        forbidden_parents = forbidden_parent

    # Initialize factor tree
    fiti = PyFactorTree([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data_missing.classes]);
    fiti.learn_parameters(data_complete, alpha = alpha)

    ok_to_proceed = True
    learn_time = 0
    
    score_best = []
    # While there is an improvement
    while ok_to_proceed:
        sys.stdout.write("\r Iteration %d" % count_i)
        sys.stdout.flush()
        count_i += 1
  
        # Learn parameters and obtain ll
        fiti_old = fiti.copy()
        ll = fiti.EM_parameters( data_missing, fiti_old, rows, weights, alpha = alpha)
        score_best = []
        for i in range(0, len_nodes):
            parents = et.nodes[i].parents.display()
            score_best.append(score_function(data_complete, i, parents, metric, metric_params,ll[i], weights = weights))
        # Complexity of the network
        if tw_bound_type != 'n':
            score_best.append(-k * et.evaluate())
        else:
            score_best.append(0)
        
        # Input, Output and new
        ta = time()
        et_new, score_new, time_score, time_compile, time_opt, best_change, forbidden_parents = best_pred_hc(data_missing, et, score_best, forbidden_parents, fiti_old, rows, weights, u, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex, cores, add_only)       
        # If mixed, update order
        if optimization_type == 'mixed' and et_new.tw()>et.tw():
            adj = get_adjacency_matrix_from_et(et_new)
            order, tw_greedy = greedy_tree_width(adj, method='fill')
            if tw_greedy < et_new.tw(False):
                et_new.compile_ordering(order.tolist())
                if tw_bound_type == 'b':
                    if tw_bound >= et_new.tw(False):
                        score_new[-1] = et_new.size() * k_complex
                    else:
                        score_new[-1] = float("-inf")
                elif tw_bound_type == 'p':
                    score_new[-1] = et_new.size() * k_complex
        
        if tw_bound_type=='n':
            adj = get_adjacency_matrix_from_et(et_new)
            order, tw_greedy = greedy_tree_width(adj, method='fill')
            et_new.compile_ordering(order.tolist()) 
        # Complete data
        tb = time()
        print 'score_new: ', sum(score_new), ', score_best: ', sum(score_best)
        if sum(score_new) > sum(score_best):
            print 'change: ', best_change, ', tw: ', et_new.tw(), ', time: ', tb-ta
            et = et_new
            fiti_aux = PyFactorTree([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data_missing.classes])
            nodes_change = []
            if best_change[0] == 'reverse':
                nodes_change = best_change[1:3]
            else:
                nodes_change = [best_change[2]]
            fiti_aux.EM_parameters(data_missing, fiti_old, rows, weights, alpha = alpha, var_update = nodes_change)
            for xi in range(data_missing.ncols):
                if xi not in nodes_change:
                    factor = fiti.get_factor(data_missing.ncols+xi)
                    fiti_aux.set_parameters([xi],[factor.get_prob()])
            fiti = fiti_aux
            score_best = score_new
        else:
            ok_to_proceed = False        
        tb = time()
        learn_time += tb - ta

    # Complete dataset with MPE
    fiti.EM_parameters(data_missing, fiti, rows, weights, alpha = alpha)
    fiti.mpe_data(data_complete, data_missing_repeat, nan_rows)
    df_complete = data_complete.to_DataFrame()
    return et, learn_time, fiti, df_complete, score_best


def sem_soft_fast(data_frame, et0=None, u=5, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1, cores=multiprocessing.cpu_count(), forbidden_parent=None, complete_prior = 'mode', add_only = False, custom_classes=None):
    """Soft tractable SEM (TSEM) using fast heuristic for candidate selection

    Args:
        data_frame (pandas.DataFrame): Input data
        et0 (elimination_tree.ElimTree): Initial elimination tree (optional)
        u (int): maximum number of parents
        metric (str): scoring functions
        metric_params (list): Parameters for the scoring function
        chc (bool): If truth efficient learning (constrained hill-climbing) is performed
        tw_bound_type (str): 'b' bound, 'n' none
        tw_bound (float): tree-width bound
        optimization_type (str): 'tw' tries to reduce tree-width, 'size' tries to reduce size. Only used if tw_bound_type!='n'
        k_complex (float): complexity penalization
        cores (int): Number of cores
        forbidden_parent (list): blacklist with forbidden parents
        complete_prior (str): complete with mode or at random
        add_only (bool): If true, allow only arc additions
        custom_classes: If not None, the classes of each variable are set to custom_classes
        alpha: Dirichlet prior for Bayesian parameter estimation
    Returns:
        elimination_tree.ElimTree Learned elimination tree
        float Learning time
        PyFactorTree C++ elimination tree for inference
        df_complete Dataframe completed with hard assignments
        score_best Best score
    """
    
    
    #Initialize variables
    [df_unique,weights] = get_unique_rows(data_frame)
    
    k = k_complex
    count_i = 0
    nan_rows = get_nan_rows(df_unique)
    if custom_classes is None:
        data_missing = data_type.data(df_unique)
    else:
        data_missing = data_type.data(df_unique, classes= custom_classes)
    rows = range(data_missing.nrows)
        
    if custom_classes is None:
        data_missing_repeat = data_type.data(data_frame)
    else:
        data_missing_repeat = data_type.data(data_frame, classes= custom_classes)
    
    if complete_prior == 'mode':
        data_frame_complete = complete_df_mode(data_frame)
    else:
        data_frame_complete = complete_df_random(data_frame, data_missing.classes) # Initialize data with random values
    data_complete = data_type.data(data_frame_complete, classes= data_missing.classes)
    
    if et0 is None:
        et = ElimTree('et', data_missing.col_names, data_missing.classes)
    else:
        et = et0.copyInfo()
    len_nodes = data_complete.ncols
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(data_complete.ncols)]
    else:
        forbidden_parents = forbidden_parent

    # Initialize factor tree
    fiti = PyFactorTree([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in data_missing.classes]);
    fiti.learn_parameters(data_complete, alpha = 1)

    ok_to_proceed = True
    learn_time = 0
    
    score_best = []
    # While there is an improvement
    while ok_to_proceed:
        sys.stdout.write("\r Iteration %d" % count_i)
        sys.stdout.flush()
        count_i += 1
  
        # Learn parameters and obtain ll
        fiti_old = fiti.copy()
        ll = fiti.EM_parameters_distribute( data_missing, fiti_old, rows, weights, alpha = 0.01)
        score_best = []
        score_best_mpe = []
        for i in range(0, len_nodes):
            parents = et.nodes[i].parents.display()
            score_best.append(score_function(data_complete, i, parents, metric, metric_params,ll[i], weights = weights))
            score_best_mpe.append(score_function(data_complete, i, parents, metric, metric_params))
        # Complexity of the network
        if tw_bound_type != 'n':
            score_best.append(-k * et.evaluate())
            score_best_mpe.append(-k * et.evaluate())
        else:
            score_best.append(0)
            score_best_mpe.append(-k * et.evaluate())
              
        # Input, Output and new
        ta = time()
        et_new, score_new_mpe, time_score, time_compile, time_opt, best_change, forbidden_parents = sem.best_pred_hc(data_complete, et, score_best_mpe, forbidden_parents, u, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex, cores, add_only)
        score_new = deepcopy(score_best)
        # If mixed, update order
        if optimization_type == 'mixed' and et_new.tw()>et.tw():
            adj = get_adjacency_matrix_from_et(et_new)
            order, tw_greedy = greedy_tree_width(adj, method='fill')
            if tw_greedy < et_new.tw(False):
                et_new.compile_ordering(order.tolist())
                if tw_bound_type == 'b':
                    if tw_bound >= et_new.tw(False):
                        score_new[-1] = et_new.size() * k_complex
                    else:
                        score_new[-1] = float("-inf")
                elif tw_bound_type == 'p':
                    score_new[-1] = et_new.size() * k_complex
        
        if tw_bound_type=='n':
            adj = get_adjacency_matrix_from_et(et_new)
            order, tw_greedy = greedy_tree_width(adj, method='fill')
            et_new.compile_ordering(order.tolist()) 
        # Complete data
        tb = time()
        
        if sum(score_new_mpe) > sum(score_best_mpe):
            fiti_aux = PyFactorTree([et_new.nodes[i].parent_et for i in range(et_new.nodes.num_nds)], [et_new.nodes[i].nFactor for i in range(et_new.nodes.num_nds)], [[i]+et_new.nodes[i].parents.display() for i in range(et_new.nodes.num_nds)], [len(c) for c in data_missing.classes])
            nodes_change = []
            if best_change[0] == 'reverse':
                nodes_change = best_change[1:3]
            else:
                nodes_change = [best_change[2]]
            score_aux = fiti_aux.EM_parameters(data_missing, fiti_old, rows, weights, alpha = 0.01, var_update = nodes_change)
            for i,xi in enumerate(nodes_change):
                score_new[xi] = score_aux[i]
            for xi in range(data_missing.ncols):
                if xi not in nodes_change:
                    factor = fiti.get_factor(data_missing.ncols+xi)
                    fiti_aux.set_parameters([xi],[factor.get_prob()])
            
            if sum(score_new) > sum(score_best):
    
                print 'change: ', best_change, ', tw: ', et_new.tw(), ', time: ', tb-ta
                et = et_new
    
                fiti = fiti_aux.copy()
                fiti.mpe_data(data_complete, data_missing_repeat, nan_rows)
                score_best = score_new
            else:
                ok_to_proceed = False  
        else:
            ok_to_proceed = False
        tb = time()
        learn_time += tb - ta

    # Complete dataset with MPE
    fiti.EM_parameters(data_missing, fiti, rows, weights, alpha = 0.1)
    fiti.mpe_data(data_complete, data_missing_repeat, nan_rows)
    df_complete = data_complete.to_DataFrame()
    return et, learn_time, fiti, df_complete, score_best

def best_pred_hc(data, et, score, forbidden_parents, fiti, rows, weights, u=5, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1, cores=multiprocessing.cpu_count(), add_only = False):
    change_lst = []

    # 1. Possible edges to add
    for i in range(et.nodes.num_nds):
        prd = et.predec(i)
        if et.nodes[i].parents.len_py() < u:
            for pred in prd:
                if (not (pred in forbidden_parents[i])) and (et.nodes[i].parents.len_py() < u):
                    change_lst.append(['add', pred, i])
    if not add_only:
        # 2. Possible edges to remove
        for i in range(et.nodes.num_nds):
            for j in et.nodes[i].parents.display():
                change_lst.append(['remove', j, i])
    
        # 3. Possible edges to reverse
        for i in range(et.nodes.num_nds):
            for j in et.nodes[i].parents.display():
                et_aux = et.copyInfo()
                et_aux.removeArcBN_py(j, i)
                if (not et_aux.isAnt(j, i)) and (et.nodes[j].parents.len_py() < u) and (not (i in forbidden_parents[j])):
                    change_lst.append(['reverse', j, i])

    # 4. Apply local changes 
    if len(change_lst) == 0:
        return et, score, 0, 0, 0, [], forbidden_parents
    best_result = local_changes_par(min(cores, len(change_lst)), change_lst, data, et, score, forbidden_parents, fiti, rows, weights, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex)
    #best_result = local_changes_par(1, change_lst, data, et, score, forbidden_parents, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex)
    
    return best_result


def local_changes_par(cores, change_lst, data, et, score, forbidden_parents, fiti, rows, weights, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1):
    queues = []
    processes = []
    for iproc in range(cores):
        queues.append(multiprocessing.Queue())
        processes.append(multiprocessing.Process(target=local_changes_par_step, args=(
            queues[iproc], cores, iproc, change_lst, data, et, score, forbidden_parents, fiti, rows, weights, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex)))
        processes[iproc].start()

    results_all = []

    for proc, que in zip(processes, queues):
        results_all.append(que.get())
        proc.join()

    scores = [sum(res[1]) for res in results_all]
    best_idx = numpy.argmax(scores)
    best_result = results_all[best_idx]
    best_change = best_result[5]
    time_score = sum([res[2] for res in results_all])
    time_comp = sum([res[3] for res in results_all])
    time_opt = sum([res[4] for res in results_all])
    best_fp = []
    for i in range(data.ncols):
        fpi = forbidden_parents[i]
        if chc:
            for result in results_all:
                fpi = fpi + result[-1][i]
            fpi = list(set(fpi))
        best_fp.append(fpi)

    return best_result[0], best_result[1], time_score, time_comp, time_opt, best_change, best_fp


def local_changes_par_step(queue, nproc, iproc, change_lst, data, et, score, forbidden_parents, fiti, rows, weights, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1):

    first = (iproc) * len(change_lst)/nproc
    last = (iproc+1) * len(change_lst) / nproc
    
    # Initialize
    scores = []
    best_score = None
    time_score = 0
    time_comp = 0
    time_opt = 0
    best_et = None
    best_change = change_lst[0]
    for i in range(first, last):
        change = change_lst[i]
        res = local_change(change, data, et, score, fiti, rows, weights, metric, metric_params, tw_bound_type, tw_bound, optimization_type, k_complex)
        scores.append(sum(res[1]))
        if (best_et is None) or  (sum(res[1]) > sum(best_score)):
            best_score = res[1]
            best_et = res[0]
            best_change = change
        time_score += res[2]
        time_comp += res[3]
        time_opt += res[4]

    if chc:
        best_fp = update_forbidden_parents(forbidden_parents, score, scores, [change_lst[i] for i in range(first, last)])
    else:
        best_fp = forbidden_parents

    queue.put([best_et, best_score, time_score, time_comp, time_opt, best_change, best_fp])


def local_change(change, data, et, score, fiti, rows, weights, metric='bic', metric_params=None, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1):
    time_comp = 0
    time_opt = 0
    et_new = et.copyInfo()
    score_new = deepcopy(score)
    if change[0] == 'add':
        if tw_bound_type == 'n':
            et_new.setArcBN_py(change[1], change[2])
        else:
            time1 = time()
            x_opt = et_new.add_arc(change[1], change[2])
            time2 = time()
            et_new.optimize_tree_width(x_opt)
            if tw_bound_type == 'b':
                if tw_bound >= et_new.tw(False):
                    score_new[-1] = -1* et_new.size() * k_complex
                else:
                    score_new[-1] = float("-inf")
            elif tw_bound_type == 'p':
                score_new[-1] = -1* et_new.size() * k_complex

            time3 = time()
            time_comp = time2 - time1
            time_opt = time3 - time2
        ll = fiti.log_lik_se_cluster(data, rows, weights, [change[2]] + et_new.nodes[change[2]].parents.display())     
        score_new[change[2]] = score_function(data, change[2], et_new.nodes[change[2]].parents.display(), metric, metric_params,ll_in = ll, weights = weights)
        time_score = time() - time1

    elif change[0] == 'remove':
        if tw_bound_type == 'n':
            et_new.removeArcBN_py(change[1],change[2])
        else:
            time1 = time()
            x_opt = et_new.remove_arc(change[1],change[2])
            time2 = time()
            et_new.optimize_tree_width(x_opt)
            if tw_bound_type == 'b':
                if tw_bound >= et_new.tw(False):
                    score_new[-1] = -1*et_new.size() * k_complex
                else:
                    score_new[-1] = float("-inf")
            elif tw_bound_type == 'p':
                score_new[-1] = -1*et_new.size() * k_complex
            time3 = time()
            time_comp = time2 - time1
            time_opt = time3 - time2
        time1 = time()
        ll = fiti.log_lik_se_cluster(data, rows, weights, [change[2]] + et_new.nodes[change[2]].parents.display())
        score_new[change[2]] = score_function(data, change[2], et_new.nodes[change[2]].parents.display(), metric, metric_params,ll_in = ll, weights = weights)
        time_score = time() - time1

    elif change[0] == 'reverse':
        if tw_bound_type == 'n':
            et_new.removeArcBN_py(change[1], change[2])
            et_new.setArcBN_py(change[2], change[1])
        else:
            # remove
            time1 = time()
            x_opt = et_new.remove_arc(change[1], change[2])
            time2 = time()
            et_new.optimize_tree_width(x_opt)
            time3 = time()
            time_comp = time2 - time1
            time_opt = time3 - time2

            time1 = time()
            x_opt = et_new.add_arc(change[2], change[1])
            time2 = time()
            et_new.optimize_tree_width(x_opt)
            if tw_bound_type == 'b':
                if tw_bound >= et_new.tw(False):
                    score_new[-1] = -1*et_new.size() * k_complex
                else:
                    score_new[-1] = float("-inf")
            elif tw_bound_type == 'p':
                score_new[-1] = -1*et_new.size() * k_complex
            time3 = time()
            time_comp = time_comp + time2 - time1
            time_opt = time_opt + time3 - time2

        time1 = time()
        ll = fiti.log_lik_se_cluster(data, rows, weights, [change[2]] + et_new.nodes[change[2]].parents.display())
        score_new[change[2]] = score_function(data, change[2], et_new.nodes[change[2]].parents.display(), metric, metric_params,ll_in = ll, weights = weights)
        ll = fiti.log_lik_se_cluster(data, rows, weights, [change[1]] + et_new.nodes[change[1]].parents.display())
        score_new[change[1]] = score_function(data, change[1], et_new.nodes[change[1]].parents.display(), metric, metric_params,ll_in = ll, weights = weights)
        time_score = time() - time1
    else:
        raise ValueError('Undefined change: '+change[0])

    return [et_new, score_new, time_score, time_comp, time_opt, change]


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




