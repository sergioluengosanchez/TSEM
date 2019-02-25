# -------------------------------------------------------#
# Learn structure of elimination tree
# -------------------------------------------------------#

from copy import deepcopy
import sys
from time import time
import cloudpickle
import data_type
from scoring_functions import score_function
from elimination_tree import ElimTree
import multiprocessing
import numpy
from tree_width import greedy_tree_width
from export import get_adjacency_matrix_from_et

# Hill climbing algorithm for ets
#


def hill_climbing(data_frame, et0=None, u=5, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1, cores=multiprocessing.cpu_count(), forbidden_parent=None):
    """Gets the adjacency matrix of the Bayesian network encoded by the elimination tree et

    Args:
        data_frame (pandas.DataFrame): Input data
        et0 (elimination_tree.ElimTree): Initial limination tree
        u (int): maximum number of parents
        metric (str): scoring functions
        metric_params (list): Parameters for the scoring function
        chc (bool): If truth efficient learning (constrained hill-climbing) is performed
        tw_bound_type (str): 'b' bound, 'n' none
        tw_bound (float): tree-width bound
        k_complex (float): complexity penalization
        optimization_type (str): 'tw' tries to reduce tree-width, 'size' tries to reduce size. Only used if tw_bound_type!='n'
        cores (int): Number of cores
        forbidden_parent (list): balcklist with forbidden parents

    Returns:
        elimination_tree.ElimTree Learned elimination tree
        float Learning time
    """
        
    #Initialize variables
    k = k_complex
    count_i = 0
    data = data_type.data(data_frame)
    if et0 is None:
        et = ElimTree('et', data.col_names, data.classes)
    else:
        et = et0.copyInfo()
    len_nodes = data.ncols
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(data.ncols)]
    else:
        forbidden_parents = forbidden_parent

    # Initialize score_best := Array with the metric value for all the variables and the complexity penalization
    score_best = []
    for i in range(0, len_nodes):
        parents = et.nodes[i].parents.display()
        score_best.append(score_function(data, i, parents, metric, metric_params))

    # Complexity of the network
    if tw_bound_type != 'n':
        score_best.append(-k * et.evaluate())
    else:
        score_best.append(0)

    ok_to_proceed = True
    learn_time = 0

    # While there is an improvement
    while ok_to_proceed:
        sys.stdout.write("\r Iteration %d" % count_i)
        sys.stdout.flush()
        count_i += 1

        # Input, Output and new
        ta = time()
        et_new, score_new, time_score, time_compile, time_opt, best_change, forbidden_parents = best_pred_hc(data, et, score_best, forbidden_parents, u, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex, cores)
        
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
            
        tb = time()
        time_total = time_score + time_compile + time_opt
        print 'total time: ', time_total, ', Score Time: ', time_score, ', Compile Time: ', time_compile, ', Opt time: ', time_opt, ', pct: ', float(time_score) / float(time_total)
        print 'change: ', best_change, ', tw: ', et_new.tw(False), ', tw again: ', et_new.tw()
        learn_time += tb - ta
        if sum(score_new) > sum(score_best):
            et = et_new
            score_best = score_new
        else:
            ok_to_proceed = False
        f = open('learn_aux_dump', 'w')
        flag_it1 = et0 is None
        cloudpickle.dump([et, et_new, forbidden_parents, learn_time, flag_it1], f)
        f.close()
        print 'return tw: ', et.tw()

    return et, learn_time, score_best


def best_pred_hc(data, et, score, forbidden_parents, u=5, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1, cores=multiprocessing.cpu_count()):
    change_lst = []

    # 1. Possible edges to add
    for i in range(et.nodes.num_nds):
        prd = et.predec(i)
        if et.nodes[i].parents.len_py() < u:
            for pred in prd:
                if (not (pred in forbidden_parents[i])) and (et.nodes[i].parents.len_py() < u):
                    change_lst.append(['add', pred, i])

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
    best_result = local_changes_par(cores, change_lst, data, et, score, forbidden_parents, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex)
    #best_result = local_changes_par(1, change_lst, data, et, score, forbidden_parents, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex)
    
    return best_result


def local_changes_par(cores, change_lst, data, et, score, forbidden_parents, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1):
    queues = []
    processes = []
    for iproc in range(cores):
        queues.append(multiprocessing.Queue())
        processes.append(multiprocessing.Process(target=local_changes_par_step, args=(
            queues[iproc], cores, iproc, change_lst, data, et, score, forbidden_parents, metric, metric_params, chc, tw_bound_type, tw_bound, optimization_type, k_complex)))
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


def local_changes_par_step(queue, nproc, iproc, change_lst, data, et, score, forbidden_parents, metric='bic', metric_params=None, chc=False, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1):

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
        res = local_change(change, data, et, score, metric, metric_params, tw_bound_type, tw_bound, optimization_type, k_complex)
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


def local_change(change, data, et, score, metric='bic', metric_params=None, tw_bound_type='b', tw_bound=5, optimization_type='size', k_complex=0.1):
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
        time1 = time()
        score_new[change[2]] = score_function(data, change[2], et_new.nodes[change[2]].parents.display(), metric, metric_params)
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
        score_new[change[2]] = score_function(data, change[2], et_new.nodes[change[2]].parents.display(), metric, metric_params)
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
        score_new[change[2]] = score_function(data, change[2], et_new.nodes[change[2]].parents.display(), metric, metric_params)
        score_new[change[1]] = score_function(data, change[1], et_new.nodes[change[1]].parents.display(), metric, metric_params)
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




