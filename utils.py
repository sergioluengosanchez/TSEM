import math 
from decimal import Decimal
import cPickle
import csv
import random
import numpy
import copy


# Stirlings function
# Approximation of the gamma function for big numbers

def stirlings(n):
	return (2*Decimal(math.pi)*n).sqrt() * ((n/Decimal(math.e)) ** n)


def dat2csv(network):
	f = open('dumps/data/Data1_'+network,'r')
	andes = cPickle.load(f)
	f.close()
	f = open('dumps/data/'+network+'.csv','w')
	a = csv.writer(f, delimiter=',')
	a.writerows([andes[0]])
	a.writerows(andes[1])
	f.close()


# Checks if 2 PTs are equal
def pEq(p1,p2):
	flag = True
	for nd1,nd2 in zip(p1.nodes,p2.nodes):
		flag = flag and (set(nd1.parents) == set(nd2.parents))
		flag = flag and (set(nd1.children) == set(nd2.children))
		flag = flag and (set(nd1.childrenPT) == set(nd2.childrenPT))
		flag = flag and (set(nd1.factors) == set(nd2.factors))
		flag = flag and (nd1.nFactor == nd2.nFactor)
		flag = flag and (nd1.parentPT == nd2.parentPT)
		if not flag:
			return flag
	return flag
	
	
def sum1(ci_v):
	res = 0
	for i in range(0,len(ci_v)):
		res_aux = ci_v[i]
		for j in range(0,i):
			res_aux = res_aux * (1-ci_v[j])
		res += res_aux
	return res
	
	

def getPredsBN(bn,i):
	ni = bn.nodes[i]
	pred = set(ni.parents.display())
	pred_aux = set(ni.parents.display())
	while len(pred_aux)>0:
		xj = pred_aux.pop()
		nj = bn.nodes[xj]
		for xp in nj.parents.display():
			if not (xp in pred):
				pred.add(xp)
				pred_aux.add(xp)
	return pred
	
def predec(bn,i):
	pred = []
	predsBN = getPredsBN(bn,i)
	ni = bn.nodes[i]
	lenNodes = bn.nodes.numNds
	for j in xrange(0,lenNodes):
		print 'j: ',j
		if i!=j and not (j in predsBN):
			pred.append(j)
	return pred




# Creates a BN from an adjacency matrix
def adjfile_2_bn(path,name):
	import bnet
	import numpy as np
	f = open(path,'r')
	r = csv.reader(f)
	rows = [row[1:(len(row)+1)] for row in r]
	varNames = rows[0]
	adj = np.matrix(rows[1:(len(row)+1)])
	print varNames
	bn = bnet.BNet(name)
	for i,name in enumerate(varNames):
		node = bnet.Node(name,['0','1'])
		node.parents = np.where(adj[:,i].T=='1')[1].tolist()
		bn.nodes.append(node)
	bn.setChildren()
	return bn


# Creates a BN from an adjacency matrix
def store_bn(bn):
	import cloudpickle
	import copy
	bn2 = copy.deepcopy(bn)
	bn2.bnName = bn.bnName + '_bn1'
	f = open('dumps/networks/' + bn2.bnName,'w')
	cloudpickle.dump(bn2,f)
	f.close()


# Creates a BN from an adjacency matrix
def store_Data(bn,network,lenL,lenT):
	import cloudpickle
	if lenL != 0:
		Data1 = bn.sample_n(lenL)
		f = open('dumps/data/Data1_' + network,'w')
		cloudpickle.dump(Data1,f)
		f.close()
	if lenL != 0:
		Data2 = bn.sample_n(lenT)
		f = open('dumps/data/Data2_' + network,'w')
		cloudpickle.dump(Data2,f)
		f.close()


#Gets var classes from data
def getCatFromData(data):
	classes = [set() for _ in data[0]]
	for di in data[1]:
		for j,dij in enumerate(di):
			classes[j].add(dij)
	return([list(ci) for ci in classes])


#Insert nan at random in the missing_pct % of the values of the data frame df, returning the new data frame df2
def insert_nan_df(df, missing_pct = 0.2):
    df2 = copy.deepcopy(df)
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df2.shape[1])]
    for row, col in random.sample(ix, int(round(missing_pct*len(ix)))):
        df2.iat[row, col] = numpy.nan
    return df2

#Obtains nan rows for the data_frame df
def get_nan_rows(df):
    return numpy.where(df.isnull().any(axis=1))[0].tolist()

#Complete dataframe at random
def complete_df_random(df,classes):
   df_complete = copy.deepcopy(df)
   for i in range(df_complete.shape[1]):
       nan_vals = df_complete.iloc[:,i].isnull()
       input_vals = numpy.random.choice(classes[i],numpy.count_nonzero(nan_vals))
       df_complete.ix[numpy.where(nan_vals)[0],i] = input_vals
   return(df_complete)

def complete_df_mode(df):
    return df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    
def get_unique_rows(df):
    df2 = df.fillna(-1)
    df2 = df2.groupby(df2.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    weights = df2.ix[:,'count'].tolist() 
    return [df2.replace(-1,numpy.nan).drop('count',axis=1),weights]

def get_topological_ordering(et):
    # get nodes without parents
    num_nds = et.nodes.num_nds
    parent_list = [nd.parents.display() for nd in et.nodes]
    has_parent = [len(parent_list[xi])==0 for xi in range(num_nds)]
    t_order = [i for i, x in enumerate(has_parent) if x]
    i = 0
    while len(t_order) < num_nds:
        xi = t_order[i]
        ch = et.nodes[xi].children.display()
        for xj in ch:
            parent_list[xj].remove(xi)
            if len(parent_list[xj])==0:
                t_order.append(xj)
        i += 1
    return t_order
        
    
    
    

    
    
	