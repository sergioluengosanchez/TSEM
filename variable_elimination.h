#ifndef VARIABLE_ELIMINATION_H
#define VARIABLE_ELIMINATION_H

#include <math.h>
#include <omp.h>
#include <vector>
#include <memory>

class Factor {
  public:
  	Factor();
  	Factor(std::vector<int> variables, std::vector<int> num_categories);
  	~Factor();
  
  	/*Learns CPT of the first variable in variables vector conditioned to its parents, that are the rest of variables in variables vector*/
  	void learn_params(const int* data, int nrows, int ncols, double alpha);
     double learn_params_se(std::vector<double> se, double alpha);
     double log_lik_se(std::vector<double> se, double alpha);
  	std::vector<int> variables; // Array with the ids of the variables ordered bottom up
  	std::vector<int> num_categories;
  	std::vector<double> prob; // Array with the probabilities 
  	std::vector<std::vector<int> > mpe; // Most-probable instances
};


/* Node of a factor tree*/
class Factor_Node {
  public:
    Factor_Node();
    Factor_Node(const Factor_Node& fnode);
    Factor_Node(std::vector<int> variables, std::vector<int> num_categories, int id);
    Factor_Node(std::shared_ptr<Factor> factor, int id);
    Factor_Node(int id);
    ~Factor_Node();
    
  /*Attributes*/
  std::shared_ptr<Factor> factor;
  std::shared_ptr<Factor> joint_factor;
  std::vector<std::weak_ptr<Factor_Node>> children;
  std::weak_ptr<Factor_Node> parent; 
  int id;
  std::shared_ptr<Factor> factor_bup;
};
        
class Factor_Tree {
  public:
  	Factor_Tree();
  	//set of param tables, 
  	Factor_Tree(std::vector<int> parents_inner, std::vector<int> parents_leave, std::vector<std::vector<int>> vec_variables_leave, std::vector<int> num_categories);
     Factor_Tree(const Factor_Tree& ft);
  	virtual ~Factor_Tree();
  	/*Learns CPTs of the leave nodes*/
  	void learn_parameters(const int* data, int nrows, int ncols, double alpha);
  	/*Computes the factor of xi from the factors of its children (MPE)*/
  	void max_compute(int xi);
  	/*Computes the factor of xi from the factors of its children (evidence propagation)*/
  	void sum_compute(int xi);
    /*Distributes evidence (Junction Tree)*/
    void distribute_evidence();
    void distribute_evidence(int xi, std::shared_ptr<Factor> message);
    void distribute_leaves(std::vector<int> evidence, std::vector<int> values, int ncols);
  	/*Computes the product of the children of xi*/
    void prod_compute(int xi);
  	/*Computes the factors all nodes (MPE)*/
  	void max_compute_tree();
  	/*Computes the factors all nodes (evidence propagation)*/
  	void sum_compute_tree();
    /*Computes the joint PD of variables*/
  	void sum_compute_query( const std::vector<bool> variables);
      
    /*Sets evidence in variables to values*/
    void set_evidence(const std::vector<int> variables, const std::vector<int> values);
    /*Retracts the evidences from the network*/
    void retract_evidence();
    /*Completes data_complete */
    void mpe_data(int* data_complete, const int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<double> &prob_mpe);
    /*Get the sufficent statistics for cluster*/
    std::vector<double> se_data(int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<int> weights, std::vector<int> cluster_int);
    /*Get the sufficent statistics for cluster (parallel)*/
    std::vector<double> se_data_parallel(int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<int> weights, std::vector<int> cluster_int);
    /*Get the sufficent statistics for each node*/
    std::vector<std::vector<double>> se_data_distribute(int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<int> weights);
    /*Predicts marginals for each row in data_complete */
    std::vector<std::vector<float>> pred_data(int* data, int ncols, int nrows, std::vector<int> qvalues, std::vector<int> query, std::vector<int> evidence);
    /*Predicts marginals for each row in data_complete */
    double cll(int* data, int ncols, int nrows, std::vector<int> query, std::vector<int> weights);
    /*Predicts marginals for each row in data_complete */
    double cll_parallel(int* data, int ncols, int nrows, std::vector<int> query, std::vector<int> weights);    
    
    /* Compute the loglikelihood of the model for the observed data */
    double ll_parallel(int* data, int ncols, int nrows);
    /* Attributes */
    std::vector<std::shared_ptr<Factor_Node>> inner_nodes, leave_nodes;
    std::shared_ptr<Factor_Node> root;
    
    void aux_omp();
    
};        
        
/*Marginalizes factor f1 over variable xi*/
std::shared_ptr<Factor> sum_factor(const Factor &f1, int xi); 

/*Obtains the product of factors*/
std::shared_ptr<Factor> prod_factor(const std::vector<std::shared_ptr<Factor>> factors);

/*Maximizes factor f1 over variable xi*/
std::shared_ptr<Factor> max_factor(const Factor &f1, int xi);

/*Eliminates variable xi (MPE)*/
std::shared_ptr<Factor> max_elim(const std::vector<std::shared_ptr<Factor>> factors, int xi);

/*Eliminates variable xi (evidence propagation)*/
std::shared_ptr<Factor> sum_elim(const std::vector<std::shared_ptr<Factor>> factors, int xi);

/* The log-likelihood is returned as the output*/
double log_lik_se(std::vector<double> se, double alpha, std::vector<int> cluster, std::vector<int> num_categories);

// Multiply vector of vectors
std::vector<double> mul_vectors(std::vector<std::vector<double>> vectors_in);
void mul_arrays(double* array_in, double* array1, double* array2, unsigned len);

// Sum vector of vectors
std::vector<double> sum_vectors(std::vector<std::vector<double>> vectors_in);


#endif
