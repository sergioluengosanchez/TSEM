#ifndef VARIABLE_ELIMINATION_H
#define VARIABLE_ELIMINATION_H

#include <math.h>
#include <omp.h>
#include <vector>
#include <memory>

class Factor_data {
    public:
        Factor_data();
        Factor_data(std::vector<int> &variables, std::vector<int> &num_categories, unsigned data_size);
        Factor_data(const Factor_data &fd);
        ~Factor_data();
        void set_params_evidence(std::vector<double> parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators, std::vector<int> &num_categories);
  
    /*Learns CPT of the first variable in variables vector conditioned to its parents, that are the rest of variables in variables vector*/
    std::vector<int> variables; // Array with the ids of the variables ordered bottom up
//    std::vector<std::vector<double>> prob; // Array with the probabilities 
    double* prob; // Array with the probabilities 
    unsigned num_conf, data_size;
};


class Factor_Node_data {
  public:
    Factor_Node_data();
    Factor_Node_data(const Factor_Node_data& fnode);
    Factor_Node_data(std::vector<int> &variables, std::vector<int> &num_categories, int id, unsigned ninst);
    Factor_Node_data(std::shared_ptr<Factor_data> factor, int id);
    Factor_Node_data(int id);
    ~Factor_Node_data();
    
  /*Attributes*/
  std::shared_ptr<Factor_data> factor;
  std::shared_ptr<Factor_data> joint_factor;
  std::vector<std::weak_ptr<Factor_Node_data>> children;
  std::weak_ptr<Factor_Node_data> parent; 
  int id;
};


class Factor_Tree_data {
  public:
    Factor_Tree_data();
    //set of param tables, 
    Factor_Tree_data(std::vector<int> parents_inner, std::vector<int> parents_leave, std::vector<std::vector<int>> vec_variables_leave, std::vector<int> num_categories);
//    Factor_Tree_data(const Factor_Tree_data& ft);
    virtual ~Factor_Tree_data();
    /*Computes the factor of xi from the factors of its children (evidence propagation)*/
    void sum_compute(int xi);
    /*Computes the product of the children of xi*/
    void prod_compute(int xi);
    /*Computes the factors all nodes (evidence propagation)*/
    void sum_compute_tree(std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators);
    /*Computes the joint PD of variables*/
    void sum_compute_query( std::vector<bool> &variables, std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators);
    /*Get the sufficent statistics for cluster*/
    std::vector<double> se_data(std::vector<int> &cluster_int, std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators);
    std::vector<double> se_data_parallel(std::vector<int> &cluster_int, std::vector<std::vector<double>> &parameters, std::vector<std::vector<std::shared_ptr<std::vector<std::vector<int>>>>> &indicators_vector);

    /*Distributes evidence (Junction Tree)*/
    void distribute_evidence();
    void distribute_evidence(int xi, std::shared_ptr<Factor_data> message);
    std::vector<std::vector<double>> se_leaves();

    /* Perform num_it iterations of em */
    std::vector<double> em(std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators, int num_it, double alpha);
    std::vector<double> em_parallel(std::vector<std::vector<std::shared_ptr<std::vector<std::vector<int>>>>> &indicators_vector, int num_it, double alpha);

    /* Learn parameters from se */
    double learn_params_se(std::vector<double> &se, double alpha, unsigned xi);
    /*Learns parameters from data*/
    void learn_params(const int* data, int nrows, int ncols, double alpha, unsigned xi);
    void learn_params(const int* data, int nrows, int ncols, double alpha);
    /*Return log-likelihood of the indicators given the model*/
    double log_likelihood(std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators);
    double log_likelihood_parallel(std::vector<std::vector<std::shared_ptr<std::vector<std::vector<int>>>>> &indicators_vector);
    std::shared_ptr<Factor_Tree_data> copy();
    /* Attributes */
    std::vector<std::shared_ptr<Factor_Node_data>> inner_nodes, leave_nodes;
    std::shared_ptr<Factor_Node_data> root;
    std::vector<std::vector<int>> variables;
    std::vector<int> num_categories;
    unsigned data_size = 0;
    std::vector<std::vector<double>> parameters;

};   


// Generate indicators matrix from data
std::vector<std::shared_ptr<std::vector<std::vector<int>>>> generate_indicators(int* data, int nrows, int ncols, std::vector<int> num_categories);
// Generate indicators matrix from data, filling in variables with na
std::vector<std::shared_ptr<std::vector<std::vector<int>>>> generate_indicators(int* data, int nrows, int ncols, std::vector<int> num_categories, std::vector<int> variables);

// Multiply vector of vectors
std::vector<double> mul_vectors(std::vector<std::vector<double>> vectors_in);
void mul_arrays(double* array_in, double* array1, double* array2, unsigned len);

// Sum vector of vectors
std::vector<double> sum_vectors(std::vector<std::vector<double>> vectors_in);

void example_dlib();


#endif
