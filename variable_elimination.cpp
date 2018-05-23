/// Code your testbench here.
// Uncomment the next line for SystemC modules.
// #include <systemc>

#include <cstdlib>     /* malloc, free, rand */
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include "variable_elimination.h"
#include <new>
#include <fstream>
#include <math.h> 


void  print_vector(std::vector<int> v){
    std::cout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) std::cout << *i << ", ";
    std::cout << "]\n";
}

void  print_vector(std::vector<std::vector<int>> v){
    std::cout << "[";
    for (auto i= v.begin(); i != v.end(); ++i){
        print_vector(*i);
        std::cout << ",";
    }
    std::cout << "]\n";
}


void  print_vector(std::vector<double> v){
    std::cout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) std::cout << *i << ", ";
    std::cout << "double]\n";
}

void  print_vector(std::vector<float> v){
    std::cout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) std::cout << *i << ", ";
    std::cout << "double]\n";
}

void  print_vector(std::vector<bool> v){
    std::cout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) std::cout << *i << ", ";
    std::cout << "bool]\n";
}



void  print_vector_f(std::vector<double> v){
    std::ofstream fout("filename.txt", std::ios::app);
    fout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) fout << *i << ", ";
    fout << "double]\n";
}

void  print_vector_f(std::vector<int> v){
    std::ofstream fout("filename.txt", std::ios::app);
    fout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) fout << *i << ", ";
    fout << "]\n";
}


void  print_vector_f(std::vector<bool> v){
    std::ofstream fout("filename.txt", std::ios::app);
    fout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) fout << *i << ", ";
    fout << "]\n";
}

/* Returns the position of the current data row in value to increment
 * its value */
int position(const int* d_row, const int* vars, const int* ncat, int l_vars, int nrows){
    int pos,times,i;
    pos = 0;
    times = 1;
    for ( i = 0; i < l_vars; i++ ){
        if (d_row[vars[i]*nrows]<= ncat[vars[i]]){
            pos += (d_row[vars[i]*nrows])*times;
            times *= ncat[vars[i]];
        } else{
            return -1;
        }
    }
    return pos;
}

/* Obtains the number of possible configurations of vars */
int nconf(const int* vars, const int* ncat, int l_vars){
    int i;
    int ncf = 1;
    for ( i = 0; i < l_vars; i++ ){
        ncf *= ncat[vars[i]];
    }
    return ncf;

}

/* Returns the frequencies of each possible configuration of vars */
int* freq(const int* data, int nrows, int ncols, const int* vars, const int* ncat, int l_vars){
    int *fq;
    int i,ncf,pos;

    ncf = nconf(vars,ncat,l_vars);
    fq = (int*) malloc(ncf * sizeof(int));
    /*Initialize fq*/
    for ( i = 0; i < ncf; i++ ){
        fq[i] = 0;
    }

    for ( i = 0; i < nrows; i++ ){
        pos = position(&(data[i]), vars, ncat, l_vars,nrows);
        if (pos == -1){
            return NULL;
        } else{
            fq[pos]++;
        }
    }

    return fq;
}


Factor::Factor(){};
Factor::Factor(std::vector<int> variables, std::vector<int> num_categories) : variables(variables), num_categories(num_categories){
            int* vars = &(this->variables[0]);
            int l_vars = this->variables.size();
            int* ncat = &(this->num_categories[0]);
            int l_ncat = this->num_categories.size();
            int num_conf;
            for (auto it=variables.begin();it!=variables.end();++it){
                    if (*it >= l_ncat){
                        throw std::runtime_error("Variable index out of bounds");
                    }
            }
            num_conf = nconf(vars,ncat,l_vars);
            this->prob.resize(num_conf);
            this->mpe.resize(num_conf);

            std::vector<int> mpe_init (l_ncat,-1); 
            for(auto it=mpe.begin();it!=mpe.end();++it){*it = mpe_init;}
            for(auto it=prob.begin();it!=prob.end();++it){*it = 0;}
        };
Factor::~Factor(){};
  
  	/*Learns CPT of the first variable in variables vector conditioned to its parents, that are the rest of variables in variables vector*/
void Factor::learn_params(const int* data, int nrows, int ncols, double alpha){
  int *fq;
  int i,j,ncf,nc_xi,loops;
  double Nij = 0;
  int* vars = &this->variables[0];
  int l_vars = this->variables.size();
  int* ncat = &this->num_categories[0];
  
  ncf = nconf(vars,ncat,l_vars);
  //Get the dimension of the probability vector
  this->prob.resize(ncf);
  
    
  /* Obtain frequencies */
  fq = freq(data, nrows, ncols, vars, ncat, l_vars);
  if (fq == NULL){
      throw std::runtime_error("Error obtaining frequency");
  }
  
  nc_xi = ncat[vars[0]];
  loops = ncf/nc_xi;

  for ( i = 0; i < loops; i++ ){
      /* Obtain Nij */
      Nij = 0;
      for ( j = 0; j < nc_xi; j++ ){
          Nij += fq[i*nc_xi + j];
      }
      if (Nij != 0) {
          for ( j = 0; j < nc_xi; j++ ){
              this->prob[i*nc_xi + j] = (1.0* (fq[i*nc_xi + j]+alpha))/(Nij+nc_xi*(alpha * 1.0));
          }
      }else{
          for ( j = 0; j < nc_xi; j++ ){
          	this->prob[i*nc_xi + j] = 1.0/nc_xi;
          }
      }
  }
  free(fq);
}


/*Learns (from sufficient statistics se) CPT of the first variable in variables vector conditioned to its parents, that are the rest of variables in variables vector*/
/* The log-likelihood is returned as the output*/
double Factor::learn_params_se(std::vector<double> se, double alpha){
  int i,j,ncf,nc_xi,loops;
  double Nij = 0;
  int* vars = &this->variables[0];
  int l_vars = this->variables.size();
  int* ncat = &this->num_categories[0];
  double ll=0.0, Nijk;
  
  ncf = nconf(vars,ncat,l_vars);
  //Get the dimension of the probability vector
  this->prob.resize(ncf);
  
  nc_xi = ncat[vars[0]];
  loops = ncf/nc_xi;
  for ( i = 0; i < loops; i++ ){
      /* Obtain Nij */
      Nij = 0;
      for ( j = 0; j < nc_xi; j++ ){
          Nij += se[i*nc_xi + j];
      }
      if (Nij != 0) {
          for ( j = 0; j < nc_xi; j++ ){
            Nijk = se[i*nc_xi + j];  
            this->prob[i*nc_xi + j] = (1.0* (Nijk+alpha))/(Nij+nc_xi*(alpha * 1.0));
            if (Nijk != 0){
                    
                ll += Nijk * log(Nijk/Nij);
            }
          }
      }else{
          for ( j = 0; j < nc_xi; j++ ){
          	this->prob[i*nc_xi + j] = 1.0/nc_xi;
          }
      }
  }
  return ll;
}


/* The log-likelihood is returned as the output*/
double Factor::log_lik_se(std::vector<double> se, double alpha){
  int i,j,ncf,nc_xi,loops;
  double Nij = 0;
  int* vars = &this->variables[0];
  int l_vars = this->variables.size();
  int* ncat = &this->num_categories[0];
  double ll=0.0, Nijk;
  
  ncf = nconf(vars,ncat,l_vars);
  nc_xi = ncat[vars[0]];
  loops = ncf/nc_xi;
  
  for ( i = 0; i < loops; i++ ){
      /* Obtain Nij */
      Nij = 0;
      for ( j = 0; j < nc_xi; j++ ){
          Nij += se[i*nc_xi + j];
      }
      if (Nij != 0) {
          for ( j = 0; j < nc_xi; j++ ){
            Nijk = se[i*nc_xi + j];  
            if (Nijk != 0){
                ll += Nijk * log(Nijk/Nij);
            }
          }
      }
  }
  return ll;
}


/* The log-likelihood is returned as the output*/
double log_lik_se(std::vector<double> se, double alpha, std::vector<int> cluster, std::vector<int> num_categories){
  int i,j,ncf,nc_xi,loops;
  double Nij = 0;
  int* vars = &cluster[0];
  int l_vars = cluster.size();
  int* ncat = &num_categories[0];
  double ll=0.0, Nijk;
  
  ncf = nconf(vars,ncat,l_vars);
  nc_xi = ncat[vars[0]];
  loops = ncf/nc_xi;
  
  for ( i = 0; i < loops; i++ ){
      /* Obtain Nij */
      Nij = 0;
      for ( j = 0; j < nc_xi; j++ ){
          Nij += se[i*nc_xi + j];
      }
      if (Nij != 0) {
          for ( j = 0; j < nc_xi; j++ ){
            Nijk = se[i*nc_xi + j];  
            if (Nijk != 0){
                ll += Nijk * log(Nijk/Nij);
            }
          }
      }
  }
  return ll;
}


Factor_Node::Factor_Node(std::vector<int> variables, std::vector<int> num_categories, int id) : id(id){
	    this->factor = std::make_shared<Factor>(variables,num_categories);
        this->factor_bup = this->factor;
}

Factor_Node::Factor_Node(std::shared_ptr<Factor> factor, int id) : factor(factor), id(id){
        this->factor_bup = this->factor;
}

Factor_Node::Factor_Node(int id) : id(id){
}

Factor_Node::Factor_Node(){
}

Factor_Node::~Factor_Node(){}

/* Constructor */
Factor_Tree::Factor_Tree(std::vector<int> parents_inner, std::vector<int> parents_leave, std::vector<std::vector<int>> vec_variables_leave, std::vector<int> num_categories) {
  if (parents_inner.size()<=0){
      throw std::runtime_error("Factor_Tree constructor: all the input vectors must have size grater than 0");
  }
  if (! (parents_inner.size() == parents_leave.size() && parents_inner.size() == vec_variables_leave.size() && parents_inner.size() == num_categories.size())){
      throw std::runtime_error("Factor_Tree constructor: all the input vectors must have the same size");
  }
  this->root = std::make_shared<Factor_Node>(-1);    
  // Create sets of nodes
  
  for(unsigned int i = 0; i < vec_variables_leave.size(); ++i){
     if (vec_variables_leave[i].size()<=0){
       throw std::runtime_error("Factor_Tree constructor: The domain of the factors of all leave nodes must have size greater then 0");
     }
     this->inner_nodes.push_back(std::make_shared<Factor_Node>(i));  
     this->leave_nodes.push_back(std::make_shared<Factor_Node>(vec_variables_leave[i],num_categories,i+num_categories.size()));  
  }
  
  // Assign parent and children
  for(unsigned int i = 0; i < parents_inner.size(); ++i){
    int pi = parents_inner[i];
    if (pi == -1){
        this->inner_nodes[i]->parent = this->root;
        this->root->children.push_back(this->inner_nodes[i]);
    } else{                                                                                                
        this->inner_nodes[i]->parent = this->inner_nodes[pi];                                                                                               
        this->inner_nodes[pi]->children.push_back(this->inner_nodes[i]); 
    }
  }

  for(unsigned int i = 0; i < parents_leave.size(); ++i){
    int pi = parents_leave[i];                                                                                                  
    this->leave_nodes[i]->parent = this->inner_nodes[pi];                                                                                               
    this->inner_nodes[pi]->children.push_back(this->leave_nodes[i]);                                                                                               
  }
                                                                                                      
}
  
Factor_Tree::Factor_Tree(){}

Factor_Tree::~Factor_Tree(){}


void Factor_Tree::learn_parameters(const int* data, int nrows, int ncols, double alpha){
    for (auto it=this->leave_nodes.begin();it!=this->leave_nodes.end();++it){
            (*it)->factor->learn_params(data, nrows, ncols, alpha);
    }
}


/* Returns the position in the prob vector of values*/
int position_in_factor(std::vector<int> values, std::vector<int> vars, std::vector<int> ncat){
    int pos,times,i;
    pos = 0;
    times = 1;
    if(vars.size()==0)
    {
        return 0;
    }

    for ( i = 0; i < vars.size(); i++){
        if (values[i]<= ncat[vars[i]]){
            pos += (values[i])*times;
            times *= ncat[vars[i]];

        } else{
            return -1;
        }
    }
    return pos;
}

/* Returns the vector of values of the variables that corresponds to pos in the 1D vector. Its similar to convert an integer to its binary expression but must be consider that the variables can be not binary*/
std::vector<int> value_from_position(int pos, std::vector<int> vars, std::vector<int> ncat, int nconf){
    int i, nconf_current, pos_current;
    std::vector<int> values(vars.size());

    pos_current = pos;
    nconf_current = nconf;
  
    for ( i = (vars.size()-1); i >= 0; i-- ){
			values[i] = pos_current / (nconf_current/ncat[vars[i]]);
			pos_current -= values[i] * (nconf_current/ncat[vars[i]]);
			nconf_current = nconf_current/ncat[vars[i]];
    }
    return values;
}

/* Obtains the number of possible configurations of vars */
int nconf_in_factor(std::vector<int> vars, std::vector<int> ncat){
    int i;
    int ncf = 1;
    for ( i = 0; i < vars.size(); i++ ){
        ncf *= ncat[vars[i]];
    }
    return ncf;
}



/*Marginalizes factor f1 over variable xi*/
std::shared_ptr<Factor> sum_factor(const Factor &f1, int xi) {
  int nconf, i, pos_f_sum;
  std::vector<int> values, new_vars;
  std::shared_ptr<Factor> f_sum;

  // Create new factor without xi, and initialize to 0
  new_vars = f1.variables;
  
  auto var_iterator=std::find(new_vars.begin(), new_vars.end(), xi);
  int index = std::distance(new_vars.begin(), var_iterator);

  new_vars.erase(var_iterator);
  f_sum = std::make_shared<Factor>(new_vars,f1.num_categories);  
  
  // Marginalize 
  nconf=nconf_in_factor(f1.variables, f1.num_categories);
  for ( i = 0; i < nconf; i++ ){
  	values = value_from_position(i, f1.variables, f1.num_categories, nconf);
    
    values.erase(values.begin() + index);
    pos_f_sum = position_in_factor(values, f_sum->variables, f_sum->num_categories);
    f_sum->prob[pos_f_sum] += f1.prob[i];
    f_sum->mpe[pos_f_sum] = f1.mpe[i];
  }

  return f_sum;
}

//Search for each factor old find its position in var_factor_prod
std::vector<int> get_value_order(std::vector<int> var_factor_old, std::vector<int> var_factor_prod)
{
    std::vector<int> idx_vector;
    int index=0;
    for(auto it=var_factor_old.begin();it!=var_factor_old.end();++it)
    {
        auto var_iterator = std::find(var_factor_prod.begin(), var_factor_prod.end(), *it);
        index = std::distance(var_factor_prod.begin(), var_iterator);
        idx_vector.push_back(index);
    }

    return idx_vector;
}

/*Product of several factors*/
std::shared_ptr<Factor> prod_factor(const std::vector<std::shared_ptr<Factor>> factors) {
    int nconf, pos_f_prod, j;
    std::vector<int> values, new_vars, values_j;
    std::vector<std::vector<int>> idx_factors;
    
    std::shared_ptr<Factor> f_prod;
    
    for(auto factor = factors.begin(); factor != factors.end(); ++factor)
    {
        new_vars.insert( new_vars.end(), (*factor)->variables.begin(), (*factor)->variables.end() );
    }

    sort( new_vars.begin(), new_vars.end() );
    new_vars.erase( unique( new_vars.begin(), new_vars.end() ), new_vars.end() );
    
    f_prod = std::make_shared<Factor>(new_vars,(*factors.begin())->num_categories); 

    for(auto factor = factors.begin(); factor != factors.end(); ++factor)
    {
        idx_factors.push_back(get_value_order((*factor)->variables, f_prod->variables));
    }

    nconf=nconf_in_factor(f_prod->variables, f_prod->num_categories);
    for(int i = 0; i < nconf; i++)
    {
        values = value_from_position(i, f_prod->variables, f_prod->num_categories, nconf);
        f_prod->prob[i] = 1;
        //Rellenar con las probabilidades
        j=0;
  	   for(auto factor = factors.begin(); factor != factors.end(); ++factor)
        {
            values_j.clear();
            for(auto idx = idx_factors.at(j).begin(); idx != idx_factors.at(j).end(); ++idx){
                values_j.push_back(values[*idx]);
            }
            pos_f_prod = position_in_factor(values_j, (*factor)->variables, (*factor)->num_categories);
            f_prod->prob[i] *= (*factor)->prob[pos_f_prod];
            std::transform (f_prod->mpe.at(i).begin(), f_prod->mpe.at(i).end(), (*factor)->mpe.at(pos_f_prod).begin(), f_prod->mpe.at(i).begin(), [] (int a, int b) {return std::max(a,b);} );
            j++;        
        }
    }


  return f_prod;
}



/*Maximizes factor f1 over variable xi*/
std::shared_ptr<Factor> max_factor(const Factor &f1, int xi) {
  int nconf, i, pos_f_max, value_index;
  std::vector<int> values, new_vars;
  std::shared_ptr<Factor> f_max;

  // Create new factor without xi, and initialize to 0
  new_vars = f1.variables;
  
  auto var_iterator=std::find(new_vars.begin(), new_vars.end(), xi);
  int index = std::distance(new_vars.begin(), var_iterator);
  new_vars.erase(var_iterator);
  f_max = std::make_shared<Factor>(new_vars,f1.num_categories);  
  
  // Marginalize 
  nconf=nconf_in_factor(f1.variables, f1.num_categories);
  for ( i = 0; i < nconf; i++ ){
  	values = value_from_position(i, f1.variables, f1.num_categories, nconf);
    value_index = values[index];
    values.erase(values.begin() + index);
    pos_f_max = position_in_factor(values, f_max->variables, f_max->num_categories);
    if (f1.prob[i] > f_max->prob[pos_f_max]){
      f_max->prob[pos_f_max] = f1.prob[i];
      f_max->mpe[pos_f_max] = f1.mpe[i];
      f_max->mpe[pos_f_max][xi] = value_index;
    }
  }

  return f_max;
}


/*Eliminates variable xi (MPE)*/
std::shared_ptr<Factor> max_elim(const std::vector<std::shared_ptr<Factor>> factors, int xi){
  std::shared_ptr<Factor> f_prod, f_max;
  
  f_prod = prod_factor(factors);
  if(std::find(f_prod->variables.begin(), f_prod->variables.end(), xi) != f_prod->variables.end()){
      f_max = max_factor(*f_prod, xi);
  } else{
      f_max = f_prod;
  }
          
  return f_max;
}

/*Eliminates variable xi (evidence propagation)*/
std::shared_ptr<Factor> sum_elim(const std::vector<std::shared_ptr<Factor>> factors, int xi){
  std::shared_ptr<Factor> f_prod, f_sum;
  
  f_prod = prod_factor(factors);
  if(std::find(f_prod->variables.begin(), f_prod->variables.end(), xi) != f_prod->variables.end()){
      f_sum = sum_factor(*f_prod, xi);
  } else{
      f_sum = f_prod;
  }

  return f_sum;
}


/*Computes the factor of xi from the factors of its children (MPE)*/
void Factor_Tree::max_compute(int xi){
  std::shared_ptr<Factor_Node> node;
  std::vector<std::shared_ptr<Factor>> children_factors;
  std::shared_ptr<Factor> result_factor;
  if (xi == -1){ // Root node
    node = this->root;
  } else if (xi >=0 && xi < this->inner_nodes.size()){
    node = this->inner_nodes.at(xi);
  } else{
    throw std::runtime_error("Variable index out of bounds");
  }

  auto children = node->children;
  for(auto factor_node = children.begin(); factor_node!=children.end(); ++factor_node)
  {
	children_factors.push_back((*factor_node).lock()->factor);
  }

  if (xi == -1){
    result_factor = prod_factor(children_factors);
    this->root->factor = result_factor;
  } else{
    result_factor = max_elim(children_factors, xi);
    this->inner_nodes.at(xi)->factor = result_factor;
  }
}
 
    
    
    
    
/*Distributes evidence (Junction Tree)*/
void Factor_Tree::distribute_evidence(){  
    auto node = this->root;
    auto children = node->children;
    for(auto j = 0; j!=children.size(); ++j){
        auto xj = children.at(j).lock()->id;
        this->distribute_evidence(xj,nullptr);
    }
}
    
void Factor_Tree::distribute_evidence(int xi, std::shared_ptr<Factor> message){
  std::shared_ptr<Factor_Node> node;
  std::vector<std::shared_ptr<Factor>> children_factors;
  std::shared_ptr<Factor> result_factor;
  bool flag = false;
  if (xi >=0 && xi < this->inner_nodes.size()){
    node = this->inner_nodes.at(xi);
  } else{
    throw std::runtime_error("Variable index out of bounds");
  }
  auto children = node->children;
  for(auto j = 0; j!=children.size(); ++j){
      auto xj = children.at(j).lock()->id;
      if (xj < this->inner_nodes.size()){
          children_factors.clear();
          for(auto k = 0; k!=children.size(); ++k){
              if (k!=j){
                  children_factors.push_back(children.at(k).lock()->factor);
            }
          }
          if (message != nullptr){
              children_factors.push_back(message);
          }  
          if (children_factors.size() == 0){
              this->distribute_evidence(xj,nullptr);
          } else{
              result_factor = prod_factor(children_factors);
              auto vars = result_factor->variables;
              auto vars_xj = children.at(j).lock()->factor->variables;
              for (auto xm=vars.begin(); xm!=vars.end();++xm){
                  if(std::find(vars_xj.begin(),vars_xj.end(), *xm) != vars_xj.end()){
                      result_factor = result_factor;
                  } else{
                      result_factor = sum_factor(*result_factor, *xm);
                  }
              }
              this->distribute_evidence(xj,result_factor);
          }
      } else{ flag = true;}
  }

  if (flag){
       children_factors.clear();
       for(auto k = 0; k!=children.size(); ++k){
           children_factors.push_back(children.at(k).lock()->factor);
       }
       if (message != nullptr){
           children_factors.push_back(message); 
       }
       node->joint_factor =  prod_factor(children_factors);
  }
}
       



/*Computes the factor of xi from the factors of its children (evidence propagation)*/
void Factor_Tree::sum_compute(int xi){
  std::shared_ptr<Factor_Node> node;
  std::vector<std::shared_ptr<Factor>> children_factors;
  std::shared_ptr<Factor> result_factor;
  if (xi == -1){ // Root node
    node = this->root;
  } else if (xi >=0 && xi < this->inner_nodes.size()){
    node = this->inner_nodes.at(xi);
  } else{
    throw std::runtime_error("Variable index out of bounds");
  }
  auto children = node->children;
  for(auto factor_node = children.begin(); factor_node!=children.end(); ++factor_node)
  {
	children_factors.push_back((*factor_node).lock()->factor);
  }
  if (xi == -1){
    result_factor = prod_factor(children_factors);
    this->root->factor = result_factor;
  } else{

    result_factor = sum_elim(children_factors, xi);
    this->inner_nodes.at(xi)->factor = result_factor;
  }
}




  
/*Computes the factor of xi from the factors of its children (evidence propagation)*/
void Factor_Tree::prod_compute(int xi){
  std::shared_ptr<Factor_Node> node;
  std::vector<std::shared_ptr<Factor>> children_factors;
  std::shared_ptr<Factor> result_factor;
  if (xi == -1){ // Root node
    node = this->root;
  } else if (xi >=0 && xi < this->inner_nodes.size()){
    node = this->inner_nodes.at(xi);
  } else{
    throw std::runtime_error("Variable index out of bounds");
  }
  auto children = node->children;
  for(auto factor_node = children.begin(); factor_node!=children.end(); ++factor_node)
  {
	children_factors.push_back((*factor_node).lock()->factor);
  }
  if (xi == -1){
    result_factor = prod_factor(children_factors);
    this->root->factor = result_factor;
  } else{
    result_factor = prod_factor(children_factors);
    this->inner_nodes.at(xi)->factor = result_factor;
  }
}  

void max_compute_tree_aux(Factor_Tree &ft, int xi){
  	std::shared_ptr<Factor_Node> node;
    int xj;
  	// Select node
  	if (xi == -1){
      node = ft.root;
      
      
    } else if (xi >=0 && xi < ft.inner_nodes.size()){
      node = ft.inner_nodes.at(xi);
    } else{
      throw std::runtime_error("Variable index out of bounds");
    }
  	
  	// Compute in children nodes
	for(auto child = node->children.begin(); child != node->children.end(); ++child)
    {
    	  xj = (*child).lock()->id;
      if (xj >=0 && xj < ft.inner_nodes.size()){
        max_compute_tree_aux(ft,xj);
      } else if (xj <0 || xj >= 2*ft.inner_nodes.size()){
        throw std::runtime_error("Variable index out of bounds");
      }
    }
    
  	// Compute node xi
  	ft.max_compute(xi);
}

/*Computes the factors all nodes (MPE)*/
void Factor_Tree::max_compute_tree(){
   max_compute_tree_aux(*this, -1);     
}

void sum_compute_tree_aux(Factor_Tree &ft, int xi){
  	std::shared_ptr<Factor_Node> node;
    int xj;
  	// Select node
  	if (xi == -1){
      node = ft.root;
    } else if (xi >=0 && xi < ft.inner_nodes.size()){
      node = ft.inner_nodes.at(xi);
    } else{
      throw std::runtime_error("Variable index out of bounds");
    }
  	
  	// Compute in children nodes
	for(auto child = node->children.begin(); child != node->children.end(); ++child)
    {
      xj = (*child).lock()->id;
      if (xj >=0 && xj < ft.inner_nodes.size()){
        sum_compute_tree_aux(ft,xj);
      } else if (xj <0 || xj >= 2*ft.inner_nodes.size()){
        throw std::runtime_error("Variable index out of bounds");
      }
    }
  
  	// Compute node xi
  	ft.sum_compute(xi);
}
      
void sum_compute_query_aux(Factor_Tree &ft, int xi, const std::vector<bool> variables){
  	std::shared_ptr<Factor_Node> node;
    int xj;
  	// Select node
  	if (xi == -1){
      node = ft.root;
    } else if (xi >=0 && xi < ft.inner_nodes.size()){
      node = ft.inner_nodes.at(xi);
    } else{
      throw std::runtime_error("Variable index out of bounds");
    }
  	
  	// Compute in children nodes
	for(auto child = node->children.begin(); child != node->children.end(); ++child)
    {
      xj = (*child).lock()->id;
      if (xj >=0 && xj < ft.inner_nodes.size()){
        sum_compute_query_aux(ft,xj,variables);
      } else if (xj <0 || xj >= 2*ft.inner_nodes.size()){
        throw std::runtime_error("Variable index out of bounds");
      }
    }
  	// Compute node xi
    if (xi == -1){
         ft.prod_compute(xi);
    }
    else if (variables[xi]){
         ft.prod_compute(xi);
    } else{
         ft.sum_compute(xi);
    }
}

/*Computes the factors all nodes (evidence propagation)*/
void Factor_Tree::sum_compute_tree(){
   sum_compute_tree_aux(*this, -1);     
}
   
/*Computes the joint PD of variables*/
void Factor_Tree::sum_compute_query( const std::vector<bool> variables){
   sum_compute_query_aux(*this, -1, variables);     
}

/*Retracts the evidences from the network*/
void Factor_Tree::retract_evidence(){
    for(auto node = this->leave_nodes.begin(); node != this->leave_nodes.end(); ++node){
        (*node)->factor = (*node)->factor_bup;
    }
}




/*Sets evidence in variables to values. Auxiliar function*/
std::shared_ptr<Factor> set_evidence_node(std::shared_ptr<Factor_Node> fn1, const std::vector<int> variables, const std::vector<int> values){
  // Create copy of factor
  std::shared_ptr<Factor> f_new;
  std::shared_ptr<Factor> f1 = fn1->factor;
  std::vector<int> v_index;
  std::vector<int> values_comp(f1->num_categories.size(),-1);
  std::vector<int> new_vars;
  std::vector<int> valuesi;
  std::vector<int> mpe(f1->num_categories.size(),-1);
  int nconf, i, j, pos_f1;
  
  //For each variable in the node check if it is in the set of evidences
  for(int i = 0;i < f1->variables.size(); ++i)
  {
    auto elem=std::find(variables.begin(),variables.end(),f1->variables.at(i));
    if(elem!=variables.end())//If the variable is in the set of evidence then fix the value
    {
      int j=std::distance(variables.begin(),elem);
      values_comp[i] = values[j];
      mpe[variables[j]] = values[j];
    }else{//If the variable was not in the set of evidence, include in the new set of variables for the new factor and save the index in the old variables vector for future computations
      new_vars.push_back(f1->variables[i]);
      v_index.push_back(i);
    }
  }
  
  
  //Create new factor
  f_new = std::make_shared<Factor>(new_vars,f1->num_categories);
  nconf = f_new->prob.size();
  
  //For each position entry in the prob vector of the new factor
  for ( i = 0; i < nconf; i++ ){
  	valuesi = value_from_position(i, f_new->variables, f_new->num_categories, nconf);//Get the values of the position
    for ( j = 0; j < valuesi.size(); j++ ){//Fill the vector of values with the new values
    	  values_comp[v_index.at(j)] = valuesi[j];
    }
    pos_f1 = position_in_factor(values_comp, f1->variables, f1->num_categories);
    f_new->prob[i] = f1->prob[pos_f1];
    f_new->mpe[i] = mpe;
  }
  
  return f_new;
}
        
/*Sets evidence in variables to values*/
void Factor_Tree::set_evidence(const std::vector<int> variables, const std::vector<int> values){
  for(auto node=this->leave_nodes.begin();node!=this->leave_nodes.end();++node)
  {
  	(*node)->factor = set_evidence_node(*node, variables, values);
  }
}

/*Gets the position of cordinates i,j in the data vector with number of rows nrows*/
int pos_datav(int i, int j, int nrows){
    return(nrows*j + i);        
}



/*Predict marginals for data*/
std::vector<std::vector<float>> Factor_Tree::pred_data(int* data, int ncols, int nrows, std::vector<int> qvalues, std::vector<int> query, std::vector<int> evidence){
    Factor_Tree ft = *this;
    // Complete each row with missing values
    std::vector<std::vector<float>> probs(query.size(),std::vector<float>(nrows, 0));
    //#pragma omp parallel for firstprivate(ft)
    for (auto i=0;i<nrows;++i){
        std::vector<int> evidence1;
        std::vector<int> values;
        std::vector<int> values2;
        std::vector<int> evidence2;
        // Obtain the positions of the observed values  in the row 
        for (auto j=0;j<evidence.size();++j){
            if(data[pos_datav(i, evidence.at(j), nrows)]>=0){
                values.push_back(data[pos_datav(i, evidence.at(j), nrows)]); 
                evidence1.push_back(evidence.at(j));
            }
        }
        std::cout << "evidence1: ";
        print_vector(evidence1);
        std::cout << "values: ";
        print_vector(values);
        ft.set_evidence(evidence1, values);
        ft.sum_compute_tree();
        float prob_e = ft.root->factor->prob.at(0);
        ft.retract_evidence();
        if (prob_e == 0){
            throw std::runtime_error( "prob of evidence = 0 \n");
        }

        
        for (auto j=0;j<query.size();++j){
            evidence2 = evidence1;
            values2 = values;
            evidence2.push_back(query.at(j));
            values2.push_back(qvalues.at(j));
            ft.set_evidence(evidence2, values2);
            ft.sum_compute_tree();
            probs[j][i] = ft.root->factor->prob.at(0) / prob_e;
            std::cout << "i,j: " << i << "," << j << ", prob: " << probs[j][i] << ", prob_e: " << prob_e << ", prob_t: " << ft.root->factor->prob.at(0) << "\n";
            ft.retract_evidence();
            values2.clear();
            evidence2.clear();
        }
        
        //Remove evidence from f for next iteration
        //Initialize vectors
    }
        
    return probs;
}



/*Complete data with the values returned by MPE*/
void Factor_Tree::mpe_data(int* data_complete, const int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<double> &prob_mpe){
    Factor_Tree ft = *this;
    // Complete each row with missing values
    std::vector<int> values;
    std::vector<int> evidence;
    std::vector<int> missing;
    std::vector<int> mpe;
    std::vector<double> mpe_prob_v(nrows, 0.0);
    int HadCatch = 0;
    #pragma omp parallel for firstprivate(ft) private(values, evidence, missing, mpe) shared(HadCatch, mpe_prob_v)
    for (auto it=0;it<rows.size();++it){
        try{
            int i = rows.at(it);
            // Obtain the positions of the observed values  in the row 
            for (auto j=0;j<ncols;++j){
                if(data_missing[pos_datav(i, j, nrows)]==-1){
                    missing.push_back(j);
                } else{
                    evidence.push_back(j);
                    values.push_back(data_missing[pos_datav(i, j, nrows)]); 
                }
            }
            // Compute the MPE of the missing values given the observed values
            ft.set_evidence(evidence, values);
            ft.max_compute_tree();
            mpe = ft.root->factor->mpe.at(0);
            mpe_prob_v[i] = ft.root->factor->prob.at(0);
            // Complete data_complete with the MPE
            for (auto itm=missing.begin();itm!=missing.end();++itm){
                if (mpe.at(*itm) == -1){
                    throw std::runtime_error( "MPE with 0 probability \n");
                }
                data_complete[pos_datav(i, *itm, nrows)] = mpe.at(*itm);
            }
            //Remove evidence from f for next iteration
            ft.retract_evidence();
            //Initialize vectors
            values.clear();
            evidence.clear();
            missing.clear();
            mpe.clear();
        } catch(std::bad_alloc){
             HadCatch++;   
        }
    }
    if (HadCatch) {throw std::bad_alloc();}
    prob_mpe = mpe_prob_v;
    
}

/* Sum expectations*/
void sum_factor_se(std::shared_ptr<Factor> fse, const std::shared_ptr<Factor> fin, const std::vector<int> evidence_cl, const std::vector<int> values_cl,const int weight){
    int nconf, pos_f_prod, j;
    std::vector<int> values;
    std::vector<int> idx_factor, idx_evidence, values_fse;
    bool flag;
    double sum_fin=0.0;

    // Compute probability of cluster variables conditional on evidence
    for(auto prob = fin->prob.begin(); prob != fin->prob.end(); ++prob){
        sum_fin+= *prob;
    }
    if (sum_fin==0){
        for(auto prob = fse->prob.begin(); prob != fse->prob.end(); ++prob){
            *prob = 1.0/fse->prob.size();
        }
        return;
    }
           
    idx_factor = get_value_order(fin->variables, fse->variables); 
    idx_evidence = get_value_order(evidence_cl, fse->variables);    
    nconf=nconf_in_factor(fin->variables, fin->num_categories);
    
    // Init vector with size fin->variables
    values_fse.resize(fse->variables.size(),-1);
    j=0;
    for(auto idx = idx_evidence.begin(); idx != idx_evidence.end(); ++idx){
            values_fse[*idx] = values_cl[j];
            j++;
    }
    
    for(int i = 0; i < nconf; i++)
    {
        values = value_from_position(i, fin->variables, fin->num_categories, nconf);
        
        j=0;
        for(auto idx = idx_factor.begin(); idx != idx_factor.end(); ++idx){
            values_fse[*idx] = values[j];
            j++;
        }
        pos_f_prod = position_in_factor(values_fse, fse->variables, fse->num_categories);
        
        if (sum_fin==0){
            fse->prob[pos_f_prod] += (1.0/nconf) * weight; 
        } else{
            fse->prob[pos_f_prod] += (fin->prob[i]/sum_fin) * weight; 
        }

    }

}




/*Get the sufficent statistics for cluster*/
std::vector<double> Factor_Tree::se_data(int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<int> weights, std::vector<int> cluster_int){
    Factor_Tree ft = *this;
    // Complete each row with missing values
    std::vector<int> values;
    std::vector<int> values_cl;
    std::vector<int> evidence;
    std::vector<int> evidence_cl;
    std::vector<bool> cluster(ncols,false);
    std::shared_ptr<Factor> fse;
    int HadCatch = 0;

    // Create new factor without xi, and initialize to 0
    fse = std::make_shared<Factor>(cluster_int, this->leave_nodes.at(0)->factor->num_categories); 
    for (auto cl=cluster_int.begin();cl<cluster_int.end();++cl){
        cluster[*cl] = true;
    }
    
    //#pragma omp parallel for firstprivate(ft) private(values, evidence,evidence_cl,values_cl) shared(HadCatch, fse)
    for (auto it=0;it<rows.size();++it){
        try{
            int i = rows.at(it);
            // Obtain the positions of the observed values  in the row 
            for (auto j=0;j<ncols;++j){
                if(data_missing[pos_datav(i, j, nrows)]!=-1){
                    if (cluster[j]){
                        evidence_cl.push_back(j);
                        values_cl.push_back(data_missing[pos_datav(i, j, nrows)]); 
                    } 
                    evidence.push_back(j);
                    values.push_back(data_missing[pos_datav(i, j, nrows)]); 
                }
                
            }
            std::shared_ptr<Factor> fin;
    
            // Check if 
            if (evidence_cl.size() == cluster_int.size()){
                fin = std::make_shared<Factor>();
                fin->prob.push_back(1);
            } 
            else{
                // Compute the MPE of the missing values given the observed values
                ft.set_evidence(evidence, values);
                ft.sum_compute_query(cluster);
                fin = ft.root->factor;
            }
            
            //Sum SE
            //#pragma omp critical
            sum_factor_se(fse, fin, evidence_cl, values_cl, weights.at(i));
            
            //Remove evidence from f for next iteration
            ft.retract_evidence();
            //Initialize vectors
            values.clear();
            evidence.clear();
            values_cl.clear();
            evidence_cl.clear();
        } catch(std::bad_alloc){
             HadCatch++;   
        }
    }
    if (HadCatch) {throw std::bad_alloc();}
    return fse->prob;
}

/*Distribute evidence to leaf nodes*/
void Factor_Tree::distribute_leaves(std::vector<int> evidence, std::vector<int> values, int ncols){
    std::vector<int> values_cl;
    std::vector<int> evidence_cl;
    for(auto node=this->leave_nodes.begin();node!=this->leave_nodes.end();++node){
        std::vector<bool> cluster(ncols,false);
        auto cluster_int = (*node)->factor_bup->variables; 
        auto fse = std::make_shared<Factor>(cluster_int, this->leave_nodes.at(0)->factor->num_categories); 
        auto parent_node = (*node)->parent;
        auto vars = parent_node.lock()->joint_factor->variables;
        auto result_factor = std::make_shared<Factor>(*(parent_node.lock()->joint_factor));
        for (auto xm=vars.begin(); xm!=vars.end();++xm){
          if(std::find(cluster_int.begin(),cluster_int.end(), *xm) != cluster_int.end()){
              result_factor = result_factor;
          } else{
              result_factor = sum_factor(*result_factor, *xm);
          }
        }

        for (auto cl=cluster_int.begin();cl<cluster_int.end();++cl){
            cluster[*cl] = true;
        }
        
        for (auto j=0;j<evidence.size();++j){
            if (cluster[evidence.at(j)]){
                evidence_cl.push_back(evidence.at(j));
                values_cl.push_back(values.at(j)); 
            } 
        }

        sum_factor_se(fse, result_factor, evidence_cl, values_cl, 1.0);
        (*node)->joint_factor = fse;
        values_cl.clear();
        evidence_cl.clear();
    }

}

/*Get the sufficent statistics for each node*/
std::vector<std::vector<double>> Factor_Tree::se_data_distribute(int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<int> weights){
    std::ofstream fout;
    Factor_Tree ft = *this;
    // Complete each row with missing values
    std::vector<int> values;
    std::vector<int> evidence;

    std::vector<std::vector<double>> v_se(ncols);
    int HadCatch = 0;
    // Create new factor without xi, and initialize to 0
    ft.retract_evidence();
    for(auto i= 0; i<v_se.size(); i++){
        v_se.at(i).resize(this->leave_nodes.at(i)->factor->prob.size(),0);
    }
    #pragma omp parallel for firstprivate(ft) private(values, evidence) shared(HadCatch)
    for (auto it=0;it<rows.size();++it){
        try{
            int i = rows.at(it);
            // Obtain the positions of the observed values  in the row 
            for (auto j=0;j<ncols;++j){
                if(data_missing[pos_datav(i, j, nrows)]!=-1){
                    evidence.push_back(j);
                    values.push_back(data_missing[pos_datav(i, j, nrows)]); 
                }
                
            }

            ft.set_evidence(evidence, values);
            ft.sum_compute_tree();
            ft.distribute_evidence();
            ft.distribute_leaves(evidence, values, ncols);

            
            //Sum SE
            for(auto k=0; k < v_se.size(); ++k){
                    for(auto l=0; l < v_se.at(k).size(); ++l){
                        #pragma omp critical
                        v_se[k][l] += ft.leave_nodes.at(k)->joint_factor->prob.at(l) * weights.at(it); 
                    }
            }
            //Remove evidence from f for next iteration
            ft.retract_evidence();
            //Initialize vectors
            values.clear();
            evidence.clear();
        } catch(std::bad_alloc){
             HadCatch++;   
        }
    }
    if (HadCatch) {throw std::bad_alloc();}
    return v_se;
}


/*Get the sufficent statistics for cluster (parallel)*/
std::vector<double> Factor_Tree::se_data_parallel(int* data_missing, int ncols, int nrows, std::vector<int> rows, std::vector<int> weights, std::vector<int> cluster_int){
    Factor_Tree ft = *this;
    // Complete each row with missing values
    std::vector<int> values;
    std::vector<int> values_cl;
    std::vector<int> evidence;
    std::vector<int> evidence_cl;
    std::vector<bool> cluster(ncols,false);
    std::shared_ptr<Factor> fse;
    int HadCatch = 0;

    // Create new factor without xi, and initialize to 0
    fse = std::make_shared<Factor>(cluster_int, this->leave_nodes.at(0)->factor->num_categories); 
    for (auto cl=cluster_int.begin();cl<cluster_int.end();++cl){
        cluster[*cl] = true;
    }
    
    #pragma omp parallel for firstprivate(ft) private(values, evidence,evidence_cl,values_cl) shared(HadCatch, fse)
    for (auto it=0;it<rows.size();++it){
        try{
            int i = rows.at(it);
            // Obtain the positions of the observed values  in the row 
            for (auto j=0;j<ncols;++j){
                if(data_missing[pos_datav(i, j, nrows)]!=-1){
                    if (cluster[j]){
                        evidence_cl.push_back(j);
                        values_cl.push_back(data_missing[pos_datav(i, j, nrows)]); 
                    } 
                    evidence.push_back(j);
                    values.push_back(data_missing[pos_datav(i, j, nrows)]); 
                }
                
            }
            std::shared_ptr<Factor> fin;
    
            if (evidence_cl.size() == cluster_int.size()){
                fin = std::make_shared<Factor>();
                fin->prob.push_back(1);
            } 
            else{
                // Compute the MPE of the missing values given the observed values
                ft.set_evidence(evidence, values);
                ft.sum_compute_query(cluster);
                fin = ft.root->factor;
            }
            
            //Sum SE
            #pragma omp critical
            sum_factor_se(fse, fin, evidence_cl, values_cl, weights.at(i));
            
            //Remove evidence from f for next iteration
            ft.retract_evidence();
            //Initialize vectors
            values.clear();
            evidence.clear();
            values_cl.clear();
            evidence_cl.clear();
        } catch(std::bad_alloc){
             HadCatch++;   
        }
    }
    if (HadCatch) {throw std::bad_alloc();}
    return fse->prob;
}


/*Complete data with the values returned by MPE*/
void Factor_Tree::aux_omp(){
    int a = 0;
    #pragma omp parallel for 
    for (auto i=0;i<100;++i){
        std::cout << a << "\n";
        a++;
    }
}
    
    
/*Copy constructor for Factor_Node*/
Factor_Node::Factor_Node(const Factor_Node& fnode){
  if (fnode.factor){
      this->factor = std::make_shared<Factor>(*fnode.factor);
  } else{
      this->factor = fnode.factor;
  }
  this->id = fnode.id;
  if (fnode.factor_bup){
      this->factor_bup = std::make_shared<Factor>(*fnode.factor_bup);       
  } else{
      this->factor_bup = fnode.factor;
  }
  if (fnode.joint_factor){
      this->joint_factor = std::make_shared<Factor>(*fnode.joint_factor);       
  } else{
      this->joint_factor = fnode.joint_factor;
  }
};
        

/*Copy constructor for Factor_Tree*/
Factor_Tree::Factor_Tree(const Factor_Tree& ft){
  unsigned int i;
  std::vector<int> parents_leave, parents_inner;
  Factor_Node f_aux =  *ft.root;
  
  
  this->root = std::make_shared<Factor_Node>(f_aux);
  
  for(auto it = ft.leave_nodes.begin(); it != ft.leave_nodes.end(); ++it)
  {
    this->leave_nodes.push_back(std::make_shared<Factor_Node>(**(it)));
  }
  
  for(auto it = ft.inner_nodes.begin(); it != ft.inner_nodes.end(); ++it)
  {
    this->inner_nodes.push_back(std::make_shared<Factor_Node>(**(it)));
  }
  
  // Get parents of inner and leave nodes
  for (i = 0; i < ft.inner_nodes.size(); i++){
  	parents_inner.push_back(ft.inner_nodes[i]->parent.lock()->id);
  }
  for (i = 0; i < ft.leave_nodes.size(); i++){
  	parents_leave.push_back(ft.leave_nodes[i]->parent.lock()->id);
  }
  // Assign parent and children
  for(i = 0; i < parents_inner.size(); ++i){
    int pi = parents_inner[i];
    if (pi == -1){
        this->inner_nodes[i]->parent = this->root;
        this->root->children.push_back(this->inner_nodes[i]);
    } else{                                                                                                
        this->inner_nodes[i]->parent = this->inner_nodes[pi];                                                                                               
        this->inner_nodes[pi]->children.push_back(this->inner_nodes[i]); 
    }
  }

  for(i = 0; i < parents_leave.size(); ++i){
    int pi = parents_leave[i];                                                                                                  
    this->leave_nodes[i]->parent = this->inner_nodes[pi];                                                                                               
    this->inner_nodes[pi]->children.push_back(this->leave_nodes[i]);                                                                                               
  }     
};
  

