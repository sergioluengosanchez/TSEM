/// Code your testbench here.
// Uncomment the next line for SystemC modules.
// #include <systemc>

#include <cstdlib>     /* malloc, free, rand */
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include "variable_elimination_data.h"
#include <new>
#include <fstream>
#include <math.h> 
#include <ctime>
#include <string.h>
#include <limits>

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

void  print_vector(std::vector<std::vector<double>> v){
    std::cout << "[";
    for (auto i= v.begin(); i != v.end(); ++i){
        print_vector(*i);
        std::cout << ",";
    }
    std::cout << "]\n";
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
        if (d_row[vars[i]*nrows] == -1){ // Modificado para available case analysis
            return -2;
        }
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


Factor_data::Factor_data(){
            this->data_size = 0;
            this->prob = NULL;
};
//Factor_data::Factor_data(std::vector<int> &variables, std::vector<int> &num_categories, unsigned ninst) : variables(variables), num_categories(num_categories){
//            int* vars = &(this->variables[0]);
//            int l_vars = this->variables.size();
//            int* ncat = &(this->num_categories[0]);
//            int l_ncat = this->num_categories.size();
//            int num_conf= nconf(vars,ncat,l_vars);
////            std::vector< std::vector<double>> prob_aux(num_conf,std::vector<double>(ninst)); 
//            for (auto it=variables.begin();it!=variables.end();++it){
//                    if (*it >= l_ncat){
//                        throw std::runtime_error("Variable index out of bounds");
//                    }
//            }
////            this->prob = prob_aux;
////            this->prob = new int[num_conf*ninst];
//        };

Factor_data::Factor_data(std::vector<int> &variables, std::vector<int> &num_categories, unsigned data_size) : variables(variables){
            int* vars = &(variables[0]);
            int l_vars = variables.size();
            int* ncat = &(num_categories[0]);
            int l_ncat = num_categories.size();
            int num_conf= nconf(vars,ncat,l_vars);
//            std::vector< std::vector<double>> prob_aux(num_conf,std::vector<double>(ninst)); 
//            for (auto it=variables.begin();it!=variables.end();++it){
//                    if (*it >= l_ncat){
//                        throw std::runtime_error("Variable index out of bounds");
//                    }
//            }
//            this->prob = prob_aux;
            this->num_conf = num_conf;
//            this->prob = (double *)malloc(num_conf*data_size * sizeof(double));
//            std::cout << "init alloc";
            this->prob = new double[num_conf*data_size];
            this->data_size=data_size;
//            std::cout << " end alloc ";
        };

Factor_data::Factor_data(const Factor_data &fd){
        this->variables = fd.variables; // Array with the ids of the variables ordered bottom up
        this->num_conf = fd.num_conf; // Array with the ids of the variables ordered bottom up
        this->data_size=fd.data_size;
        /*TODO: what happens if this->prob is initialized, free memory*/
//        this->prob = (double *)malloc(fd.num_conf*fd.data_size * sizeof(double));
//        std::cout << "init alloc cpy ";
        this->prob = new double[fd.num_conf*fd.data_size];
//        std::cout << " end alloc cpy ";
        for (unsigned i=0; i<fd.num_conf; ++i){
            for (unsigned j=0; j<fd.data_size; ++j){
                this->prob[i*fd.data_size+j] = fd.prob[i*fd.data_size+j];        
            }
        }
};

Factor_data::~Factor_data(){
//        free(this->prob); 
//        std::cout << "init destroy 0";
        if (this->prob != NULL){
//            std::cout << "init destroy";
            delete [] this->prob;
//            std::cout << " end ";
        }   
};




// Multiply arrays
void mul_data_arrays(double* array_in, const double* array1, const double* array2, unsigned len){
    for (unsigned j = 0; j < len; ++j){
          array_in[j] = array1[j] * array2[j];          
    }
}

void mul_data_arrays(int* array_in, const int* array1, const int* array2, int len){
    for (unsigned j = 0; j < len; ++j){
          array_in[j] = array1[j] * array2[j];          
    }
}

void mul_arrays(double* array_in, const double* array1, const double* array2, unsigned len){
    for (unsigned j = 0; j < len; ++j){
          array_in[j] = array1[j] * array2[j];          
    }
}

void mul_arrays(int* array_in, const int* array1, const int* array2, int len){
    for (unsigned j = 0; j < len; ++j){
          array_in[j] = array1[j] * array2[j];          
    }
}


// Multiply arrays
void sum_arrays(double* array_in, const double* array1, const double* array2, unsigned len){
    for (unsigned j = 0; j < len; ++j){
          array_in[j] = array1[j] + array2[j];          
    }
}


/* Returns the vector of values of the variables that corresponds to pos in the 1D vector. Its similar to convert an integer to its binary expression but must be consider that the variables can be not binary*/
std::vector<int> value_from_position(int pos, std::vector<int> &vars, std::vector<int> &ncat, int nconf){
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

/* Increase values_in by one */
void increase_value(std::vector<int> &values_in, std::vector<int> &vars, std::vector<int> &ncat){
    bool flag = true;
    int i = 0;
    int* vars_d = vars.data();
    int* values = values_in.data();
    int* ncat_d = ncat.data();
    while(flag==true && i<vars.size()){
        if (values[i] >= ncat_d[vars_d[i]] -1){
            values[i] = 0;
            ++i;
        } else{
            values[i] += 1;
            flag=false;
        }
    }
}


// Set parameters and multiply with indicator matrix
void Factor_data::set_params_evidence(std::vector<double> parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators, std::vector<int> &num_categories){
    std::vector<int> values(this->variables.size(),0);
    for (unsigned i=0; i< parameters.size(); ++i){
        //Get values of the parameters
//        std::vector<int> values = value_from_position(i, this->variables, num_categories, parameters.size());
//        std::cout << "value_from_position \n";
//        print_vector(values);
//        std::cout << "values_inc \n";
//        print_vector(values_inc);
        double pi = parameters[i];
        unsigned data_size = indicators[0]->at(0).size();   
        double *prob_data = &this->prob[i*data_size];
        //Compute indicators and multiply
        std::vector<int> indicators_result(data_size);
        if (this->variables.size() == 1){
            indicators_result = indicators[this->variables[0]]->at(values[0]);
        }
        else{
            mul_data_arrays(indicators_result.data(), indicators[this->variables[0]]->at(values[0]).data(), indicators[this->variables[1]]->at(values[1]).data(), data_size);
            for (unsigned j=2; j < this->variables.size(); ++j){
                 mul_data_arrays(indicators_result.data(), indicators_result.data(), indicators[this->variables[j]]->at(values[j]).data(), data_size);
            }
        }
        for (unsigned j=0; j < indicators_result.size(); ++j){
            prob_data[j] = pi * indicators_result[j];
        }
        increase_value(values, this->variables, num_categories);
        this->data_size = data_size;
    }   
}


// Generate indicators matrix from data
std::vector<std::shared_ptr<std::vector<std::vector<int>>>> generate_indicators(int* data, int nrows, int ncols, std::vector<int> num_categories){
    std::vector<std::shared_ptr<std::vector<std::vector<int>>>> indicators;
    //Initialize indicators    
    for (unsigned i=0; i<ncols; ++i){
        std::vector<std::vector<int>> ind_col;
        for (unsigned j=0; j<num_categories.at(i); ++j){
            std::vector<int> ind_cat(nrows, 0);
            ind_col.push_back(ind_cat);
        }
        indicators.push_back(std::make_shared<std::vector<std::vector<int>>>(ind_col));
    }
    
    //Assign values to the indicators
    for (unsigned row=0; row< nrows; ++row){
        for (unsigned col=0; col< ncols; ++col){
            int value = data[nrows*col + row];
            if (value==-1){
                for (unsigned cat=0; cat<num_categories[col]; ++cat){
                    indicators[col]->at(cat)[row] = 1;
                }  
            } else{
                indicators[col]->at(value)[row] = 1;
            }  
        }        
    }
    return(indicators);
   
}

// Generate indicators matrix from data, filling in variables with na
std::vector<std::shared_ptr<std::vector<std::vector<int>>>> generate_indicators(int* data, int nrows, int ncols, std::vector<int> num_categories, std::vector<int> variables){
    std::vector<std::shared_ptr<std::vector<std::vector<int>>>> indicators;
    //Initialize indicators    
    for (unsigned i=0; i<ncols; ++i){
        std::vector<std::vector<int>> ind_col;
        for (unsigned j=0; j<num_categories.at(i); ++j){
            std::vector<int> ind_cat(nrows, 0);
            ind_col.push_back(ind_cat);
        }
        indicators.push_back(std::make_shared<std::vector<std::vector<int>>>(ind_col));
    }
    
    //Assign values to the indicators
    for (unsigned col=0; col< ncols; ++col){
        if (std::find(variables.begin(), variables.end(), col)!=variables.end()){
            for (unsigned row=0; row< nrows; ++row){
                for (unsigned cat=0; cat<num_categories[col]; ++cat){
                    indicators[col]->at(cat)[row] = 1;
                }  
            }          
        }
        else{
            for (unsigned row=0; row< nrows; ++row){
                int value = data[nrows*col + row];
                if (value==-1){
                    for (unsigned cat=0; cat<num_categories[col]; ++cat){
                        indicators[col]->at(cat)[row] = 1;
                    }  
                } else{
                    indicators[col]->at(value)[row] = 1;
                }  
            }  
        }      
    }
    return(indicators);
   
}


/* Factor_Node_data constructor */

Factor_Node_data::Factor_Node_data(std::vector<int> &variables, std::vector<int> &num_categories, int id, unsigned ninst) : id(id){
	   this->factor = std::make_shared<Factor_data>(variables,num_categories, ninst);
}

Factor_Node_data::Factor_Node_data(std::shared_ptr<Factor_data> factor, int id) : factor(factor), id(id){
}

Factor_Node_data::Factor_Node_data(int id) : id(id){
}

Factor_Node_data::Factor_Node_data(){
}

Factor_Node_data::~Factor_Node_data(){}


///* Factor_Tree constructor */

Factor_Tree_data::Factor_Tree_data(){}

Factor_Tree_data::Factor_Tree_data(std::vector<int> parents_inner, std::vector<int> parents_leave, std::vector<std::vector<int>> vec_variables_leave, std::vector<int> num_categories) {
  if (parents_inner.size()<=0){
      throw std::runtime_error("Factor_Tree constructor: all the input vectors must have size grater than 0");
  }
  if (! (parents_inner.size() == parents_leave.size() && parents_inner.size() == vec_variables_leave.size() && parents_inner.size() == num_categories.size())){
      throw std::runtime_error("Factor_Tree constructor: all the input vectors must have the same size");
  }
  this->root = std::make_shared<Factor_Node_data>(-1);    
  // Create sets of nodes
  
  for(unsigned int i = 0; i < vec_variables_leave.size(); ++i){
     if (vec_variables_leave[i].size()<=0){
       throw std::runtime_error("Factor_Tree constructor: The domain of the factors of all leave nodes must have size greater then 0");
     }
     this->inner_nodes.push_back(std::make_shared<Factor_Node_data>(i));  
     this->leave_nodes.push_back(std::make_shared<Factor_Node_data>(i+vec_variables_leave.size()));
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
  //Initialize parameter vector
  for(unsigned int i = 0; i < parents_leave.size(); ++i){
    int* vars = &(vec_variables_leave[i][0]);
    int l_vars = vec_variables_leave[i].size();
    int* ncat = &(num_categories[0]);
    int num_conf= nconf(vars,ncat,l_vars);
    std::vector<double> parame(num_conf);
    this->parameters.push_back(parame);
  }
  this->num_categories = num_categories;
  this->variables = vec_variables_leave;
  this->data_size = 0;

                                                                                                      
}

Factor_Tree_data::~Factor_Tree_data(){}

/* Returns the position in the prob vector of values*/
int position_in_factor(std::vector<int> &values, std::vector<int> &vars, std::vector<int> &ncat){
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

/* Obtains the number of possible configurations of vars */
int nconf_in_factor(std::vector<int> &vars, std::vector<int> &ncat){
    int i;
    int ncf = 1;
    for ( i = 0; i < vars.size(); i++ ){
        ncf *= ncat[vars[i]];
    }
    return ncf;
}



/*Marginalizes factor f1 over variable xi*/
std::shared_ptr<Factor_data> sum_factor(Factor_data &f1, int xi, std::vector<int> &num_categories, int data_size) {
  int nconf, i, j, pos_f_sum;
  std::vector<int> values, new_vars;
  std::shared_ptr<Factor_data> f_sum;
//   std::ofstream myfile;
  // Create new factor without xi, and initialize to 0
  new_vars = f1.variables;
  
  auto var_iterator=std::find(new_vars.begin(), new_vars.end(), xi);
  int index = std::distance(new_vars.begin(), var_iterator);
  

  new_vars.erase(var_iterator);
  f_sum = std::make_shared<Factor_data>(new_vars,num_categories, data_size);  
  std::vector<int> visited_position(f_sum->num_conf,0);  
  // Marginalize 
  nconf=nconf_in_factor(f1.variables, num_categories);
  if (nconf == 1){}
  std::vector<int> values_inc(f1.variables.size(),0);
  for ( i = 0; i < nconf; i++ ){
    std::vector<int> values = values_inc;
//    values = value_from_position(i, f1.variables, num_categories, nconf);

    double *prob_data = &f_sum->prob[pos_f_sum*data_size];
    double *prob_f1 = &f1.prob[i*data_size];
    values.erase(values.begin() + index);
    pos_f_sum = position_in_factor(values, f_sum->variables, num_categories);
    prob_data = &f_sum->prob[pos_f_sum*data_size];
    if (visited_position[pos_f_sum]==0){
          visited_position[pos_f_sum] = 1;
          for (j =0; j<data_size;++j){prob_data[j] = prob_f1[j];}

    }
    else{
         sum_arrays(prob_data, prob_data, prob_f1, data_size);
    }

    increase_value(values_inc, f1.variables, num_categories);

  }
  return f_sum;
}


//Search for each factor old find its position in var_factor_prod
std::vector<int> get_value_order(std::vector<int> &var_factor_old, std::vector<int> &var_factor_prod, std::vector<int> &num_categories)
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
std::shared_ptr<Factor_data> prod_factor(std::vector<std::shared_ptr<Factor_data>> &factors, std::vector<int> &num_categories, int data_size) {
    int nconf, pos_f_prod, j,k;
    std::vector<int> new_vars, values_j;
    std::vector<std::vector<int>> idx_factors;
    
    std::shared_ptr<Factor_data> f_prod;

    for(auto factor = factors.begin(); factor != factors.end(); ++factor)
    {
        new_vars.insert( new_vars.end(), (*factor)->variables.begin(), (*factor)->variables.end() );
    }

    sort( new_vars.begin(), new_vars.end() );
    new_vars.erase( unique( new_vars.begin(), new_vars.end() ), new_vars.end() );
    
    f_prod = std::make_shared<Factor_data>(new_vars,num_categories, data_size); 

    for(auto factor = factors.begin(); factor != factors.end(); ++factor)
    {
        idx_factors.push_back(get_value_order((*factor)->variables, f_prod->variables,num_categories));
    }

    nconf=nconf_in_factor(f_prod->variables, num_categories);
    std::vector<int> values(f_prod->variables.size(),0);
    for(int i = 0; i < nconf; i++)
    {
        double *prob_data = &f_prod->prob[i*data_size];
    
        //Rellenar con las probabilidades
        j=0;
  	   for(auto factor = factors.begin(); factor != factors.end(); ++factor)
        {
            values_j.clear();
            for(auto idx = idx_factors.at(j).begin(); idx != idx_factors.at(j).end(); ++idx){
                values_j.push_back(values[*idx]);
            }
            pos_f_prod = position_in_factor(values_j, (*factor)->variables, num_categories);
            double *prob_f = &(*factor)->prob[pos_f_prod*data_size];
            if (j==0){
                for (k =0; k<data_size;++k){prob_data[k] = prob_f[k];}
            }
            else{
                mul_data_arrays(prob_data, prob_data, prob_f, data_size);
            }
            j++;        
        }
        increase_value(values, f_prod->variables, num_categories);
    }

  return f_prod;
}


/*Eliminates variable xi (evidence propagation)*/
std::shared_ptr<Factor_data> sum_elim(std::vector<std::shared_ptr<Factor_data>> &factors, int xi, std::vector<int> &num_categories, int data_size){
  std::shared_ptr<Factor_data> f_prod, f_sum;
  f_prod = prod_factor(factors, num_categories, data_size);
  if(std::find(f_prod->variables.begin(), f_prod->variables.end(), xi) != f_prod->variables.end()){
      f_sum = sum_factor(*f_prod, xi, num_categories, data_size);
  } else{
      f_sum = f_prod;
  }
  return f_sum;
}


/*Computes the factor of xi from the factors of its children (evidence propagation)*/
void Factor_Tree_data::sum_compute(int xi){
  std::shared_ptr<Factor_Node_data> node;
  std::vector<std::shared_ptr<Factor_data>> children_factors;
  std::shared_ptr<Factor_data> result_factor;
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
    result_factor = prod_factor(children_factors, this->num_categories, this->data_size);
    this->root->factor = result_factor;
  } else{
    result_factor = sum_elim(children_factors, xi, this->num_categories, this->data_size);
    this->inner_nodes.at(xi)->factor = result_factor;
  }
}


/*Computes the factor of xi from the factors of its children (evidence propagation)*/
void Factor_Tree_data::prod_compute(int xi){
  std::shared_ptr<Factor_Node_data> node;
  std::vector<std::shared_ptr<Factor_data>> children_factors;
  std::shared_ptr<Factor_data> result_factor;
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
    result_factor = prod_factor(children_factors, this->num_categories, this->data_size);
    this->root->factor = result_factor;
  } else{
    result_factor = prod_factor(children_factors, this->num_categories, this->data_size);
    this->inner_nodes.at(xi)->factor = result_factor;
  }
}  


void sum_compute_tree_aux(Factor_Tree_data &ft, int xi, std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators){
  	std::shared_ptr<Factor_Node_data> node;
    int xj;
    
   // Select node
    if (xi == -1){
      node = ft.root;
    } else if (xi >=0 && xi < ft.inner_nodes.size()){
      node = ft.inner_nodes.at(xi);
    } else if (xi >=0 && xi < 2*ft.inner_nodes.size()){
      xj = xi - ft.inner_nodes.size();
      ft.leave_nodes[xj]->factor = std::make_shared<Factor_data>(ft.variables[xj], ft.num_categories, indicators[0]->at(0).size());
      ft.leave_nodes[xj]->factor->set_params_evidence(parameters[xj], indicators, ft.num_categories);        
      return;
    }
    else{
      throw std::runtime_error("Variable index out of bounds");
    }
  	
  	// Compute in children nodes
    // TODO: Compute in leaf children
    for(auto child = node->children.begin(); child != node->children.end(); ++child)
    {
      xj = (*child).lock()->id;
//      std::cout << "xi: " << xi << ", xj: " << xj << "\n";
      if (xj >=0 && xj < 2*ft.inner_nodes.size()){
        sum_compute_tree_aux(ft,xj, parameters, indicators);
      } else if (xj <0 || xj >= 2*ft.inner_nodes.size()){
        throw std::runtime_error("Variable index out of bounds");
      }
    }
  
  	// Compute node xi
//    std::cout << "sum_compute: " << xi << "\n";
    ft.sum_compute(xi);

     // TODO: Clean memory of the children
}



/*Computes the factors all nodes (evidence propagation)*/
void Factor_Tree_data::sum_compute_tree(std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators){
    this->data_size = indicators[0]->at(0).size();
    sum_compute_tree_aux(*this, -1, parameters, indicators);     
}





void sum_compute_query_aux(Factor_Tree_data &ft, int xi, std::vector<bool> &variables, std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators){
    std::shared_ptr<Factor_Node_data> node;
    int xj;
   // Select node
    if (xi == -1){
      node = ft.root;
    } else if (xi >=0 && xi < ft.inner_nodes.size()){
      node = ft.inner_nodes.at(xi);
    } else if (xi >=0 && xi < 2*ft.inner_nodes.size()){
      xj = xi - ft.inner_nodes.size();
      ft.leave_nodes[xj]->factor = std::make_shared<Factor_data>(ft.variables[xj], ft.num_categories, indicators[0]->at(0).size());
      ft.leave_nodes[xj]->factor->set_params_evidence(parameters[xj], indicators, ft.num_categories);        
      return;
    }
    else{
      throw std::runtime_error("Variable index out of bounds");
    }
  	
  	// Compute in children nodes
    for(auto child = node->children.begin(); child != node->children.end(); ++child)
    {
      xj = (*child).lock()->id;
      if (xj >=0 && xj < 2*ft.inner_nodes.size()){
        sum_compute_query_aux(ft,xj, variables, parameters, indicators);
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
std::ofstream myfile;
         ft.sum_compute(xi);
    }
}



/*Computes the joint PD of variables*/
void Factor_Tree_data::sum_compute_query( std::vector<bool> &variables, std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators){
    this->data_size = indicators[0]->at(0).size();
   sum_compute_query_aux(*this, -1, variables, parameters, indicators);     
}





std::vector<double> reorder_se(std::vector<double> &se, std::vector<int> &num_categories, std::vector<int> &variables_old, std::vector<int> &variables_new){

    std::vector<double> se_new(se.size());
    std::vector<int> values(variables_old.size(),0);
    std::vector<int> values_new(variables_old.size(),0);
    for(unsigned i = 0; i < se.size(); i++)
    {
        auto order = get_value_order(variables_new, variables_old, num_categories);
        for (unsigned j=0; j<values.size();++j){
            values_new[j] = values[order[j]];
        }
        unsigned pos_f_prod = position_in_factor(values_new, variables_new, num_categories);    
        se_new[pos_f_prod] = se[i];
        increase_value(values, variables_old, num_categories);
    }
    return se_new;

}






/*Get the sufficent statistics for cluster*/
std::vector<double> Factor_Tree_data::se_data(std::vector<int> &cluster_int, std::vector<std::vector<double>> &parameters, std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators){
    unsigned ncols = indicators.size();
    std::vector<bool> cluster(ncols,false);
    for (auto cl=cluster_int.begin();cl<cluster_int.end();++cl){
        cluster[*cl] = true;
    }
    
    this->sum_compute_query(cluster,parameters,indicators);
    unsigned nconf =this->root->factor->num_conf;   
    std::vector<double> vprob(nconf);
    double *prob_array = this->root->factor->prob;
    unsigned data_size = this->data_size;
    for (unsigned j=0; j<data_size; ++j){
        double sum = 0;
        for (unsigned i=0; i<nconf; ++i){
            sum += prob_array[i*data_size + j];
        }
        if (sum == 0){
            for (unsigned i=0; i<nconf; ++i){vprob[i] += 1.0 / nconf;}
        } else{
            for (unsigned i=0; i<nconf; ++i){
                vprob[i] += prob_array[i*data_size + j]/sum;
            }
        }
    }
    return reorder_se(vprob, this->num_categories, this->root->factor->variables, cluster_int);
}

/*Get the sufficent statistics for cluster in parallel*/
std::vector<double> Factor_Tree_data::se_data_parallel(std::vector<int> &cluster_int, std::vector<std::vector<double>> &parameters, std::vector<std::vector<std::shared_ptr<std::vector<std::vector<int>>>>> &indicators_vector){
        
    std::vector<double> se; 
    #pragma omp parallel for shared(cluster_int, parameters, indicators_vector)
    for (auto it=0;it<indicators_vector.size();++it){
        std::shared_ptr<Factor_Tree_data> ftd = this->copy();
        std::vector<double> se_aux = ftd->se_data(cluster_int, parameters, indicators_vector[it]);
        #pragma omp critical
        {
        if (se.size()==0){se=se_aux;}
        else{for (auto i = 0; i < se.size(); ++i){se[i] += se_aux[i];}}
        }
    }
    return se;
}




/*Distributes evidence (Junction Tree)*/
void Factor_Tree_data::distribute_evidence(){  
    auto node = this->root;
    auto children = node->children;
    for(auto j = 0; j!=children.size(); ++j){
        auto xj = children.at(j).lock()->id;
        this->distribute_evidence(xj,nullptr);
    }
}


void Factor_Tree_data::distribute_evidence(int xi, std::shared_ptr<Factor_data> message){
  std::shared_ptr<Factor_Node_data> node;
  std::vector<std::shared_ptr<Factor_data>> children_factors;
  std::shared_ptr<Factor_data> result_factor;
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
              result_factor = prod_factor(children_factors, this->num_categories, this->data_size);
              auto vars = result_factor->variables;
              auto vars_xj = children.at(j).lock()->factor->variables;
              for (auto xm=vars.begin(); xm!=vars.end();++xm){
                  if(std::find(vars_xj.begin(),vars_xj.end(), *xm) == vars_xj.end()){
                      result_factor = sum_factor(*result_factor, *xm, this->num_categories, this->data_size);
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
       node->joint_factor =  prod_factor(children_factors, this->num_categories, this->data_size);
  }
}





// Compute the sufficient statistics of all the leaves
 std::vector<std::vector<double>> Factor_Tree_data::se_leaves(){
    std::vector<std::vector<double>> vprobs;
    unsigned xi = 0;
    for(auto node=this->leave_nodes.begin();node!=this->leave_nodes.end();++node){
        auto cluster_int = (*node)->factor->variables; 
        auto fse = std::make_shared<Factor_data>(cluster_int, this->num_categories, this->data_size); 
        auto parent_node = (*node)->parent;
        auto vars = parent_node.lock()->joint_factor->variables;
        auto result_factor = std::make_shared<Factor_data>(*(parent_node.lock()->joint_factor));
        //Sum irrelevant variables
        for (auto xm=vars.begin(); xm!=vars.end();++xm){
          if(std::find(cluster_int.begin(),cluster_int.end(), *xm) == cluster_int.end()){
              result_factor = sum_factor(*result_factor, *xm, this->num_categories, this->data_size);
          }
        }
        //Divide by probability of evidence and sum
        unsigned nconf = result_factor->num_conf;   
        std::vector<double> vprob(nconf);
        for (unsigned i=0; i<nconf; ++i){vprob[i]=0;}
        double *prob_array = result_factor->prob;
        unsigned data_size = this->data_size;
        for (unsigned j=0; j<data_size; ++j){
            double sum = 0;
            for (unsigned i=0; i<nconf; ++i){
                sum += prob_array[i*data_size + j];
            }
            if (sum == 0){
                for (unsigned i=0; i<nconf; ++i){vprob[i] += 1.0 / nconf;}
            } else{
                for (unsigned i=0; i<nconf; ++i){
                    vprob[i] += prob_array[i*data_size + j]/sum;
                }
            }
        }
        vprobs.push_back(reorder_se(vprob, this->num_categories, result_factor->variables, this->variables[xi]));
        xi++;
    }

    return vprobs;
}


/* Perform num_it iterations of em */
std::vector<double> Factor_Tree_data::em(std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators, int num_it, double alpha){
    std::vector<double> ll_v;    
    for (unsigned i=0; i<num_it; ++i){
        this->sum_compute_tree(this->parameters,indicators);
        this->distribute_evidence();
        std::vector<std::vector<double>> se = this->se_leaves();
        for (unsigned j=0; j<this->parameters.size(); ++j){
            double ll = this->learn_params_se(se[j], alpha, j);
            if (i == (num_it-1)){ll_v.push_back(ll);}
        }
    }
    return ll_v;
}



std::vector<double> Factor_Tree_data::em_parallel(std::vector<std::vector<std::shared_ptr<std::vector<std::vector<int>>>>> &indicators_vector, int num_it, double alpha){
    std::vector<double> ll_v;    

    for (unsigned it_em=0; it_em<num_it; ++it_em){
        std::vector<std::vector<double>> se_vector(this->parameters.size()); 
        for (auto i=0; i<se_vector.size(); ++i){
            se_vector[i].resize(this->parameters[i].size());
        }
        
        #pragma omp parallel for shared(indicators_vector)
        for (auto it=0;it<indicators_vector.size();++it){
            std::shared_ptr<Factor_Tree_data> ftd = this->copy();
            ftd->sum_compute_tree(ftd->parameters,indicators_vector[it]);
            ftd->distribute_evidence();
            std::vector<std::vector<double>> se = ftd->se_leaves();
            #pragma omp critical  
            {for (auto i = 0; i < se.size(); ++i){
               for(auto j = 0; j < se[i].size(); ++j){
                    se_vector[i][j] += se[i][j];
                }
            }}
        }

        for (unsigned j=0; j<this->parameters.size(); ++j){
            double ll = this->learn_params_se(se_vector[j], alpha, j);
            if (it_em == (num_it-1)){ll_v.push_back(ll);}
        }
    }

    return ll_v;
}




/* Learn parameters from se */
double Factor_Tree_data::learn_params_se(std::vector<double> &se, double alpha, unsigned xi){
  int i,j,nc_xi,loops;
  double Nij = 0;
  double ll=0.0, Nijk;
  int* vars = &(this->variables[xi][0]);
  int l_vars = this->variables[xi].size();
  int* ncat = &(this->num_categories[0]);
  int ncf= nconf(vars,ncat,l_vars);
  nc_xi = this->num_categories[xi];
  loops = ncf/nc_xi;
  this->parameters[xi].resize(ncf);
//  std::cout << "ncf: " << ncf << ", se.size() " << se.size() << "\n";
  for ( i = 0; i < loops; i++ ){
      /* Obtain Nij */
      Nij = 0;
      for ( j = 0; j < nc_xi; j++ ){
          Nij += se[i*nc_xi + j];
      }
      if (Nij != 0) {
          for ( j = 0; j < nc_xi; j++ ){
            Nijk = se[i*nc_xi + j];  
            this->parameters[xi][i*nc_xi + j] = (1.0* (Nijk+alpha))/(Nij+nc_xi*(alpha * 1.0));
            if (Nijk != 0){
                    
                ll += Nijk * log(Nijk/Nij);
            }
          }
      }else{
          for ( j = 0; j < nc_xi; j++ ){
          	this->parameters[xi][i*nc_xi + j] = 1.0/nc_xi;
          }
      }
  }
  return ll;
}


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
        if (pos != -2){
            if (pos == -1){
                return NULL;
            } else{
                fq[pos]++;
            }
        }
    }

    return fq;
}

/*Learns parameters from data*/
void Factor_Tree_data::learn_params(const int* data, int nrows, int ncols, double alpha, unsigned xi){
  int *fq;
  int i,j,nc_xi,loops;
  double Nij = 0;
  int* vars = &(this->variables[xi][0]);
  int l_vars = this->variables[xi].size();
  int* ncat = &this->num_categories[0];
  int ncf= nconf(vars,ncat,l_vars);
  //Get the dimension of the probability vector
  
    
  /* Obtain frequencies */
//  std::cout << "LLLEEEGOOOO_antes\n";
  fq = freq(data, nrows, ncols, vars, ncat, l_vars);
//  std::cout << "LLLEEEGOOOO_des\n";
  if (fq == NULL){
      throw std::runtime_error("Error obtaining frequency");
  }
  
  nc_xi = ncat[vars[0]];
  loops = ncf/nc_xi;
//  std::cout << "LLLEEEGOOOO\n";
//  std::cout << "xi: "<< xi<< ", this->parameters[xi].size()" << this->parameters[xi].size() << "\n";

  for ( i = 0; i < loops; i++ ){
      /* Obtain Nij */
      Nij = 0;
      for ( j = 0; j < nc_xi; j++ ){
          Nij += fq[i*nc_xi + j];
      }
      if (Nij != 0) {
          for ( j = 0; j < nc_xi; j++ ){
              this->parameters[xi][i*nc_xi + j] = (1.0* (fq[i*nc_xi + j]+alpha))/(Nij+nc_xi*(alpha * 1.0));
          }
      }else{
          for ( j = 0; j < nc_xi; j++ ){
          	this->parameters[xi][i*nc_xi + j] = 1.0/nc_xi;
          }
      }
  }
  free(fq);
}

void Factor_Tree_data::learn_params(const int* data, int nrows, int ncols, double alpha){
  for (unsigned xi=0;xi<this->leave_nodes.size();++xi){
      this->learn_params(data, nrows, ncols, alpha, xi);
  }
}

double Factor_Tree_data::log_likelihood(std::vector<std::shared_ptr<std::vector<std::vector<int>>>> &indicators){
    sum_compute_tree(this->parameters,indicators);
    double* prob_array = this->root->factor->prob;
    double ll = 0;
    for(unsigned i=0; i<this->data_size; ++i){
        if (prob_array[i]!=0){ll += log(prob_array[i]);}
        else{ll += log(std::numeric_limits<double>::denorm_min());}
    } 
    return ll;
}

double Factor_Tree_data::log_likelihood_parallel(std::vector<std::vector<std::shared_ptr<std::vector<std::vector<int>>>>> &indicators_vector){
    double ll_sum = 0;
    #pragma omp parallel for shared(ll_sum, indicators_vector)
    for (auto it=0;it<indicators_vector.size();++it){
        double ll = 0;
        std::shared_ptr<Factor_Tree_data> ftd = this->copy();
        ftd->sum_compute_tree(ftd->parameters,indicators_vector[it]);
        double* prob_array = ftd->root->factor->prob;
        for(unsigned i=0; i<ftd->data_size; ++i){
            if (prob_array[i]!=0){ll += log(prob_array[i]);}
            else{ll += log(std::numeric_limits<double>::denorm_min());}
        } 
        #pragma omp critical
        ll_sum += ll;
    }
    return ll_sum;
}





/*Generate smart pointer to a copy of the Factor_Tree*/
std::shared_ptr<Factor_Tree_data> Factor_Tree_data::copy(){
    std::vector<int> parents_inner, parents_leave;
    for (auto i = 0; i < this->inner_nodes.size(); i++){
        parents_inner.push_back(this->inner_nodes[i]->parent.lock()->id);
    }
    for (auto i = 0; i < this->leave_nodes.size(); i++){
        parents_leave.push_back(this->leave_nodes[i]->parent.lock()->id);
    }
    std::shared_ptr<Factor_Tree_data> ftd = std::make_shared<Factor_Tree_data>(parents_inner, parents_leave, this->variables, this->num_categories);
    ftd->parameters = this->parameters;
    return ftd;
};








