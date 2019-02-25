#ifndef ELIM_TREE_H
#define ELIM_TREE_H


#include <vector>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <math.h>


class Elimination_Tree {
    public:
        Elimination_Tree();
        Elimination_Tree(int n);
        Elimination_Tree(const Elimination_Tree& et);
        ~Elimination_Tree();
  
        //Methods
        void set_parent(int xp, int xi, bool inner);
        std::vector<int> get_pred(int xi, bool inner);
        std::vector<int> add_arc(int xout,int xin);
        std::vector<int> remove_arc(int xout,int xin);
        void compute_cluster(int xi);
        void compute_clusters();
        int tw();
        int size();
        int sum_width();
        bool soundness_node(int xi);
        bool soundness();
        bool completeness_node(int xi, bool inner);
        bool completeness();
        void swap(int xi);
        void optimize(std::vector<int> xopt);
        bool check_clusters();
        bool check_cluster(int xi);
        std::vector<int> topo_sort(std::vector<int> parents_inner, std::vector<std::vector<int>> children_inner);

        //Attributes
        std::vector<std::vector<int>> cluster_inner, cluster_leaves, children_inner, children_leaves; 
        std::vector<int> parents_inner, parents_leaves;
};
        
        
        
#endif