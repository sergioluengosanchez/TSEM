#include "elim_tree.h"


Elimination_Tree::Elimination_Tree(){};
Elimination_Tree::Elimination_Tree(int n){
            this->cluster_inner.resize(n);
            this->cluster_leaves.resize(n);
            this->children_inner.resize(n);
            this->children_leaves.resize(n);
            this->parents_inner.resize(n);
            this->parents_leaves.resize(n);
            for(auto it = this->parents_inner.begin(); it != this->parents_inner.end(); ++it){ *it = -1;}
            for(auto i = 0; i < this->parents_leaves.size(); ++i){ this->parents_leaves[i] = i;}
            for(auto i = 0; i < this->cluster_inner.size(); ++i){ this->cluster_inner[i].push_back(i);}
            for(auto i = 0; i < this->cluster_leaves.size(); ++i){ this->cluster_leaves[i].push_back(i);}
            for(auto i = 0; i < this->children_leaves.size(); ++i){ this->children_leaves[i].push_back(i);}
        };
Elimination_Tree::~Elimination_Tree(){};

/*Copy constructor for Factor_Tree*/
Elimination_Tree::Elimination_Tree(const Elimination_Tree& et){
            this->cluster_inner = et.cluster_inner;
            this->cluster_leaves = et.cluster_leaves;
            this->parents_inner = et.parents_inner;
            this->parents_leaves = et.parents_leaves;
            this->children_inner = et.children_inner;
            this->children_leaves = et.children_leaves;
};
  

std::vector<int> Elimination_Tree::get_pred(int xi, bool inner){
    std::vector<int> pred;
    int current_x = xi;
    if (!inner){
        current_x = this->parents_leaves.at(current_x);
//        std::cout << "current_x: " << current_x <<"\n";
        pred.push_back(current_x);
    }
    current_x = this->parents_inner.at(current_x);
    while (current_x != -1){
        pred.push_back(current_x);
        current_x = this->parents_inner.at(current_x);
    }
    return pred;
}

void remove(std::vector<int> &v, int x){
    auto it = std::find(v.begin(),v.end(),x);
    // check that there actually is a 3 in our vector
    if (it != v.end()) {
      v.erase(it);
    }
}


void Elimination_Tree::set_parent(int xp, int xi, bool inner){
    std::vector<int> cluster;

    if (inner){
        int xp_old = this->parents_inner.at(xi);
        if (xp_old!= -1){ remove(this->children_inner[xp_old], xi);}
        if (xp!=-1){this->children_inner[xp].push_back(xi);}
        this->parents_inner[xi] = xp ;
    }    
    else{
        int xp_old = this->parents_leaves.at(xi);
        if (xp_old!= -1){ remove(this->children_leaves[xp_old], xi);}
        this->children_leaves[xp].push_back(xi);
        this->parents_leaves[xi] = xp ;
    }
}

void  print_vector(std::vector<int> v){
    std::cout << "[";
    for (auto i = v.begin(); i != v.end(); ++i) std::cout << *i << ", ";
    std::cout << "]\n";
}

void Elimination_Tree::compute_cluster(int xi){
    std::vector<int> cluster;
//    std::cout << "node xi: "<<xi<<"\n";
    for(auto it = this->children_inner[xi].begin(); it != this->children_inner[xi].end(); ++it)
    {
//        std::cout << "child inner: "<<*it<<"\n";
        std::vector<int> cluster_aux, cluster_aux2;
//        print_vector(this->cluster_inner[*it]);
        if (cluster.size()==0){
            cluster_aux =  this->cluster_inner[*it];
            remove(cluster_aux,*it);       
        } else{
            cluster_aux2 = this->cluster_inner[*it];
            remove(cluster_aux2,*it);
            set_union(cluster.begin(), cluster.end(), cluster_aux2.begin(), cluster_aux2.end(), std::back_inserter(cluster_aux));
        }        
        cluster = cluster_aux;
//        print_vector(cluster);
//        std::cout << "\n";
    }
    for(auto it = this->children_leaves[xi].begin(); it != this->children_leaves[xi].end(); ++it)
    {
//        std::cout <<"child leave: "<<*it<<"\n";
//        print_vector(this->cluster_leaves[*it]);
        std::vector<int> cluster_aux;
        if (cluster.size()==0){
            cluster_aux =  this->cluster_leaves[*it];       
        } else{
            std::set_union (cluster.begin(), cluster.end(), this->cluster_leaves[*it].begin(), this->cluster_leaves[*it].end(), std::back_inserter(cluster_aux));
        }        
        cluster = cluster_aux;
//        std::cout << "CLUSTEEER\n";
//        print_vector(cluster);
    }

    this->cluster_inner[xi]=cluster;

}


std::vector<int> Elimination_Tree::topo_sort(std::vector<int> parents_inner, std::vector<std::vector<int>> children_inner){
    std::vector<int> tsorted, aux, aux2;
    
    for(auto i = 0; i != parents_inner.size(); ++i){
        if (parents_inner[i]==-1){
            tsorted.push_back(i);
            aux.push_back(i);
        }
    }
    while(aux.size()>0){
        aux2 = aux;
        aux.clear();
        for(auto it = aux2.begin(); it != aux2.end(); ++it){
            for(auto it2 = children_inner[*it].begin(); it2 != children_inner[*it].end(); ++it2){
                tsorted.push_back(*it2);
                aux.push_back(*it2);
            }
        }
    }
    return tsorted;
}


void Elimination_Tree::compute_clusters(){
    std::vector<int> tsorted = topo_sort(this->parents_inner, this->children_inner);
    for(auto it = tsorted.rbegin(); it != tsorted.rend(); ++it){this->compute_cluster(*it);}
}
    
int  Elimination_Tree::sum_width(){
    int mwidth = 0;
    for (auto it = this->cluster_inner.begin(); it!=this->cluster_inner.end(); ++it){
       mwidth += it->size();     
    }
    return mwidth;        
}


int  Elimination_Tree::tw(){
    int mwidth = 0;
    for (auto it = this->cluster_inner.begin(); it!=this->cluster_inner.end(); ++it){
        if (it->size()> mwidth){
           mwidth = it->size();     
        }    
    }
    return mwidth-1;        
}

int  Elimination_Tree::size(){
    int mwidth = 0;
    for (auto it = this->cluster_inner.begin(); it!=this->cluster_inner.end(); ++it){
        mwidth += pow(2,it->size());     
    }
    return mwidth;        
}

// Check if a node is sound
bool Elimination_Tree::soundness_node(int xi){
    std::vector<int> pred = this->get_pred(xi,true);
    std::vector<int> cluster = this->cluster_inner.at(xi);
    bool flag;

    for (auto it=cluster.begin(); it!=cluster.end(); ++it){
        if (*it!=xi){        
            flag = false;        
            for (auto it2 = pred.begin(); it2!=pred.end();++it2){
                if (*it ==*it2) {flag = true; break;}        
            }
            if (flag ==false){
                std::cout << *it << " not in pred of " << xi << "\n";
                return false;
            }
        }
    }
    return true;
}

bool Elimination_Tree::soundness(){
    for (auto i=0; i!=this->cluster_inner.size(); ++i){
        if (!this->soundness_node(i)){return false;}
    }
    return true;
}




std::vector<int> Elimination_Tree::add_arc(int xout,int xin){
    int xf = this->parents_leaves.at(xin);
    std::vector<int> xopt, xopt_aux;
//    std::cout << "entro -2\n";
    std::vector<int> pred_fin = this->get_pred(xin,false);
//    std::cout << "entro -1\n";
    auto it1 = std::find(pred_fin.begin(),pred_fin.end(),xout);
    int xk,xh,x_current;
//    std::cout << "entro 0\n";
    this->cluster_leaves[xin].push_back(xout);
    std::sort(this->cluster_leaves[xin].begin(), this->cluster_leaves[xin].end());
//    std::cout << "entro 1\n";
    if (it1 != pred_fin.end()){
//        std::cout << "ES1111111111111 \n";
//        std::cout << "entro 2\n";
        for (auto it2 = pred_fin.begin(); it2!=it1; ++it2){
            this->compute_cluster(*it2);
            xopt.push_back(*it2);
        }
    } else {
        std::vector<int> pred_xout = this->get_pred(xout,true);
        it1 = std::find(pred_xout.begin(),pred_xout.end(),xf);
        if (it1 != pred_xout.end()){
//            std::cout << "ES222222222222222 \n";
//            std::cout << "entro 3\n";
            this->set_parent(xout,xin,false);
            this->compute_cluster(xout);
            xopt.push_back(xout);
            for (auto it2 = pred_xout.begin(); it2!=it1; ++it2){
                this->compute_cluster(*it2);
                xopt.push_back(*it2);
            }    
        } else{
//            std::cout << "entro\n";
            Elimination_Tree et_aux = *this;
            auto it1 = pred_xout.begin();
            auto it2 = it1;
            bool flag = true;
            // Find xk and xh
//            std::cout << "preds fin old\n";
//            print_vector(this->get_pred(xin,false));
//            std::cout << "preds xout\n";
//            print_vector(this->get_pred(xout,true));
            while (it1!= pred_xout.end() && flag){
                it2 = std::find(pred_fin.begin(),pred_fin.end(),*it1);
                if (it2 != pred_fin.end()){
                    flag = false;
//                    std::cout << "*(it1): " << *(it1) << ", *(it2) " << *(it2) <<"\n";
                    it2--;
                    xk = *(it2);
                    if (it1==pred_xout.begin()){xh=xout;}
                    else {
                        it1--;
                        xh = *(it1);
                    }
                }
                it1++;
            }
            if (flag){
//                std::cout << "CCCCCC3 \n";
                xk = pred_fin.back();
                if (pred_xout.size()>0){xh = pred_xout.back();}
                else {xh = xout;}
            }
//            std::cout << "xout: " << xout <<  ", xin: " << xin <<  ", xk: " << xk <<  ", xh: " << xh <<  ", xf: " << xf << "\n";
//            std::cout << "xk: " << xk << ", xh: " << xh <<"\n";
            this->set_parent(xout, xk,true);
            et_aux.set_parent(xf, xh,true);
            et_aux.set_parent(xout, xin,false);
//            std::cout << "preds fin in this\n";
//            print_vector(this->get_pred(xin,false));
            x_current = xf;
            flag = true;            
            while(flag){
                xopt.push_back(x_current);
//                std::cout << "\n";
//                std::cout << "x_current: "<< x_current <<"\n";
                this->compute_cluster(x_current);
//                print_vector(this->cluster_inner.at(x_current));
                x_current = this->parents_inner.at(x_current);
                if (x_current == this->parents_inner.at(xh)){flag=false;} 
            }

            x_current = xout;
            flag = true ;           
            while(flag){
                xopt_aux.push_back(x_current);
                et_aux.compute_cluster(x_current);
                x_current = et_aux.parents_inner.at(x_current);
                if (x_current == et_aux.parents_inner.at(xk)){flag=false;} 
            }
            
//            std::cout << "et_aux.tw(): " << et_aux.tw() << ", this->tw(): " << this->tw() <<"\n";
//            std::cout << "ES33333333333333 \n";
            if (et_aux.tw() < this->tw() || (et_aux.tw() == this->tw() && et_aux.size() < this->size())){
//                std::cout << "ESAUUX ";
                *this = et_aux;
                xopt = xopt_aux;
            }                
        }
    
    }
    return(xopt);
        
}
        
        
// Check if a node is sound
bool Elimination_Tree::completeness_node(int xi, bool inner){
    std::vector<int> cluster;
    int xp;
    if (inner){
        xp = this->parents_inner.at(xi);
        if ( xp == -1){
            return true;
        }
        cluster = this->cluster_inner.at(xi);
    } else{ 
        xp = this->parents_leaves.at(xi);
        cluster = this->cluster_leaves.at(xi);            
    }
    auto it1 = std::find(cluster.begin(),cluster.end(),xp);
    if (it1 == cluster.end()) {
//        std::cout << xp << " not in cluster of " << xi << "\n";
        return false;
    }
    return true;
}

bool Elimination_Tree::completeness(){
    for (auto i=0; i!=this->cluster_inner.size(); ++i){
        if (!this->completeness_node(i,true)){return false;}
    }
    return true;
}
            
std::vector<int> Elimination_Tree::remove_arc(int xout,int xin){
    std::vector<int> xopt;
    int xi, xp, xpp;
    bool inner = false;
//    std::cout << "xout: " << xout << ", xin: " << xin << "\n";
    // Remove xout from the cluster of leave node xin
    remove(this->cluster_leaves[xin],xout);
    xi = xin;
    xp = this->parents_leaves.at(xi);
    
    //Find xi
    while (xp != xout){
        this->compute_cluster(xp);
        xopt.push_back(xp);
        xi = xp;
        xp = this->parents_inner.at(xi);
        inner = true;
    }
//    std::cout << "xi: " << xi << ", xp: " << xp << "\n\n";
//    std::cout << "inner: " << inner << "\n" ;
//    if (inner) {print_vector(this->cluster_inner[xi]);}
//    else{print_vector(this->cluster_leaves[xi]);}
    while (!this->completeness_node(xi,inner)){
        while (!this->completeness_node(xi,inner)){
            if (inner) {xp = this->parents_inner.at(xi);}
            else{xp = this->parents_leaves.at(xi);}
            xpp = this->parents_inner.at(xp);
//            std::cout << " cluster xi: \n";
//            if (inner) {print_vector(this->cluster_inner[xi]);}
//            else{print_vector(this->cluster_leaves[xi]);}
//            std::cout << " cluster xp: \n";
//            print_vector(this->cluster_inner[xp]);
//            std::cout << " new xp: " << xp<< "\n";
//            std::cout << " new xpp: " << xpp<< "\n";
            this->set_parent(xpp,xi,inner);
            this->compute_cluster(xp);
//            std::cout << " cluster xp after: \n";
//            print_vector(this->cluster_inner[xp]);
            xopt.push_back(xp);
        }
        inner = true;
        xi = xp;
//        std::cout << " new xi: " << xi<< "\n\n";
    }
    return xopt;

}        
        
        
// Swap node xi with its parent
void Elimination_Tree::swap(int xi){
    int xj = this->parents_inner.at(xi);
    std::vector<int> ch_aux;
//    std::cout << "\n";
//    std::cout << "children leaves of " << xi << "\n";
//    print_vector(this->children_leaves[xi]);
    this->set_parent(this->parents_inner[xj],xi,true);
    this->set_parent(xi,xj,true);
//    std::cout << "xi: " << xi<< "\n";
//    std::cout << "xj: " << xj<< "\n";
//    std::cout << "xpp: " << this->parents_inner[xj]<< "\n";
    ch_aux = this->children_inner[xi];
    for (auto it=ch_aux.begin(); it != ch_aux.end(); ++it){
        int xk = *it;
//        std::cout << "xk: " << xk<< "\n";
        if (xk != xj){
            auto it1 = std::find(this->cluster_inner[xk].begin(),this->cluster_inner[xk].end(),xj);
            if (it1!=this->cluster_inner[xk].end()){
                this->set_parent(xj,xk,true);
            }
        }
    }
//    std::cout << "children leaves of " << xi << "\n";
//    print_vector(this->children_leaves[xi]);
    ch_aux = this->children_leaves[xi];
    for (auto it=ch_aux.begin(); it != ch_aux.end(); ++it){
        int xk = *it;
//        std::cout << "xk leave: " << xk<< "\n";
        auto it1 = std::find(this->cluster_leaves[xk].begin(),this->cluster_leaves[xk].end(),xj);
        if (it1!=this->cluster_leaves[xk].end()){
            this->set_parent(xj,xk,false);
        }
    }
//    std::cout << "casco\n";
    this->compute_cluster(xj);
    this->compute_cluster(xi);
//    std::cout << "carrasco\n";
}
        
        
// Swap node xi with its parent
void Elimination_Tree::optimize(std::vector<int> xopt){
    bool flag;
    int old_tw, old_size, xp;
    for (auto it=xopt.begin(); it != xopt.end(); ++it){
//        std::cout << " new xi: " << *it<< "\n";
        flag = true;
        while (flag){
            xp = this->parents_inner.at(*it);
//            std::cout << " new xp: " << xp<< "\n";
            if (xp==-1){
                flag = false;
            } else{            
                old_tw = this->tw();
                old_size = this->size();
//                std::cout << " swap: " << *it<< "\n";
                this->swap(*it);
//                std::cout << " end swap: " << *it<< "\n";
                if (this->tw()> old_tw || (this->tw()== old_tw && this->size() > old_size)){
                    this->swap(xp);
                    flag = false;
                }
            }
        }
    }
}
            
            
bool Elimination_Tree::check_cluster(int xi){
    std::vector<int> cluster;
    for(auto it = this->children_inner[xi].begin(); it != this->children_inner[xi].end(); ++it)
    {
        std::vector<int> cluster_aux, cluster_aux2;
        if (cluster.size()==0){
            cluster_aux =  this->cluster_inner[*it];
            remove(cluster_aux,*it);       
        } else{
            cluster_aux2 = this->cluster_inner[*it];
            remove(cluster_aux2,*it);
            set_union(cluster.begin(), cluster.end(), cluster_aux2.begin(), cluster_aux2.end(), std::back_inserter(cluster_aux));
        }        
        cluster = cluster_aux;
    }
    for(auto it = this->children_leaves[xi].begin(); it != this->children_leaves[xi].end(); ++it)
    {
        std::vector<int> cluster_aux;
        if (cluster.size()==0){
            cluster_aux =  this->cluster_leaves[*it];       
        } else{
            std::set_union (cluster.begin(), cluster.end(), this->cluster_leaves[*it].begin(), this->cluster_leaves[*it].end(), std::back_inserter(cluster_aux));
        }        
        cluster = cluster_aux;
    }

    return cluster_inner[xi]==cluster;
}
        
bool Elimination_Tree::check_clusters(){
    for (auto i=0; i!=this->cluster_inner.size(); ++i){
        if (!this->check_cluster(i)){return false;}
    }
    return true;
}