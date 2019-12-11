//
// Created by andy on 2019-11-25.
//

#ifndef TSNE_TSNE_HPP
#define TSNE_TSNE_HPP

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <ctime>
#include <tree.hpp>

namespace tsne{

template<typename T>
class TSNE{

    protected:

    ushort x_dim;
    ushort y_dim;
    size_t n_total;
    std::vector<T*> X;
    std::vector<T*> Y;
    RedBlackTree<size_t, T> *rb_tree;

//    void initX(size_t n, T *x);
    void makeSymmtric(size_t n_offset, std::unordered_map<size_t, T> *lk, size_t **row_P, size_t **col_P, T **val_P);
    void computeGradient(size_t run_n, size_t offset, T theta, size_t *row_P, size_t *col_P, T *val_P, T *dY);
    void searchGaussianPerplexity(size_t k, T perplexity, T *dist, T *cur_P);
    void computeGaussianPerplexity(size_t n_offset, size_t k, T perplexity, size_t **row_P, size_t **col_P, T **val_P);
//    void computeGaussianPerplexity(size_t n, T perplexity, T *x, Matrix &mat);

    public:

    TSNE():x_dim(0), y_dim(0), n_total(0){}

    TSNE(ushort x_dim, ushort y_dim): x_dim(x_dim), y_dim(y_dim), n_total(0), rb_tree(nullptr){}

    TSNE(size_t n, ushort x_dim, ushort y_dim, T *x, T *y): TSNE(x_dim, y_dim){
        insertItems(n, x, y);
    };


    ~TSNE(){
        delete rb_tree;
        for(auto iter = X.begin(); iter != X.end(); iter++){
            delete (*iter);
        }
        for(auto iter = Y.begin(); iter != Y.end(); iter++){
            delete (*iter);
        }
    }

    void insertItems(size_t n, T *x, T *y);

    void testGaussianPerplexity(size_t n_offset, size_t k, T perplexity, size_t **row_P, size_t **col_P, T **val_P);

    void testGradient(size_t offset, T perplexity, T theta, T *dY);

    void run(size_t n, T *x, T* y, T perplexity, T theta, bool exact,
             bool partial, int max_iter, int stop_lying_iter, int mom_switch_iter);


};

}

#endif //TSNE_TSNE_HPP
