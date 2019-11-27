//
// Created by andy on 2019-11-25.
//

#ifndef TSNE_TSNE_HPP
#define TSNE_TSNE_HPP

#include <tree.hpp>

namespace tsne{

template<typename T>
class TSNE{

    protected:

    ushort x_dim;
    ushort y_dim;
    VpTree<T> *vp_tree;
    std::vector<T*> X;
    std::vector<T*> Y;
    T* X_mean;

    struct Matrix{
        size_t n_rows;
        size_t n_cols;
        T *vals;
        size_t *indices;
        bool is_symmtric;

        Matrix():n_rows(0),n_cols(0), is_symmtric(false), vals(nullptr), indices(nullptr){}
        Matrix(size_t n_rows, size_t n_cols): n_rows(n_rows), n_cols(n_cols),is_symmtric(false){
           vals = new T[n_rows * n_cols];
           indices = new size_t[n_rows * n_cols];
        }
        ~Matrix(){
            delete vals;
            delete indices;
        }

        T* get(size_t row_i, size_t col_j);
        void set(size_t row_i, size_t col_j, size_t idx, T v);
        void makeSymmtric();
    };

    void initX(size_t n, T *x);
    void computeGradient();
    void computeGaussianPerplexity(size_t n, T perplexity, T *x, Matrix &mat);
    void computeGaussianPerplexity(size_t n, size_t k, T perplexity, T *x, Matrix &mat);

    void zeroMean(size_t n, ushort d, T *mean, T **inp);

    public:

    TSNE():x_dim(0), y_dim(0), vp_tree(nullptr), X_mean(nullptr){}

    explicit TSNE(ushort x_dim, ushort y_dim): x_dim(x_dim), y_dim(y_dim), vp_tree(nullptr), X_mean(nullptr){}

    ~TSNE(){
        delete vp_tree;
        for(auto iter = X.begin(); iter != X.end(); iter++){
            delete (*iter);
        }
        for(auto iter = Y.begin(); iter != Y.end(); iter++){
            delete (*iter);
        }
    }

    void run(size_t n, T *x, T* y, T perplexity, T theta, int rand_seed,
             bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter);

    void reIndex();

};

}

#endif //TSNE_TSNE_HPP
