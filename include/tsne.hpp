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
    const ushort dim;
    size_t n_total;
    VpTree<T> *vp_tree;
    BarnesHutTree<T> *bh_tree;

    struct Matrix{
        size_t n_rows;
        size_t n_cols;
        T *vals;
        bool is_symmtric;

        Matrix():n_rows(0),n_cols(0), is_symmtric(false){}
        Matrix(size_t n_rows, size_t n_cols): n_rows(n_rows), n_cols(n_cols),is_symmtric(false){
           vals = new T[n_rows * n_cols];
        }

        T* get(size_t row_i, size_t col_j);
        void set(size_t row_i, size_t col_j, T v);
        void makeSymmtric();
    };

    void computeGradient();
    void computeGaussianPerplexity(size_t n, T perplexity, Matrix &mat);

    public:

    TSNE():dim(0), n_total(0), vp_tree(nullptr), bh_tree(nullptr){}

    explicit TSNE(const ushort dim): dim(dim), n_total(0), vp_tree(nullptr), bh_tree(nullptr){}

    TSNE(size_t n, ushort dim, T *data);

    void run(T* y, T perplexity, T theta, int rand_seed,
             bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter);

    void run(size_t xn, T *x, T* y, T perplexity, T theta, int rand_seed,
             bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter);



};

}

#endif //TSNE_TSNE_HPP
