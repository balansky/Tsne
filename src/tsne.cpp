//
// Created by andy on 2019-11-25.
//

#include <cstring>
#include "tsne.hpp"

namespace tsne{


    template<typename T>
    void TSNE<T>::run(size_t n, T *x, T *y, T perplexity, T theta, int rand_seed, bool skip_random_init, int max_iter,
                      int stop_lying_iter, int mom_switch_iter) {
//        if(X.empty()){
//            initX(n, x);
//        }
//        else{
//            for(size_t i = 0; i < n; i ++){
//                T *cx = new T[x_dim];
//                memcpy(cx, x + i*x_dim, sizeof(T)*x_dim);
//                X.push_back(cx);
//            }
//            zeroMean(n, x_dim, X_mean, X.data() + (X.size() - n));
//        }
//        vp_tree = new VpTree<T>(n, x_dim, x);

    }

//    template<typename T>
//    void TSNE<T>::initX(size_t n, T *x) {
//        delete X_mean;
//        X_mean = (T*)calloc(x_dim, sizeof(T));
//        for(size_t i = 0; i < n; i ++){
//            T *cx = new T[x_dim];
//            memcpy(cx, x + i*x_dim, sizeof(T)*x_dim);
//            X.push_back(cx);
//            for(ushort d = 0; d < x_dim; d++){
//                X_mean[d] += cx[d];
//            }
//        }
//        for(ushort d = 0; d < x_dim; d++) X_mean[d] /= (T)n;
//        zeroMean(n, x_dim, X_mean, X.data());
//    }
//
//    template<typename T>
//    void TSNE<T>::zeroMean(size_t n, ushort dim, T *mean, T **inps) {
//        for(size_t i = 0; i < n; i++){
//            for(ushort d = 0; d < dim; d++){
//                inps[i][d] -= mean[d];
//            }
//        }
//    }


    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t n, T perplexity, T *x, tsne::TSNE<T>::Matrix &mat) {

    }

    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t n, size_t k, T perplexity, T *x, tsne::TSNE<T>::Matrix &mat) {
//        if(k == n) computeGaussianPerplexity(n, perplexity, x, mat);
//
//        for(size_t i = 0; i < n; i++){
//            std::vector<T> dists(k);
//            std::vector<size_t> indices(k);
//            vp_tree->search(x + i * x_dim, k, indices, dists);
//        }

    }

    template<typename T>
    void TSNE<T>::computeGradient() {

    }

    template <typename T>
    T* TSNE<T>::Matrix::get(size_t row_i, size_t col_j) {

    }


    template <typename T>
    void TSNE<T>::Matrix::set(size_t row_i, size_t col_j, size_t idx, T v) {
        vals[row_i * n_cols + col_j] = v;
        indices[row_i * n_cols + col_j] = idx;
    }

    template <typename T>
    void TSNE<T>::Matrix::makeSymmtric() {

    }


    template class TSNE<float>;
    template class TSNE<double>;

}
