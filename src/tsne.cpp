//
// Created by andy on 2019-11-25.
//

#include <cstring>
#include "tsne.hpp"


#ifdef _OPENMP
#include <omp.h>
#endif


namespace tsne{

    template<typename T>
    void TSNE<T>::insertItems(size_t n, T *x, T *y){
        if(!rb_tree){
            rb_tree = new RedBlackTree<size_t, T>(x_dim);
        }
        for(size_t i = 0; i < n; i++){
            T* x_ = new T[x_dim];
            T* y_ = new T[y_dim];
            memcpy(x_, x + i * x_dim, x_dim);
            memcpy(y_, y + i * y_dim, y_dim);
            X.push_back(x);
            Y.push_back(y);
            rb_tree->insert(1, &n_total, &x_);
            n_total++;
        }
    }


    template<typename T>
    void TSNE<T>::run(size_t n, T *x, T *y, T perplexity, T theta, bool exact, bool partial, int max_iter,
                      int stop_lying_iter, int mom_switch_iter) {
        size_t offset = 0;
        size_t run_n = n_total + n;
        if(partial) {
            offset = n_total;
            run_n = n;
        }
        insertItems(n, x, y);

        // Set learning parameters
        float total_time = .0;
        clock_t start, end;
        T momentum = .5, final_momentum = .8;
        T eta = 200.0;

        T* dY    = new T[run_n * y_dim];
        T* uY    = new T[run_n * y_dim];
        T* gains =  new T[run_n * y_dim];
        for(size_t i = 0; i < run_n * y_dim; i++)    uY[i] =  .0;
        for(size_t i = 0; i < run_n * y_dim; i++) gains[i] = 1.0;
        start = clock();
        if(exact){
            Matrix mat(run_n, n_total);

        }
        else{
            size_t k = (int) (3 * perplexity);
            Matrix mat(run_n, k);

        }


    }



    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t n, T perplexity, T *x, tsne::TSNE<T>::Matrix &mat) {

    }

    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t n_offset, size_t k, T perplexity, T *x, tsne::TSNE<T>::Matrix &mat) {

        // Allocate the memory we need
        size_t *indices = new size_t[n_total * k];
        T *distances = new T[n_total * k];

        #pragma omp parallel for default(none) shared(indices, distances)
        for(size_t i = 0; i < n_total; i++) {

//             Find nearest neighbors
            size_t *__restrict idxi = indices + i * k;
            T *__restrict dist = distances + i * k;

            rb_tree->search(X[i], k, true, false, indices + i * k, distances + i * k);
        }

        for(size_t i = 0; i < n_total; i++){
            for(size_t j = 0; j < k; j++){
                if(indices[i*k + j] > n_offset){

                }

        }

            // Initialize some variables for binary search
//            bool found = false;
//            T beta = 1.0;
//            T min_beta = -std::numeric_limits<T>::min();
//            T max_beta =  std::numeric_limits<T>::max();
//            T tol = 1e-5;

            // Iterate until we found a good perplexity
//            int iter = 0; T sum_P;
//            while(!found && iter < 200) {
//
//                // Compute Gaussian kernel row
//                for(int m = 0; m < k; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);
//
//                // Compute entropy of current row
//                sum_P = DBL_MIN;
//                for(int m = 0; m < K; m++) sum_P += cur_P[m];
//                double H = .0;
//                for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
//                H = (H / sum_P) + log(sum_P);
//
//                // Evaluate whether the entropy is within the tolerance level
//                double Hdiff = H - log(perplexity);
//                if(Hdiff < tol && -Hdiff < tol) {
//                    found = true;
//                }
//                else {
//                    if(Hdiff > 0) {
//                        min_beta = beta;
//                        if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
//                            beta *= 2.0;
//                        else
//                            beta = (beta + max_beta) / 2.0;
//                    }
//                    else {
//                        max_beta = beta;
//                        if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
//                            beta /= 2.0;
//                        else
//                            beta = (beta + min_beta) / 2.0;
//                    }
//                }
//
//                // Update iteration counter
//                iter++;
//            }
//
//            // Row-normalize current row of P and store in matrix
//            for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
//            for(unsigned int m = 0; m < K; m++) {
//                col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
//                val_P[row_P[n] + m] = cur_P[m];
//            }

        // Clean up memory
//        obj_X.clear();
//        free(cur_P);
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
