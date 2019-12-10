//
// Created by andy on 2019-11-25.
//

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
            memcpy(x_, x + i * x_dim, x_dim*sizeof(T));
            memcpy(y_, y + i * y_dim, y_dim*sizeof(T));
            X.push_back(x_);
            Y.push_back(y_);
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
//            Matrix mat(run_n, n_total);

        }
        else{
            size_t k = (int) (3 * perplexity);
            size_t *row_P;
            size_t *col_P;
            T *val_P;
            computeGaussianPerplexity(offset, k, perplexity, &row_P, &col_P, &val_P);
            for(size_t i = 0; i < row_P[run_n]; i++) val_P[i] /= (T)n_total;

        }


    }

    template<typename T>
    void TSNE<T>::makeSymmtric(size_t n_offset, std::unordered_map<size_t, T> *lk, size_t **row_P, size_t **col_P, T **val_P) {
        size_t n = n_total - n_offset;
        (*row_P) = new size_t[n + 1];
        (*row_P)[0] = 0;

        size_t nn = 0;
        for(size_t i = 0; i < n; i++){
            nn += lk[i].size();
            (*row_P)[i + 1] = nn;
        }
        (*col_P) = new size_t[nn];
        (*val_P) = new T[nn];

        size_t j = 0;
        for(size_t i = 0; i < n; i++){
            for(auto iter = lk[i].begin(); iter != lk[i].end(); iter++){
                (*col_P)[j] = iter->first;
                (*val_P)[j] = iter->second / 2.0;
                j++;
            }
        }
    }

    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t n, T perplexity, T *x, tsne::TSNE<T>::Matrix &mat) {

    }
    template<typename T>
    void TSNE<T>::searchGaussianPerplexity(size_t k, T perplexity, T *__restrict dist, T *cur_P){
        bool found = false;
        T beta = 1.0;
        T min_beta = -std::numeric_limits<T>::min();
        T max_beta =  std::numeric_limits<T>::max();
        T tol = 1e-5;
        T sum_P;
        int iter = 0;
        while(!found && iter < 200){
            for(int m = 0; m < k; m++) cur_P[m] = exp(-beta * dist[m] * dist[m]);

            // Compute entropy of current row
            sum_P = std::numeric_limits<T>::min();
            for(int m = 0; m < k; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < k; m++) H += beta * (dist[m] * dist[m] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == std::numeric_limits<T>::max() || max_beta == -std::numeric_limits<T>::max())
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -std::numeric_limits<T>::max() || min_beta == std::numeric_limits<T>::max())
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }
        for(size_t m = 0; m < k; m++) cur_P[m] /= sum_P;

    }

    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t n_offset, size_t k, T perplexity, size_t **row_P, size_t **col_P, T **val_P) {

        // Allocate the memory we need
        size_t *indices = new size_t[n_total * k];
        T *distances = new T[n_total * k];

        std::unordered_map<size_t, T> *lk = new std::unordered_map<size_t, T>[n_total - n_offset];

        #pragma omp parallel for default(none) shared(indices, distances)
        for(size_t i = 0; i < n_total; i++) {

//             Find nearest neighbors
            size_t *__restrict idxi = indices + i * k;
            T *__restrict dist = distances + i * k;

            rb_tree->search(X[i], k, false, false, idxi, dist);
            size_t n = 0;
            std::vector<size_t> i_pos;
            std::vector<size_t> j_pos;
            std::vector<T> j_P;
            for(size_t j = 0; j < k; j++){
                if(idxi[j] >= n_offset){
                    i_pos.push_back(idxi[j]);
                    j_pos.push_back(j);
                    n ++;
                }
            }
            if(n > 0 || i >= n_offset){

                T *cur_P = new T[k];
                searchGaussianPerplexity(k, perplexity, dist, cur_P);

                #pragma omp critical
                {
                    if(i >= n_offset){
                        for(size_t m = 0; m < k; m++){
                            size_t pos = i - n_offset;
                            if(lk[pos].find(idxi[m]) == lk[pos].end()){
                                lk[pos].emplace(idxi[m], cur_P[m]);
                            }
                            else{
                                lk[pos][idxi[m]] += cur_P[m];
                            }
                        }
                    }
                    for(size_t jj = 0; jj < n; jj++){
                        size_t pos = i_pos[jj] - n_offset;
                        if(lk[pos].find(i) == lk[pos].end()){
                            lk[pos].emplace(i, cur_P[j_pos[jj]]);
                        }
                        else{
                            lk[pos][i] += cur_P[j_pos[jj]];
                        }
                    }

                }
                delete []cur_P;
            }
        }

        makeSymmtric(n_offset, lk, row_P, col_P, val_P);
        delete []indices;
        delete []distances;
        delete []lk;
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
