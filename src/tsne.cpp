//
// Created by andy on 2019-11-25.
//

#include "tsne.hpp"


#ifdef _OPENMP
#include <omp.h>
#endif


namespace tsne{

    static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

    template<typename T>
    static void mean(size_t begin_n, size_t end_n, ushort dim, std::vector<T*> &v, T *m){
        for(ushort d = 0; d < dim; d++) m[d] = 0;
        for(size_t i = begin_n; i < end_n; i++){
            for(ushort d = 0; d < dim; d++){
                m[d]+= v[i][d];
            }
        }
        for(ushort d = 0; d < dim; d++) m[d] /= (T)(end_n - begin_n);
    }

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
        size_t n_var = n_total + n;
        std::unique_ptr<T[]> offset_mean(new T[y_dim]{.0});
        std::unique_ptr<T[]> y_mean(new T[y_dim]{.0});
        if(partial) {
            mean(0, n_total, y_dim, Y, offset_mean.get());
            offset = n_total;
            n_var = n;
        }
        insertItems(n, x, y);

        // Set learning parameters
        float total_time = .0;
        clock_t start, end;
        T momentum = .5, final_momentum = .8;
        T eta = 200.0;

        std::unique_ptr<T[]> dY(new T[n_var * y_dim]);
        std::unique_ptr<T[]> uY(new T[n_var * y_dim]{.0});
        std::unique_ptr<T[]> gains(new T[n_var * y_dim]);
//        std::fill(uY.get(), uY.get() + (n_var * y_dim), .0);
        std::fill(gains.get(), gains.get() + (n_var * y_dim), 1.0);

        size_t *row_P = nullptr;
        size_t *col_P = nullptr;
        T *val_P = nullptr;
        start = clock();
        if(exact){


        }
        else{
            size_t k = (int) (3 * perplexity);
            computeGaussianPerplexity(offset, k, perplexity, &row_P, &col_P, &val_P);

        }
        end = clock();

        for(size_t i = 0; i < row_P[n_var]; i++) val_P[i] *= 12.0;

        for(int iter = 0; iter < max_iter; iter++) {

            computeGradient(n_var, offset, theta, row_P, col_P, val_P, dY.get());

            // Update gains
            for(size_t i = 0; i < n_var * y_dim; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
            for(size_t i = 0; i < n_var * y_dim; i++) if(gains[i] < .01) gains[i] = .01;

            // Perform gradient update (with momentum and gains)
            for(size_t i = 0; i < n_var * y_dim; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            for(size_t i = 0; i < n_var; i++) {
                for(size_t dd = 0; dd < y_dim; dd++){
                    Y[i + offset][dd] = Y[i + offset][dd] + uY[i*y_dim + dd];
                }
            }
            // Make solution zero-mean
            mean(offset, n_total, y_dim, Y, y_mean.get());
            for(ushort d = 0; d < y_dim; d++) y_mean[d] = ((T)offset / (T)n_total)*offset_mean[d] + ((T)(n_total - offset)/ (T)n_total)*y_mean[d];
            for(size_t i = 0; i < n_var; i++){
                for(size_t dd = 0; dd < y_dim; dd++){
                    Y[i + offset][dd] -= y_mean[dd];
                }
            }

            // Stop lying about the P-values after a while, and switch momentum
            if(iter == stop_lying_iter) {
                for(size_t i = 0; i < row_P[n_var]; i++) val_P[i] /= 12.0;
//                if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
//                else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
            }
            if(iter == mom_switch_iter) momentum = final_momentum;

            // Print out progress
//            if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
//                end = clock();
//                double C = .0;
//                if(exact) C = evaluateError(P, Y, N, no_dims);
//                else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
//                if(iter == 0)
//                    printf("Iteration %d: error is %f\n", iter + 1, C);
//                else {
//                    total_time += (float) (end - start) / CLOCKS_PER_SEC;
//                    printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC);
//                }
//                start = clock();
//            }
        }
        for(size_t i = 0; i < n_var; i++){
            memcpy(y + i *y_dim, Y[i + offset], y_dim*sizeof(T));
        }

        delete []row_P;
        delete []col_P;
        delete []val_P;

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
                (*val_P)[j] = iter->second / (2.0 * (T)n_total);
//                (*val_P)[j] = iter->second / 2.0 ;
                j++;
            }
        }

//        T sum_P = .0;
//        for(int i = 0; i < (*row_P)[n]; i++) sum_P += (*val_P)[i];
//        for(int i = 0; i < (*row_P)[n]; i++) (*val_P)[i] /= sum_P;
//        for(int i = 0; i < (*row_P)[n]; i++) (*val_P)[i] /= (T)n_total;
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
            size_t *idxi = indices + i * k;
            T *dist = distances + i * k;

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
    void TSNE<T>::testGaussianPerplexity(size_t n_offset, size_t k, T perplexity, size_t **row_P, size_t **col_P,
                                         T **val_P) {
        computeGaussianPerplexity(n_offset, k, perplexity, row_P, col_P, val_P);
    }

    template<typename T>
    void TSNE<T>::computeGradient(size_t run_n,  size_t offset, T theta, size_t *row_P, size_t *col_P, T *val_P, T *dY) {

        // Construct space-partitioning tree on current map
        tsne::BarnesHutTree<T> *tree = new tsne::BarnesHutTree<T>(n_total, y_dim, Y.data());
//
//        // Compute all terms required for t-SNE gradient
        T *sum_q = (T*) calloc(run_n, sizeof(T));
        T* pos_f = (T*) calloc(run_n * y_dim, sizeof(T));
        T* neg_f = (T*) calloc(run_n * y_dim, sizeof(T));
        T* buffs = new T[run_n * y_dim];

//        if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        // Loop over all edges in the graph
        #pragma omp parallel for default(none) shared(row_P, col_P, val_P, sum_q, pos_f, neg_f, buffs, sum_q, Y)
        for(size_t n = 0; n < run_n; n++) {
            T *sum = sum_q + n;
            T *neg = neg_f + n*y_dim;
            T *pos = pos_f + n*y_dim;
            for(size_t i = row_P[n]; i < row_P[n + 1]; i++) {

                // Compute pairwise distance and Q-value
                T * buff = buffs + n*y_dim;
                T D = 1.0;
                for(size_t d = 0; d < y_dim; d++) buff[d] = Y[n + offset][d] - Y[col_P[i]][d];
                for(size_t d = 0; d < y_dim; d++) D += buff[d] * buff[d];
                D = val_P[i] / D;

                // Sum positive force
                for(size_t d = 0; d < y_dim; d++) pos[d] += D * buff[d];
            }
            tree->computeNonEdgeForces(Y[n + offset], theta, neg, (*sum));
        }
        T sum_Q = 0.0;
        for(size_t n = 0; n < run_n; n++) sum_Q += sum_q[n];

        // Compute final t-SNE gradient
        for(size_t i = 0; i < run_n * y_dim; i++) {
            dY[i] = pos_f[i] - (neg_f[i] / sum_Q);
        }

        free(sum_q);
        free(pos_f);
        free(neg_f);
        delete []buffs;
        delete tree;

    }


    template<typename T>
    void TSNE<T>::testGradient(size_t offset, T perplexity, T theta, T *dY) {

        size_t *row_P; size_t *col_P; T *val_P;
        computeGaussianPerplexity(offset, int(3*perplexity), perplexity, &row_P, &col_P, &val_P);

        computeGradient(n_total, offset, theta, row_P, col_P, val_P, dY);
        delete row_P;
        delete col_P;
        delete val_P;
    }
    template class TSNE<float>;
    template class TSNE<double>;

}
