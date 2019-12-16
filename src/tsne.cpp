//
// Created by andy on 2019-11-25.
//

#include "tsne.hpp"


#ifdef _OPENMP
#include <omp.h>
#endif


namespace tsne{

    static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

    static inline void recordVarPoints(size_t k, size_t th, size_t *idxi, std::vector<std::pair<size_t, size_t>> &ret){
        for(size_t i = 0; i < k; i++){
            if(idxi[i] >= th){
                ret.emplace_back(idxi[i], i);
            }
        }
    }

    template<typename T>
    static double randn() {
        T x, y, radius;
        do {
            x = 2 * (rand() / ((T) RAND_MAX + 1)) - 1;
            y = 2 * (rand() / ((T) RAND_MAX + 1)) - 1;
            radius = (x * x) + (y * y);
        } while((radius >= 1.0) || (radius == 0.0));
        radius = sqrt(-2 * log(radius) / radius);
        x *= radius;
        return x;
    }

    template<typename T>
    static inline T maxValue(ushort dim, std::vector<T*> &v, T *mean){
        T ret = .0;

        for(size_t i = 0; i < v.size(); i++){
            for(ushort d = 0; d < dim; d++){
                T m = fabs(v[i][d] - mean[d]);
                if(m > ret) ret = m;
            }
        }
        return ret;
    }



    template<typename T>
    TSNE<T>::TSNE():verbose(false), x_dim(0), y_dim(0), n_total(0), bh_tree(nullptr), rb_tree(nullptr){}

    template<typename T>
    TSNE<T>::TSNE(ushort x_dim, ushort y_dim, bool verbose): verbose(verbose), x_dim(x_dim), y_dim(y_dim), n_total(0){
        bh_tree = new BarnesHutTree<T>(y_dim);
        rb_tree = new RedBlackTree<size_t, T>(x_dim);
    }

    template<typename T>
    TSNE<T>::TSNE(size_t n, ushort x_dim, ushort y_dim, T *x, T *y, bool verbose):TSNE(x_dim, y_dim, verbose) {
        insertItems(n, x, y);
    }

    template<typename T>
    TSNE<T>::~TSNE() {
        delete rb_tree;
        delete bh_tree;
        for(auto iter = X.begin(); iter != X.end(); iter++){
            delete [](*iter);
        }
        for(auto iter = Y.begin(); iter != Y.end(); iter++){
            delete [](*iter);
        }
    }

    template<typename T>
    void TSNE<T>::insertX(size_t n, T *x) {
        for(size_t i = 0; i < n; i++){
            T* x_ = new T[x_dim];
            memcpy(x_, x + i * x_dim, x_dim*sizeof(T));
            X.push_back(x_);
            rb_tree->insert(1, &n_total, &x_);
            n_total++;
        }
    }

    template<typename T>
    void TSNE<T>::insertY(size_t n, T *y){
        for(size_t i = 0; i < n; i++){
            T* y_ = new T[y_dim];
            memcpy(y_, y + i * y_dim, y_dim*sizeof(T));
            Y.push_back(y_);
            bh_tree->insert(1, &y_);
        }
    }

    template<typename T>
    void TSNE<T>::insertRandomY(size_t n) {
        for(size_t i = 0; i < n; i++){
            T *y_ = new T[y_dim];
            for(ushort d = 0; d < y_dim; d++){
                y_[d] = randn<T>();
            }
            Y.push_back(y_);
        }
    }

    template<typename T>
    void TSNE<T>::insertItems(size_t n, T *x, T *y){
        insertX(n, x);
        insertY(n, y);
    }

    template<typename T>
    void TSNE<T>::computeGaussianPerplexity(size_t k, T perplexity, tsne::TSNE<T>::DynamicMatrix *dynamic_val_P) {

        std::unique_ptr<size_t[]> indices = std::unique_ptr<size_t[]>(new size_t[n_total * k]);
        std::unique_ptr<T[]> distances = std::unique_ptr<T[]>(new T[n_total * k]);
        T max_v = maxValue(x_dim, X, rb_tree->treeMean());

        size_t bh_size = bh_tree->treeTotal();

#pragma omp parallel for default(none) shared(k, perplexity, bh_size, max_v, indices, distances, dynamic_val_P)
        for (size_t i = 0; i < n_total; i++) {

//          Find nearest neighbors
            size_t *idxi = indices.get() + i * k;
            T *dist = distances.get() + i * k;

            rb_tree->search(X[i], k, false, idxi, dist);


            std::vector<std::pair<size_t, size_t>> var_points;
            recordVarPoints(k, bh_size, idxi, var_points);

            if (!var_points.empty() || i >= bh_size) {

                for (size_t m = 0; m < k; m++) {
                    dist[m] /= max_v;
                }
                std::unique_ptr<T[]> cur_P = std::unique_ptr<T[]>(new T[k]);
                searchGaussianPerplexity(k, perplexity, dist, cur_P.get());

#pragma omp critical
                {
                    if (i >= bh_size) {
                        for (size_t m = 0; m < k; m++) dynamic_val_P->add(i - bh_size, idxi[m], cur_P[m]);
                    }
                    for (auto &var_point : var_points) {
                        dynamic_val_P->add(var_point.first - bh_size, i, cur_P[var_point.second]);
                    }
                }
            }
        }
        makeSymmtric(dynamic_val_P);
    }


    template<typename T>
    T TSNE<T>::computeGradient(T theta, T sum_Q, tsne::TSNE<T>::Matrix *val_P, T *dY, bool eval) {

        // Construct space-partitioning tree on current map
        size_t bh_size = bh_tree->treeTotal();

        std::unique_ptr<tsne::BarnesHutTree<T>> sub_bh_tree =
                std::unique_ptr<tsne::BarnesHutTree<T>>(new tsne::BarnesHutTree<T>(val_P->n_row, y_dim, Y.data() + bh_size));

        // Compute all terms required for t-SNE gradient
        std::unique_ptr<T[]> pos_f = std::unique_ptr<T[]>(new T[val_P->n_row * y_dim]());
        std::unique_ptr<T[]> neg_f = std::unique_ptr<T[]>(new T[n_total * y_dim]());

        T val_P_sum = .0;
        T C = .0;

        // Loop over all edges in the graph
#pragma omp parallel for reduction(+:sum_Q, val_P_sum, C)
        for(size_t i = 0; i < n_total; i++) {
            T *neg = neg_f.get() + i*y_dim;
            T sum_q = .0;
            T i_p_sum = .0;
            T i_c = .0;
            if(i >= bh_size){
                size_t row_i = i - bh_size;
                T *pos = pos_f.get() + row_i*y_dim;
                computeEdgeForces(row_i, val_P, pos,i_p_sum,i_c, eval);
                bh_tree->computeNonEdgeForces(Y[i], theta, neg, sum_q);
            }
            sub_bh_tree->computeNonEdgeForces(Y[i], theta, neg, sum_q);
            sum_Q += sum_q;
            val_P_sum += i_p_sum;
            C += i_c;
        }
        // Compute final t-SNE gradient
        for(size_t i = 0; i < val_P->n_row * y_dim; i++) {
            dY[i] = pos_f[i] - (neg_f[i + bh_size*y_dim] / sum_Q);
        }
        C += val_P_sum * log(sum_Q);
        return C;
    }

    template<typename T>
    void TSNE<T>::computeEdgeForces(size_t i, tsne::TSNE<T>::Matrix *val_P, T *pos, T &i_p_sum, T &i_c, bool eval) {
        for(size_t col_i = 0; col_i < val_P->getRowSize(i); col_i++){
            size_t idx = val_P->getIndex(i, col_i);
            T D = 1.0;
            for(size_t d = 0; d < y_dim; d++){
                T t = Y[bh_tree->treeTotal() + i][d] - Y[idx][d];
                D += t * t;
            }
            T inp_val_P = val_P->getValue(i, col_i);
            if(eval){
                i_p_sum += inp_val_P;
                i_c += inp_val_P * log((inp_val_P + FLT_MIN) / ((1.0 / D) + FLT_MIN));
            }
            D = inp_val_P / D;
            for(size_t d = 0; d < y_dim; d++) pos[d] += D * (Y[bh_tree->treeTotal() + i][d] - Y[idx][d]);
        }

    }


    template<typename T>
    T TSNE<T>::computeSumQ(T theta) {
        T sum_Q = .0;
        size_t bh_size = bh_tree->treeTotal();
        if(bh_size > 0){
            std::unique_ptr<T[]> neg_f = std::unique_ptr<T[]>(new T[bh_size * y_dim]());
            T sum_q = .0;
#pragma omp parallel for reduction(+:sum_Q)
            for(size_t i = 0; i < bh_size; i++){
                bh_tree->computeNonEdgeForces(Y[i], theta, neg_f.get() + i*y_dim, sum_q);
                sum_Q += sum_q;
            }
        }
        return sum_Q;
    }

    template<typename T>
    void TSNE<T>::updateGradient(size_t n, T eta, T momentum,  T *dY, T *uY, T *gains, T **y) {

        size_t idx;
        for(size_t i = 0; i < n; i++){
            for(ushort d = 0; d < y_dim; d++){
                idx = i*y_dim + d;
                gains[idx] = (sign(dY[idx]) != sign(uY[idx])) ? (gains[idx] + .2) : (gains[idx] * .8 + 0.1);
                uY[idx] = momentum * uY[idx] - eta * gains[idx] * dY[idx];
                y[i][d] = y[i][d] + uY[idx];
            }
        }
    }

    template<typename T>
    void TSNE<T>::zeroMean(size_t n, T **y) {
        if(n == 1) return;
        size_t i;
        ushort d;
        std::unique_ptr<T[]> mean_ = std::unique_ptr<T[]>(new T[y_dim]());
        for(i = 0; i < n; i++){
            for(d = 0; d < y_dim; d++){
                mean_[d] += y[i][d];
            }
        }
        for(d = 0; d < y_dim; d++) mean_[d] /= n;

        for(i = 0; i < n; i++){
            for(d = 0; d < y_dim; d++){
                y[i][d] -= mean_[d];
            }
        }
    }

    template<typename T>
    void TSNE<T>::runTraining(size_t n, T perplexity, T theta,
                              int max_iter, int stop_lying_iter, int mom_switch_iter, T *ret) {
        if(n_total < 4){
            if (verbose)
                fprintf(stdout, "Dataset Is Too Small(%d)...\n", n_total);
            return;
        }
        else if (n_total - 1 < 3 * perplexity) {
            perplexity = T(n_total - 1) / 3.;
            if (verbose)
                fprintf(stdout, "Perplexity too large for the number of data points! Adjusting To %f ...\n", perplexity);
        }
        float total_time = .0;
        time_t start, end;
        T momentum = .5, final_momentum = .8;
        T eta = 200.0;
        size_t bh_size = bh_tree->treeTotal();
        if (verbose)
            fprintf(stdout, "Using no_dims = %d, perplexity = %f, and theta = %f\n", y_dim, perplexity, theta);
        std::unique_ptr<T[]> dY(new T[n * y_dim]);
        std::unique_ptr<T[]> uY(new T[n * y_dim]());
        std::unique_ptr<T[]> gains(new T[n * y_dim]);
        std::fill(gains.get(), gains.get() + (n * y_dim), 1.0);

        std::unique_ptr<Matrix> val_P;

        start = time(0);

        size_t k = (int) (3 * perplexity);
        DynamicMatrix dynamic_val_P(n);
        computeGaussianPerplexity(k, perplexity, &dynamic_val_P);
        val_P = std::unique_ptr<SparseMatrix>(new SparseMatrix(std::move(dynamic_val_P)));
        T sum_Q = computeSumQ(theta);

        (*val_P) *= 12.0;
        end = time(0);
        if (verbose)
            fprintf(stdout, "Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float)(end - start) , (double) val_P->n / ((double) n_total * (double) n_total));

        start = time(0);
        for(int iter = 0; iter < max_iter; iter++) {

            bool need_eval_error = (verbose && ((iter > 0 && iter % 50 == 0) || (iter == max_iter - 1)));

            T error = computeGradient(theta, sum_Q, val_P.get(), dY.get(), need_eval_error);
            updateGradient(n, eta, momentum, dY.get(), uY.get(), gains.get(), Y.data() + bh_size);
            zeroMean(n, Y.data() + bh_size);

            if(iter == stop_lying_iter) {
                (*val_P) /= 12.0;
            }
            if(iter == mom_switch_iter) momentum = final_momentum;
            // Print out progress
            if (need_eval_error) {
                end = time(0);

                if (iter == 0)
                    fprintf(stdout, "Iteration %d: error is %f\n", iter + 1, error);
                else {
                    total_time += (float) (end - start);
                    fprintf(stdout, "Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter + 1, error, (float) (end - start) );
                }
                start = time(0);
            }
        }

        end = time(0); total_time += (float) (end - start) ;
        if (verbose)
            fprintf(stdout, "Fitting performed in %4.2f seconds.\n", total_time);

        for(size_t i = 0; i < n; i++){
            memcpy(ret + i *y_dim, Y[i + bh_size], y_dim*sizeof(T));
        }

    }

    template<typename T>
    void TSNE<T>::run(T perplexity, T theta, int max_iter, int stop_lying_iter, int mom_switch_iter,
                      T *ret) {
        for(size_t i = 0; i < n_total; i++){
            for(ushort d = 0; d < y_dim; d++){
                Y[i][d] = randn<T>();
            }
        }
        delete bh_tree;
        bh_tree = new BarnesHutTree<T>(y_dim);
        runTraining(n_total, perplexity, theta, max_iter, stop_lying_iter, mom_switch_iter, ret);
        bh_tree->insert(n_total, Y.data());
    }

    template<typename T>
    void TSNE<T>::run(size_t n, T *x, T perplexity, T theta, int max_iter, int stop_lying_iter,
                      int mom_switch_iter, T *ret) {
        insertX(n, x);
        insertRandomY(n);
        runTraining(n, perplexity, theta, max_iter, stop_lying_iter, mom_switch_iter, ret);
        bh_tree->insert(n, Y.data() + bh_tree->treeTotal());
    }

    template<typename T>
    void TSNE<T>::makeSymmtric(tsne::TSNE<T>::DynamicMatrix *mat) {
        for(size_t i = 0; i < mat->n_row; i++){
            for(auto iter = mat->rows[i].begin(); iter != mat->rows[i].end(); iter++){
                iter->second /=(2.0 * (T)n_total);
            }
        }
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
            for(size_t m = 0; m < k; m++) cur_P[m] = exp(-beta * dist[m] * dist[m]);

            // Compute entropy of current row
            sum_P = std::numeric_limits<T>::min();
            for(size_t m = 0; m < k; m++) sum_P += cur_P[m];
            double H = .0;
            for(size_t m = 0; m < k; m++) H += beta * (dist[m] * dist[m] * cur_P[m]);
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

    template class TSNE<float>;
    template class TSNE<double>;

}
