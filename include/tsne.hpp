//
// Created by andy on 2019-11-25.
//

#ifndef TSNE_TSNE_HPP
#define TSNE_TSNE_HPP

#include <float.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <tree.hpp>

namespace tsne{

template<typename T>
class TSNE{

    protected:

    bool verbose;
    ushort x_dim;
    ushort y_dim;
    size_t n_total;
    std::vector<T*> X;
    std::vector<T*> Y;
    BarnesHutTree<T> *bh_tree;
    RedBlackTree<size_t, T> *rb_tree;

    struct Matrix{
        size_t n;
        size_t n_row;
        T *val_P;
        Matrix():n(0), n_row(0), val_P(nullptr){}
        explicit Matrix(size_t n, size_t n_row):n(n), n_row(n_row){
            val_P = new T[n];
        }

        Matrix& operator*=(T val){
            for(size_t i = 0; i < n; i++){
                val_P[i]*= val;
            }
            return *this;
        }

        Matrix& operator/=(T val){
            for(size_t i = 0; i < n; i++){
                val_P[i]/= val;
            }
            return *this;
        }

        virtual T getValue(size_t row_i, size_t col_i) = 0;
        virtual size_t getIndex(size_t row_i, size_t col_i) = 0;
        virtual size_t getRowSize(size_t row_i) = 0;
        ~Matrix(){
            delete []val_P;
        }
    };

    struct DynamicMatrix{

        size_t n_row;
        size_t n;
        std::unordered_map<size_t, T> *rows;

        DynamicMatrix(): n(0), n_row(0), rows(nullptr){}
        explicit DynamicMatrix(size_t n_row): DynamicMatrix(){
            this->n_row = n_row;
            rows = new std::unordered_map<size_t, T>[n_row];
        }
        ~DynamicMatrix(){
            delete[] rows;
        }

        void add(size_t row_i, size_t col_i, T val){
            if(rows[row_i].find(col_i) == rows[row_i].end()){
                rows[row_i].emplace(col_i, val);
                n++;
            }
            else{
                rows[row_i][col_i] += val;
            }
        }

    };

    struct StaticMatrix:Matrix{
        size_t n_col;

        StaticMatrix():Matrix(), n_col(0){}
        StaticMatrix(size_t n_row, size_t n_col): Matrix(n_row * n_col, n_row), n_col(n_col){}

        void assign(size_t row_i, size_t col_i, T val){
            (this->val_P + row_i*n_col)[col_i] = val;
        }

        T getValue(size_t row_i, size_t col_i) override{
            return (this->val_P + row_i*n_col)[col_i];
        }

        size_t getIndex(size_t row_i, size_t col_i) override{
            return col_i;
        }

        size_t getRowSize(size_t row_i) override {
            return n_col;
        };

    };

    struct SparseMatrix: Matrix{
        size_t *row_P;
        size_t *col_P;

        SparseMatrix():Matrix(), row_P(nullptr), col_P(nullptr){}
        explicit SparseMatrix(const DynamicMatrix &mat): Matrix(mat.n, mat.n_row){
            row_P = new size_t[this->n_row + 1];
            row_P[0] = 0;
            size_t nn = 0;
            for(size_t i = 0; i < this->n_row; i++){
                nn += mat.rows[i].size();
                row_P[i + 1] = nn;
            }
            col_P = new size_t[nn];
            size_t j = 0;
            for(size_t i = 0; i < this->n_row; i++){
                for(auto iter = mat.rows[i].begin(); iter != mat.rows[i].end(); iter++){
                    col_P[j] = iter->first;
                    this->val_P[j] = iter->second;
                    j++;
                }
            }
        }

        SparseMatrix(DynamicMatrix &&mat):SparseMatrix(mat){
            delete []mat.rows;
            mat.rows = nullptr;
        }

        ~SparseMatrix(){
            delete []row_P;
            delete []col_P;
        }

        T getValue(size_t row_i, size_t col_i) override{
            return this->val_P[row_P[row_i] + col_i];
        }

        size_t getIndex(size_t row_i, size_t col_i) override{
            return col_P[row_P[row_i] + col_i];
        }
        size_t getRowSize(size_t row_i) override{
            return row_P[row_i + 1] - row_P[row_i];
        }
    };

//    void runTraining(size_t offset, size_t max_iters, T theta, T eta, T momentum, Matrix *mat);
    void insertX(size_t n, T *x);
    void insertY(size_t n, T *y);
    void insertRandomY(size_t n);
    void zeroMean(size_t n, T **y);
    void makeSymmtric(DynamicMatrix *mat);
    T computeSumQ(T theta);
    T computeGradient(T theta, T sum_Q, tsne::TSNE<T>::Matrix *val_P, T *dY);
    void updateGradient(size_t n, T eta, T momentum, T *dY, T *uY, T *gains, T **y);
    void computeEdgeForces(size_t i, tsne::TSNE<T>::Matrix *val_P, T *pos, T &i_sum_P, T &C);
    void searchGaussianPerplexity(size_t k, T perplexity, T *dist, T *cur_P);
    void computeGaussianPerplexity(size_t k, T perplexity, tsne::TSNE<T>::DynamicMatrix *dynamic_val_P);
    void runTraining(size_t n, T perplexity, T theta,
            int max_iter, int stop_lying_iter, int mom_switch_iter, T *ret);

    public:

    TSNE();
    TSNE(ushort x_dim, ushort y_dim, bool verbose=false);
    TSNE(size_t n, ushort x_dim, ushort y_dim, T *x, T *y, bool verbose=false);
    ~TSNE();

    void insertItems(size_t n, T *x, T *y);

    void run(T perplexity, T theta, int max_iter, int stop_lying_iter, int mom_switch_iter, T *ret);

    void run(size_t n, T *x, T perplexity, T theta, int max_iter, int stop_lying_iter, int mom_switch_iter, T *ret);

};

}

#endif //TSNE_TSNE_HPP
