//
// Created by andy on 2019-11-25.
//

#ifndef TSNE_TSNE_HPP
#define TSNE_TSNE_HPP

#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>
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
                    size_t tmpid = iter->first;
                    if(tmpid < 2000){
                        int f = 1;
                    }
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
    void makeSymmtric(size_t n_offset, std::unordered_map<size_t, T> *lk, size_t **row_P, size_t **col_P, T **val_P);
    void makeSymmtric(DynamicMatrix *mat);
    void computeGradient(size_t run_n, size_t offset, T theta, size_t *row_P, size_t *col_P, T *val_P, T *dY);
    void computeGradient(size_t offset, T theta, Matrix *mat, T *dY);
    void searchGaussianPerplexity(size_t k, T perplexity, T *dist, T *cur_P);
    void computeGaussianPerplexity(size_t offset, size_t k, T perplexity, size_t **row_P, size_t **col_P, T **val_P);

    void computeGaussianPerplexity(size_t offset, size_t k, T perplexity, DynamicMatrix *mat);
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
            delete [](*iter);
        }
        for(auto iter = Y.begin(); iter != Y.end(); iter++){
            delete [](*iter);
        }
    }

    void insertItems(size_t n, T *x, T *y);

    void testGaussianPerplexity(size_t n_offset, size_t k, T perplexity, size_t **row_P, size_t **col_P, T **val_P);

    void testGradient(size_t offset, T perplexity, T theta, T *dY);

    void run(size_t n, T *x, T* y, T perplexity, T theta, bool exact,
             bool partial, int max_iter, int stop_lying_iter, int mom_switch_iter);

    std::vector<T*>& getX(){ return X;}
    std::vector<T*>& getY(){ return Y;}

};

}

#endif //TSNE_TSNE_HPP
