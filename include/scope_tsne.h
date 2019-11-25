
#ifndef SCOPE_TSNE_H
#define SCOPE_TSNE_H


#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "splittree.h"
#include "tree.hpp"

#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif

namespace tsne{

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class TSNE
{
public:
    void run(double* X, int N, int D, double* Y,
             int no_dims = 2, double perplexity = 30, double theta = .5,
             int num_threads = 1, int max_iter = 1000, int random_state = 0,
             int init_from_Y = 0, int verbose = 0,
             double early_exaggeration = 12, double learning_rate = 200,
             double *final_error = NULL);
    void symmetrizeMatrix(int** row_P, int** col_P, double** val_P, int N);
private:
    double computeGradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, bool eval_error);
    double evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, int no_dims, double theta);
    void zeroMean(double* X, int N, int D, int init_from_y=0);
    void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, int verbose);
    double randn();
};


}

#endif