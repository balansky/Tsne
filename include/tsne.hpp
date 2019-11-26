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
    VpTree<T> *vp_tree;
    BarnesHutTree<T> *bh_tree;

    struct SymmetricMatrix;

    void computeGradient();
    void computeGaussianPerplexity(int n,double perplexity, double* X, double* P);

    public:

    TSNE():dim(0), vp_tree(nullptr), bh_tree(nullptr){}

    explicit TSNE(const ushort dim): dim(dim), vp_tree(nullptr), bh_tree(nullptr){}

    TSNE(size_t n, ushort dim, T *data);


    void run();



};

}

#endif //TSNE_TSNE_HPP
