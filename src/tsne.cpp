//
// Created by andy on 2019-11-25.
//

#include "tsne.hpp"

namespace tsne{

    template<typename T>
    TSNE<T>::TSNE(size_t n, ushort dim, T *data): TSNE(dim){

    }

    template <typename T>
    T* TSNE<T>::Matrix::get(size_t row_i, size_t col_j) {

    }


    template <typename T>
    void TSNE<T>::Matrix::set(size_t row_i, size_t col_j, T v) {

    }

//    template<typename T>
//    class TSNE<T>::SymmetricMatrix{
//        size_t n;
//        T *val;
//        size_t *offset;
//
//        SymmetricMatrix():n(0), val(nullptr), offset(nullptr){}
//        explicit SymmetricMatrix(size_t n): SymmetricMatrix(){
//            this->n = n;
//        }
//
//        ~SymmetricMatrix(){
//            delete val;
//            delete offset;
//        }
//
//        T* get(size_t row_i, size_t col_j){
//            return val + offset[row_i] + col_j;
//        }
//
//    };



}
