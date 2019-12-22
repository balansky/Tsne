#include <cmath>


namespace tsne{

template<typename T>
struct Distance{
    ushort dim;
    
    Distance(): dim(0){}
    explicit Distance(ushort dim):dim(dim){}
    virtual ~Distance() = default;

    virtual T get(const T* t1, const T* t2) const = 0;
    

};

template<typename T>
struct EuclideanDistance:Distance<T> {

    EuclideanDistance(){}
    explicit EuclideanDistance(int dim): Distance<T>(dim){}
    ~EuclideanDistance() = default;

    T get(const T* t1, const T* t2) const{
            T dd = .0;
            for(int d = 0; d < Distance<T>::dim; d++){
                T t = (t1[d] - t2[d]);
                dd += t * t;
            }
            return sqrt(dd);
    }

};
}