#ifndef TSNE_TREE_H
#define TSNE_TREE_H

#include <vector>
#include <limits>
#include <queue>
#include <algorithm>

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


template<typename T>
class VpTree{

public:

    VpTree(): dim(0), n_total(0), _distance(nullptr),  _root(nullptr){}

    explicit VpTree(ushort dim): dim(dim), n_total(0), _root(nullptr){
        _distance = new EuclideanDistance<T>(dim);
    }

    VpTree(size_t n, ushort dim, const T *x): VpTree(dim){
        n_total = n;
        _distance = new EuclideanDistance<T>(dim);
        auto *pts = new DataPoint[n];
        for(int i = 0; i < n; i++){
            pts[i] = std::move(DataPoint(i, dim, x + i*dim));
        }
        _root = insert(pts, 0, n);
        delete []pts;
    }

    ~VpTree(){
        delete _root;
        delete _distance;
    }



    void search(const T *target, int k, std::vector<size_t> &results, std::vector<T> &distances){

        std::priority_queue<HeapItem> heap;

        // Variable that tracks the distance to the farthest point in our results
        T tau = std::numeric_limits<T>::max();

        // Perform the searcg
        search(_root, target, k, heap, tau);

        // Gather final results
        results.clear(); distances.clear();
        while (!heap.empty()) {
            results.push_back(heap.top().index);
            distances.push_back(heap.top().dist);
            heap.pop();
        }

        // Results are in reverse order
        std::reverse(results.begin(), results.end());
        std::reverse(distances.begin(), distances.end());

    }

protected:

    Distance<T> *_distance;

    struct DataPoint{
        size_t index;
        ushort dim;
        T *val;

        DataPoint():index(0), dim(0), val(nullptr){}

        DataPoint(size_t i, ushort dim, const T *v): index(i), dim(dim){
            val = new T[dim];
            memcpy(val, v, sizeof(T)*dim);
        }

        DataPoint(const DataPoint &other):index(0), dim(0), val(nullptr){
            *this = other;
        }

        DataPoint& operator =(const DataPoint &other){
            if(this != &other){
                T *tmp = new T[other.dim];
                memcpy(tmp, other.val, sizeof(T)*other.dim);
                delete val;
                val = tmp;
                index = other.index;
                dim = other.dim;
            }
            return *this;
        }

        DataPoint(DataPoint &&other) noexcept: index(0), dim(0), val(nullptr){
            *this = std::move(other);
        }

        DataPoint& operator =(DataPoint &&other) noexcept {

            if(this != &other){
                delete val;
                index = other.index;
                dim = other.dim;
                val = other.val;
                other.val = nullptr;
            }
            return *this;
        }

        ~DataPoint(){
            delete val;
        }
    };

    struct Node{
        T radius;
        DataPoint pt;
        Node *left;
        Node *right;

        Node(): radius(0),left(nullptr),right(nullptr){}

        Node(DataPoint &pt, T r): radius(r), pt(pt), left(nullptr),right(nullptr){}

        ~Node(){
            delete left;
            delete right;
        }
    } *_root;

    struct HeapItem {
        HeapItem( size_t index, T dist) :
                index(index), dist(dist) {}
        size_t index;
        T dist;
        bool operator<(const HeapItem& o) const {
            return dist < o.dist;
        }
    };

    struct DistanceComparator
    {
        const DataPoint &pt;
        Distance<T> *dist;

        DistanceComparator(const DataPoint &pt, Distance<T> *dist) : pt(pt), dist(dist) {}

        bool operator()(const DataPoint &a, const DataPoint &b) {

            return dist->get(pt.val, a.val) < dist->get(pt.val, b.val);
        }
    };

    Node* insert(DataPoint *pts, size_t lower, size_t upper){
        if (upper == lower) return nullptr;

        Node *node = new Node();
        node->pt = pts[lower];

        if(upper - lower > 1){
            int i = (int) ((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
            std::swap(pts[lower], pts[i]);

            // Partition around the median distance
            int median = (upper + lower) / 2;
            std::nth_element(pts + lower + 1,
                             pts + median,
                             pts + upper,
                             DistanceComparator(pts[lower], _distance));

            // Threshold of the new node will be the distance to the median
//            const T *lower_pt = _data + indexes[lower]*dim;
//            const T *median_pt = _data + indexes[median]*dim;
            node->pt = pts[lower];
            node->radius = (_distance->get(pts[lower].val, pts[median].val));

            // Recursively build tree
            node->left = insert(pts, lower + 1, median);
            node->right = insert(pts, median, upper);

        }
        return node;
    }

    void search(Node *node, const T *target, unsigned int k, std::priority_queue<HeapItem>& heap, T& tau)
    {
        if (node == NULL) return;    // indicates that we're done here

        // Compute distance between target and current node
        T dist = _distance->get(node->pt.val, target);

        // If current node within radius tau
        if (dist < tau) {
            if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
            heap.push(HeapItem(node->pt.index, dist));           // add current node to result list
            if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
        }

        // Return if we arrived at a leaf
        if (node->left == NULL && node->right == NULL) {
            return;
        }

        // if node is laied inside or intersect with the search radius
        if(dist <= tau || dist - node->radius <= tau){
            search(node->left, target, k, heap, tau);
            search(node->right, target, k, heap, tau);
        }
            // if node is laied outside of the search radius
        else if(dist - node->radius > tau){
            search(node->right, target, k, heap, tau);
        }
    }


private:
    ushort dim;
    size_t n_total;

};


template <typename T>
class BarnesHutTree{

    protected:
    int n_dims;
    int n_splits;
    T *data;
    struct Cell {
        int index;
        int cum_size;
        bool is_leaf;
        T *center;
        T *width;
        T *center_of_mass;
	    std::vector<Cell*> children;
        Cell():index(-1),cum_size(0),is_leaf(true), center(nullptr),width(nullptr),center_of_mass(nullptr){};
        explicit Cell(int dims):Cell(){
            center = (T*)calloc(dims, sizeof(T));
            center_of_mass = (T*)calloc(dims, sizeof(T));
            width = new T[dims]; 
        }
        Cell(int n, int dims, T *inp):Cell(dims){
            std::vector<T> min_Y(dims, std::numeric_limits<T>::max());
            std::vector<T> max_Y(dims, std::numeric_limits<T>::lowest());

            for(int i = 0; i < n; i++){
                for(int j = 0; j < dims; j++){
                    center[j] += inp[j + i*dims];
                    min_Y[j] = std::min<T>(min_Y[j], inp[j + i*dims]);
                    max_Y[j] = std::max<T>(max_Y[j], inp[j + i*dims]);
                }
            }
            for (int d = 0; d < dims; d++) {
                center[d] /= (T) n;
                width[d] = std::max<T>(max_Y[d] - center[d], center[d] - min_Y[d]) + 1e-5;    
            }
        };

        ~Cell(){
            delete []center;
            delete []width;
            delete []center_of_mass;
            for(auto iter = children.begin(); iter != children.end(); iter++){
                delete (*iter);
            }
        }

        

    } *_root;

    inline static void getBits(int n, int bitswanted, int *bits){
        for(int k=0; k<bitswanted; k++) {
            int mask =  1 << k;
            int masked_n = n & mask;
            int thebit = masked_n >> k;
            bits[k] = thebit;
        }
    }

    inline bool containsPoint(Cell *cell, T *point) const
    {   
        for (int i = 0; i< n_dims; ++i) {
            if (std::abs(cell->center[i] - point[i]) > cell->width[i]) {
                return false;
            }
        }
        return true;
    }

    void subdivide(Cell *cell){
        T *sub_centers = new T[2 * n_dims];
        int *bits = new int[n_dims];
        for(int i = 0; i < n_dims; ++i) {
            sub_centers[i*2]     = cell->center[i] - .5 * cell->width[i];
            sub_centers[i*2 + 1] = cell->center[i] + .5 * cell->width[i];
        }
        cell->children.reserve(n_splits);
        
        for(int i = 0; i < n_splits; i++){
            getBits(i, n_dims, bits);
            Cell *child = new Cell(n_dims);
            // fill the means and width
            for (int d = 0; d < n_dims; d++) {
                child->center[d] = sub_centers[d*2 + bits[d]];
                child->width[d] = .5*cell->width[d];
            }
            cell->children.push_back(child);
        }
        delete []sub_centers; 
        delete []bits;

        // Move existing points to correct children
        for(int i = 0; i < n_splits; i++){
            insert(cell->index, cell->children[i]);
        }
        cell->index = -1;
        cell->is_leaf = false;
    }

    bool isDuplicate(const T *point, const Cell *cell) const {
        bool duplicate = true;
        for (int d = 0; d < n_dims; d++) {
            if (point[d] != data[cell->index * n_dims + d]) { duplicate = false; break; }
        }
        return duplicate;
    }

    bool insert(int idx, Cell *cell){

        T *point = data + idx*n_dims;
        if(!containsPoint(cell, point)){
            return false;
        }

        cell->cum_size++;

        T mult1 = (T) (cell->cum_size - 1) / (T) cell->cum_size;
        T mult2 = 1.0 / (T) cell->cum_size;
        for (int d = 0; d < n_dims; d++) {
            cell->center_of_mass[d] = cell->center_of_mass[d] * mult1 + mult2 * point[d];
        }

        // If there is space in this quad tree and it is a leaf, add the object here
        if (cell->is_leaf && cell->index == -1) {
            cell->index = idx;
            return true;
        }

        // Don't add duplicates for now (this is not very nice)
        
        if (isDuplicate(point, cell)) {
            return true;
        }

        // Otherwise, we need to subdivide the current cell
        if (cell->is_leaf) {
            subdivide(cell);

            for (int i = 0; i < n_splits; ++i) {
                if (insert(idx, cell->children[i])) {
                    return true;
                }
            }
        }

        // Find out where the point can be inserted
        // Otherwise, the point cannot be inserted (this should never happen)
        // printf("%s\n", "No no, this should not happen");
        return false;

    }

	void computeNonEdgeForces(const T *point, const Cell *cell, const T theta, T* neg_f, T* sum_Q) const {

        if (cell->cum_size == 0 || (cell->is_leaf && cell->index != -1 && isDuplicate(point, cell))) {
            return;
        }
        // Compute distance between point and center-of-mass
        T D = .0;

        for (int d = 0; d < n_dims; d++) {
            double t  = point[d] - cell->center_of_mass[d];
            D += t * t;
        }

        // Check whether we can use this node as a "summary"
        T m = std::numeric_limits<T>::lowest();
        for (int i = 0; i < n_dims; ++i) {
            m = std::max(m, cell->width[i]);
        }
        if (cell->is_leaf || m / sqrt(D) < theta) {

            // Compute and add t-SNE force between point and current node
            T Q = 1.0 / (1.0 + D);
            *sum_Q += cell->cum_size * Q;
            T mult = cell->cum_size * Q * Q;
            for (int d = 0; d < n_dims; d++) {
                neg_f[d] += mult * (point[d] - cell->center_of_mass[d]);
            }
        }
        else {
            // Recursively apply Barnes-Hut to children
            for (int i = 0; i < n_splits; ++i) {
                computeNonEdgeForces(point, cell->children[i], theta, neg_f, sum_Q);
            }
        }
    }

    public:
    BarnesHutTree() = default;
    BarnesHutTree(int n, int n_dims, T *data):n_dims(n_dims), data(data){
        n_splits = 1 << n_dims;
        _root = new Cell(n, n_dims, data);
        for(int i = 0; i < n; i++){
            insert(i, _root);
        }
    };
    ~BarnesHutTree(){
        delete _root;
    }

    void appendN(int n, T *d){
        data = d;
        for(int i = 0; i < n; i++){
            insert(_root->cum_size + i, _root);
        }
    }

    void computeNonEdgeForces(const T *point, const T theta,  T *neg_f, T *sum_Q) const{
        computeNonEdgeForces(point, _root, theta, neg_f, sum_Q);
    }

};

}


#endif