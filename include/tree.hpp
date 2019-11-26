#ifndef TSNE_TREE_H
#define TSNE_TREE_H

#include <vector>
#include <limits>
#include <queue>
#include <algorithm>
#include <distance.hpp>

namespace tsne{


template<typename T>
class VpTree{

private:

    ushort dim;
    size_t n_total;
    Distance<T> *_distance;

protected:

    struct DataPoint{
        size_t index;
        T *val;

        DataPoint():index(0), val(nullptr){}

        DataPoint(size_t i, ushort dim, const T *v): index(i){
            val = new T[dim];
            memcpy(val, v, sizeof(T)*dim);
        }

        DataPoint(DataPoint &&other) noexcept: DataPoint(){
            *this = std::move(other);
        }

        DataPoint& operator =(DataPoint &&other) noexcept {

            if(this != &other){
                delete val;
                index = other.index;
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
            node->radius = (_distance->get(pts[lower].val, pts[median].val));
            node->pt = std::move(pts[lower]);

            // Recursively build tree
            node->left = insert(pts, lower + 1, median);
            node->right = insert(pts, median, upper);
        }
        else{
            node->pt = std::move(pts[lower]);
        }
        return node;
    }

    void insert(DataPoint &pt, Node *node){
        T dist = _distance->get(pt.val, node->pt.val);
        if(dist < node->radius){
            if(node->left)
            {
                insert(pt, node->left);
            }
            else{
                node->left = new Node();
                node->left->pt = std::move(pt);
            }
        }
        else{
            if(node->right){
                insert(pt, node->right);
            }
            else{
                node->right = new Node();
                node->right->pt = std::move(pt);
            }
        }
        if(node->radius == 0){
            node->radius = dist;
        }
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


public:

    VpTree(): dim(0), n_total(0), _distance(nullptr),  _root(nullptr){}

    explicit VpTree(ushort dim): dim(dim), _root(nullptr){
        _distance = new EuclideanDistance<T>(dim);
    }

    VpTree(size_t n, ushort dim, const T *x): VpTree(dim){
        this->n_total = n;
        _distance = new EuclideanDistance<T>(dim);
        auto *pts = new DataPoint[n];
        for(size_t i = 0; i < n; i++){
            pts[i] = std::move(DataPoint(i, dim, x + i*dim));
        }
        _root = insert(pts, 0, n);
        delete []pts;
    }

    ~VpTree(){
        delete _root;
        delete _distance;
    }

    void insert(const T *inp, const size_t index){
        DataPoint pt(index, dim, inp);
        insert(pt, _root);
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


};


template <typename T>
class BarnesHutTree{

    protected:

    int dim;
    int n_splits;

    struct Cell {

        size_t cum_size;
        bool is_leaf;
        T *center;
        T *width;
        T *center_of_mass;
	    std::vector<Cell*> children;

        Cell():cum_size(0),is_leaf(true), center(nullptr),width(nullptr),center_of_mass(nullptr){};

        explicit Cell(int dim):Cell(){
            center = (T*)calloc(dim, sizeof(T));
            width = new T[dim];
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
        for (int i = 0; i< dim; ++i) {
            if (std::abs(cell->center[i] - point[i]) > cell->width[i]) {
                return false;
            }
        }
        return true;
    }

    void subdivide(Cell *cell){
        T *sub_centers = new T[2 * dim];
        int *bits = new int[dim];
        for(int i = 0; i < dim; ++i) {
            sub_centers[i*2]     = cell->center[i] - .5 * cell->width[i];
            sub_centers[i*2 + 1] = cell->center[i] + .5 * cell->width[i];
        }
        cell->children.reserve(n_splits);
        
        for(int i = 0; i < n_splits; i++){
            getBits(i, dim, bits);
            Cell *child = new Cell(dim);
            // fill the means and width
            for (int d = 0; d < dim; d++) {
                child->center[d] = sub_centers[d*2 + bits[d]];
                child->width[d] = .5*cell->width[d];
            }
            cell->children.push_back(child);
        }
        delete []sub_centers; 
        delete []bits;

        // Move existing points to correct children
        for(int i = 0; i < n_splits; i++){
            if(insert(cell->center_of_mass, cell->children[i])){ break;}
        }
        cell->is_leaf = false;
    }

    bool isDuplicate(const T *a_point, const T *b_point) const {
        bool duplicate = true;
        for (int d = 0; d < dim; d++) {
            if (a_point[d] != b_point[d]) { duplicate = false; break; }
        }
        return duplicate;
    }

    bool insert(T *point, Cell *cell){

        if(!containsPoint(cell, point)){
            return false;
        }

        cell->cum_size++;

        if(cell->is_leaf){
            if(!cell->center_of_mass){
                cell->center_of_mass = new T[dim];
                memcpy(cell->center_of_mass, point, sizeof(T)*dim);
                return true;
            }
            else if(isDuplicate(cell->center_of_mass, point)){
                return true;
            }
            else{
                subdivide(cell);
            }

        }

        // update center mass of non-leaf cell
        T mult1 = (T) (cell->cum_size - 1) / (T) cell->cum_size;
        T mult2 = 1.0 / (T) cell->cum_size;
        for (int d = 0; d < dim; d++) {
            cell->center_of_mass[d] = cell->center_of_mass[d] * mult1 + mult2 * point[d];
        }

        for (int i = 0; i < n_splits; ++i) {
            if (insert(point, cell->children[i])) {
                return true;
            }
        }

        // Find out where the point can be inserted
        // Otherwise, the point cannot be inserted (this should never happen)
        // printf("%s\n", "No no, this should not happen");
        return false;

    }

	void computeNonEdgeForces(const T *point, const Cell *cell, const T theta, T* neg_f, T &sum_Q) const {

        if (cell->cum_size == 0 || (cell->is_leaf && isDuplicate(cell->center_of_mass, point))) {
            return;
        }
        // Compute distance between point and center-of-mass
        T D = .0;

        for (int d = 0; d < dim; d++) {
            double t  = point[d] - cell->center_of_mass[d];
            D += t * t;
        }

        // Check whether we can use this node as a "summary"
        T m = std::numeric_limits<T>::lowest();
        for (int i = 0; i < dim; ++i) {
            m = std::max(m, cell->width[i]);
        }
        if (cell->is_leaf || m / sqrt(D) < theta) {

            // Compute and add t-SNE force between point and current node
            T Q = 1.0 / (1.0 + D);
            sum_Q += cell->cum_size * Q;
            T mult = cell->cum_size * Q * Q;
            for (int d = 0; d < dim; d++) {
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
    BarnesHutTree():dim(0), n_splits(0), _root(nullptr){};
    BarnesHutTree(int n, int dim, T *data):dim(dim){
        n_splits = 1 << dim;
        _root = new Cell(n, dim, data);
        for(int i = 0; i < n; i++){
            insert(data + i*dim, _root);
        }
    };
    ~BarnesHutTree(){
        delete _root;
    }

    bool insert(T *d){
        insert(d, _root);
    }

    void computeNonEdgeForces(const T *point, const T theta,  T *neg_f, T &sum_Q) const{
        computeNonEdgeForces(point, _root, theta, neg_f, sum_Q);
    }

};

}


#endif