#ifndef TSNE_TREE_H
#define TSNE_TREE_H

#include <vector>
#include <limits>
#include <queue>
#include <algorithm>
#include <distance.hpp>

namespace tsne{


template <typename I=size_t, typename T=float>
class RedBlackTree{

private:
    enum Color {RED, BLACK};

    ushort dim;
    size_t n_total;

    struct Node{
        T radius;
        bool color;
        T *v;
        Node *left;
        Node *right;
        Node *parent;
        std::vector<I> indices;

        Node(): radius(0), color(RED), v(nullptr), left(nullptr),right(nullptr), parent(nullptr){}
        Node(I i, T *val):Node(){
            v = val;
            indices.push_back(i);
        }

        ~Node(){
            delete left;
            delete right;
        }
    } *_root;

    T *_mean;

    std::vector<Node*> data;

protected:

    T distance(const T* t1, const T* t2) const{
        T dd = .0;
        for(int d = 0; d < dim; d++){
            T t = (t1[d] - t2[d]);
            dd += t * t;
        }
        return sqrt(dd);
    }

    struct HeapItem {

        HeapItem( I index, T dist) : index(index), dist(dist) {}
        I index;
        T dist;
        bool operator<(const HeapItem& o) const {
            return dist < o.dist;
        }
    };

    void leftRotate(Node *x){
        Node *y = x->right;
        x->right = y->left;
        if(y->left){
            y->left->parent = x;
            x->radius = distance(x->v, y->v);
        }
        else{
            x->radius = 0;
        }
        y->parent = x->parent;
        if(!x->parent){
            _root = y;
        }
        else if(x->parent->left == x){
            x->parent->left = y;
        }
        else{
            x->parent->right = y;
        }
        y->left = x;
        x->parent = y;
        std::swap(x->color, y->color);
    }

    void rightRotate(Node *x){

        Node *y = x->left;
        x->left = y->right;
        if(y->right){
            y->right->parent = x;
        }
        y->parent = x->parent;
        if(!x->parent){
            _root = y;
        }
        else if(x->parent->left == x){
            x->parent->left = y;
        }
        else{
            x->parent->right = y;
        }

        y->right = x;
        x->parent = y;
        y->radius = distance(y->v, x->v);
        std::swap(x->color, y->color);

    }

    Node* getUncle(Node *node){
        Node *parent = node->parent;
        if(parent){
            Node *grandpa = parent->parent;
            if(grandpa){
                if(grandpa->left == parent){
                    return grandpa->right;
                }
                else{
                    return grandpa->left;
                }
            }
            return nullptr;
        }
        return nullptr;
    }

    void balanceTree(Node *node){
        if(node->parent->color == RED){
            Node *uncle = getUncle(node);
            if(!uncle || uncle->color == BLACK){
                //left left case
                if(node->parent->parent->left == node->parent && node->parent->left == node){
                    rightRotate(node->parent->parent);
                }
                //left right case
                else if(node->parent->parent->left == node->parent && node->parent->right == node){
                    leftRotate(node->parent);
                    rightRotate(node->parent);
                }
                // right right case
                else if(node->parent->parent->right == node->parent && node->parent->right == node){
                    leftRotate(node->parent->parent);
                }
                // right left case
                else{
                    rightRotate(node->parent);
                    leftRotate(node->parent);
                }
            }
            else{
                node->parent->color = BLACK;
                uncle->color = BLACK;
                if(node->parent->parent != _root){
                    node->parent->parent->color = RED;
                    balanceTree(node->parent->parent);
                }
            }
        }

    }

    Node* insert(Node *parent, Node *node){
        T dist = distance(parent->v, node->v);
        if(dist == 0){
            parent->indices.push_back(node->indices[0]);
            delete node;
            node = parent;
        }
        else if(dist < parent->radius){
            if(parent->left){
                node = insert(parent->left, node);
            }
            else{
                node->parent = parent;
                parent->left = node;
                balanceTree(node);
            }
        }
        else{
            if(parent->right){
                node = insert(parent->right, node);
            }
            else{
                node->parent = parent;
                parent->right = node;
                if(parent->radius == 0) parent->radius = dist;
                balanceTree(node);
            }
        }
        return node;
    }

    void search(Node *node, const T *target, size_t k, bool dup, std::priority_queue<HeapItem>& heap, T& tau)
    {
        if (!node) return;    // indicates that we're done here

        // Compute distance between target and current node
        T dist = distance(node->v, target);

        // If current node within radius tau
        if (dist < tau) {
            size_t n = dup ? (node->indices.size()):(dist != 0 ? 1 : 0);

            for(size_t i = 0; i < n; i++){
                if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
                heap.push(HeapItem(node->indices[i], dist));           // add current node to result list
                if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
            }
        }

        // Return if we arrived at a leaf
        if (!node->left && !node->right) {
            return;
        }

        // if node is laied inside or intersect with the search radius
        if(dist <= tau || dist - node->radius <= tau){
            search(node->left, target, k, dup, heap, tau);
            search(node->right, target, k, dup,  heap, tau);
        }
            // if node is laied outside of the search radius
        else if(dist - node->radius > tau){
            search(node->right, target, k, dup, heap, tau);
        }
    }

public:
    RedBlackTree(): dim(0), n_total(0), _root(nullptr), _mean(nullptr){}
    explicit RedBlackTree(ushort dim): RedBlackTree(){
        this->dim = dim;
        _mean = new T[dim]();
    }
    RedBlackTree(size_t n, ushort dim, I *ids, T **inps): RedBlackTree(dim){
        insert(n, ids, inps);
    }

    ~RedBlackTree(){
        delete _mean;
        delete _root;
    }

    void search(const T *target, size_t k, bool dup,  I *results, T *distances){

        std::priority_queue<HeapItem> heap;

        // Variable that tracks the distance to the farthest point in our results
        T tau = std::numeric_limits<T>::max();

        // Perform the searcg
        search(_root, target, k, dup, heap, tau);

        // Gather final results
        int i = 0;
        while (!heap.empty()) {
            T dist = heap.top().dist;
            I idx = heap.top().index;
            heap.pop();
            results[i] = idx;
            distances[i] = dist;
            i++;
        }

        // Results are in reverse order
        std::reverse(results, results + i);
        std::reverse(distances, distances + i);
    }

    void search(const T *target, size_t k, std::vector<I> &results, std::vector<T> &distances){
        results.reserve(k);
        distances.reserve(k);
        search(target, k, false, results.data(), distances.data());
    }

    void insert(size_t n, I *ids, T **inps){
        std::unique_ptr<T[]> tmp = std::unique_ptr<T[]>(new T[dim]());
        ushort d;
        for(size_t i = 0; i < n; i++){
            Node *node = new Node(ids[i], inps[i]);
            if(!_root){
                _root = node;
                _root->color = BLACK;
            }
            else{
                node = insert(_root, node);
            }
            data.push_back(node);
            for(d = 0; d < dim; d++) tmp[d]+=inps[i][d];
            n_total++;
        }
        for(d = 0; d < dim; d++) _mean[d] = ((T)(n_total - n)/(T)n_total)*_mean[d] + (tmp[d]/(T)n_total);
    }

    T* treeMean(){
        return _mean;
    }

    T treeTotal(){
        return n_total;
    }


};


template <typename T>
class BarnesHutTree{

    protected:

    ushort dim;
    int n_splits;
    size_t n_total;

    struct Cell {

        size_t cum_size;
        bool is_leaf;
        T *center;
        T *width;
        T *center_of_mass;
	    std::vector<Cell*> children;

        Cell():cum_size(0),is_leaf(true), center(nullptr),width(nullptr),center_of_mass(nullptr){};

        explicit Cell(ushort dim):Cell(){
            center = new T[dim]();
            width = new T[dim];
        }

        Cell(size_t n, ushort dims, T **inps):Cell(dims){
            std::vector<T> min_Y(dims, std::numeric_limits<T>::max());
            std::vector<T> max_Y(dims, std::numeric_limits<T>::lowest());

            for(size_t i = 0; i < n; i++){
                for(ushort j = 0; j < dims; j++){
                    center[j] += inps[i][j];
                    min_Y[j] = std::min<T>(min_Y[j], inps[i][j]);
                    max_Y[j] = std::max<T>(max_Y[j], inps[i][j]);
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

//        if (cell->cum_size == 0 || (cell->is_leaf)) {
//            return;
//        }

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
        T m = -1;
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
    BarnesHutTree():dim(0), n_splits(0), n_total(0), _root(nullptr){};
    explicit BarnesHutTree(ushort dim): dim(dim), n_total(0), _root(nullptr){
        n_splits = 1 << dim;
    }
    BarnesHutTree(size_t n, ushort dim, T **data):BarnesHutTree(dim){
        insert(n, data);
    };
    ~BarnesHutTree(){
        delete _root;
    }

    void insert(size_t n, T **data){
        if(!_root){
            _root = new Cell(n, dim, data);
        }
        for(size_t i = 0; i < n; i++){
            insert(data[i], _root);
            n_total++;
        }
    }

    T* treeMean(){
        if(_root){
            return _root->center_of_mass;
        }
        else{
            return nullptr;
        }
    }

    size_t treeTotal(){
        return n_total;
    }

    void computeNonEdgeForces(const T *point, const T theta,  T *neg_f, T &sum_Q) const{
        if(_root && _root->cum_size > 0){
            computeNonEdgeForces(point, _root, theta, neg_f, sum_Q);
        }
    }

};

}


#endif