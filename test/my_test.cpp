#define BOOST_TEST_MODULE MyTest

#include <boost/test/unit_test.hpp>
#include <boost/range/numeric.hpp>
#include <memory>
#include <iostream>
// #include <vptree.h>
// #include <splittree.h>
// #include <tsne.h>
#include <tree.hpp>
#include <rand.hpp>
#include <scope_tsne.h>


BOOST_AUTO_TEST_SUITE(tsne_test)
    
    BOOST_AUTO_TEST_CASE(vptree_org_test){
        int nx = 1000;
        int dim = 512;
        std::unique_ptr<float[]> rnd_f = std::unique_ptr<float[]>(new float[nx*dim]);
        std::unique_ptr<float[]> rnd_qf = std::unique_ptr<float[]>(new float[dim]);
        simile::float_rand(rnd_f.get(), nx*dim, 1988);
        simile::float_rand(rnd_qf.get(), dim, 1989);
        std::vector<size_t> res_ids;
        std::vector<float> res_dists;
        tsne::VpTree<float> tree(nx, dim, rnd_f.get());
        tree.search(rnd_qf.get(), 10, res_ids, res_dists);
        float min_f = res_dists[0]; 
        for (int i = 1; i < 10; i++){
            std::cout << res_ids[i] << ": " << res_dists[i] << std::endl;
            BOOST_CHECK_LT(min_f, res_dists[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(bhtree_test){
        int nx = 1000;
        int dim = 2;
        std::unique_ptr<float[]> rnd_f = std::unique_ptr<float[]>(new float[nx*dim]);
        simile::float_rand(rnd_f.get(), nx*dim, 1988);
        tsne::BarnesHutTree<float> tree(nx, 2, rnd_f.get());

        BOOST_CHECK_EQUAL(1, 1);

    }

//    BOOST_AUTO_TEST_CASE(gaussian_perplexity_test){
//        int nx = 1000;
//        int dim = 512;
//        tsne::TSNE Ts;
//        std::unique_ptr<float[]> rnd_f = std::unique_ptr<float[]>(new float[nx*dim]);
//        simile::float_rand(rnd_f.get(), nx*dim, 1988);
//        std::unique_ptr<double[]> rnd_d = std::unique_ptr<double[]>(new double[nx*dim]);
//        for(int i = 0; i < nx * dim; i++){
//            rnd_d.get()[i] = rnd_f.get()[i];
//        }
//        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[nx*2]);
//        std::unique_ptr<double[]> rnd_yd = std::unique_ptr<double[]>(new double[nx*2]);
//        simile::float_rand(rnd_y.get(), nx*2, 1988);
//
//        for(int i = 0; i < nx * 2; i++){
//            rnd_yd.get()[i] = rnd_y.get()[i];
//        }
//
//        Ts.run(rnd_d.get(), nx, dim, rnd_yd.get(), 2, 30, 0.5, 12, 1000, 0, 0, 1);
//
//    }

    // BOOST_AUTO_TEST_CASE(vptree_org_test){
    //     int nx = 1000;
    //     int dim = 512;
    //     int K = 10;
    //     std::unique_ptr<float[]> rnd_f = std::unique_ptr<float[]>(new float[nx*dim]);
    //     std::unique_ptr<double[]> rnd_d = std::unique_ptr<double[]>(new double[nx*dim]);
    //     simile::float_rand(rnd_f.get(), nx*dim, 1988);
    //     for(int i = 0; i < nx*dim; i++){
    //         rnd_d.get()[i] = static_cast<double>(rnd_f.get()[i]);
    //     }

    //     std::vector<DataPoint> object_X(nx, DataPoint(dim, -1, rnd_d.get()));
    //     for(int i = 0; i < nx; i++){
    //         object_X[i] = DataPoint (dim, i, rnd_d.get() + i*dim);

    //     }
    //     VpTree<DataPoint, euclidean_distance>*tree = new VpTree<DataPoint, euclidean_distance>(); 
    //     tree->create(object_X);
    //     for (int i = 0; i < nx; i++){
    //         std::vector<DataPoint> indices;
    //         std::vector<double> distances;
    //         tree->search(object_X[i], K + 1, &indices, &distances);
    //         BOOST_CHECK_EQUAL(indices[0].index(), i);
    //     }
    
    // }

    // BOOST_AUTO_TEST_CASE(tsne_test){
    //     int nx = 1000;
    //     int dim = 128;

    //     std::unique_ptr<double[]> rnd_x = std::unique_ptr<double[]>(new double[nx*dim]);
    //     std::unique_ptr<double[]> rnd_y = std::unique_ptr<double[]>(new double[nx*2]);

    //     std::unique_ptr<float[]> rnd_x_f = std::unique_ptr<float[]>(new float[nx*dim]);
    //     std::unique_ptr<float[]> rnd_y_f = std::unique_ptr<float[]>(new float[nx*2]);

    //     for(int i = 0; i < nx*dim; i++){
    //         rnd_x.get()[i] = static_cast<double>(rnd_x_f.get()[i]);
    //     }

    //     for(int i = 0; i < nx*2; i++){
    //         rnd_y.get()[i] = static_cast<double>(rnd_y_f.get()[i]);
    //     }
    //     TSNE<SplitTree, euclidean_distance> tsne; 
    //     tsne.run(rnd_x.get(), nx, dim, rnd_y.get());
    //     int ff = 0;

    // }


BOOST_AUTO_TEST_SUITE_END()