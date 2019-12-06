#define BOOST_TEST_MODULE MyTest

#include <boost/test/unit_test.hpp>
#include <boost/range/numeric.hpp>
#include <memory>
#include <iostream>
#include <sptree.h>
#include <cfloat>
#include <vptree.h>
// #include <splittree.h>
// #include <tsne.h>
#include <tree.hpp>
#include <rand.hpp>
#include <tsne.hpp>


BOOST_AUTO_TEST_SUITE(tsne_test)
    
    BOOST_AUTO_TEST_CASE(vptree_org_test){
        int nx = 1000;
        int dim = 512;
        int k = 10;
        std::unique_ptr<float[]> rnd_f = std::unique_ptr<float[]>(new float[nx*dim]);
        std::unique_ptr<float[]> rnd_qf = std::unique_ptr<float[]>(new float[dim]);
        simile::float_rand(rnd_f.get(), nx*dim, 2023);
        simile::float_rand(rnd_qf.get(), dim, 1231);

        std::unique_ptr<double[]> rnd_d = std::unique_ptr<double[]>(new double[nx*dim]);
        std::unique_ptr<double[]> rnd_qd = std::unique_ptr<double[]>(new double[dim]);

        for(int i = 0; i < nx*dim; i++){
            rnd_d.get()[i] = static_cast<double>(rnd_f.get()[i]);
        }
        for(int i = 0; i < dim; i++){
            rnd_qd.get()[i] = static_cast<double>(rnd_qf.get()[i]);
        }

        std::vector<size_t> res_ids;
        std::vector<double> res_dists;
//        tsne::VpTree<double> tree(nx, dim, rnd_d.get());
        std::vector<double*> rnd_vd;
        for(int i = 0; i < nx; i++){
//            double *d = new double[dim];
//            memcpy(d, rnd_d.get() + i*dim, dim);
            rnd_vd.push_back(rnd_d.get() + i*dim);
        }
        rnd_vd.push_back(rnd_d.get());

//        for(int i = 0; i < nx; i++){
//            rnd_vd.push_back(rnd_d.get() + i*dim);
//
//        }
//        rnd_vd.insert(rnd_vd.end(), rnd_d.get(), rnd_d.get() + dim);
        std::vector<size_t> dummy_ids;
        for(int i = 0; i < nx; i++){
            dummy_ids.push_back(i);
        }
        dummy_ids.push_back(nx);


        tsne::RedBlackTree<size_t, double> tree(nx + 1, dim, dummy_ids.data(), rnd_vd.data());
        tree.search(rnd_qd.get(), k, res_ids, res_dists);


        std::vector<DataPoint> vpDataPoint;
        std::vector<DataPoint> vpR;
        std::vector<double> vpD;
        for(int i = 0; i < nx; i++){
            vpDataPoint.push_back(DataPoint(dim, i, rnd_d.get() + i*dim));
        }
        VpTree<DataPoint, euclidean_distance> ss = VpTree<DataPoint, euclidean_distance>();
        ss.create(vpDataPoint);
        DataPoint q(dim, -1, rnd_qd.get());
        ss.search(q, k, &vpR, &vpD);

        for(int i = 0; i < 10; i++){
            BOOST_CHECK_EQUAL(vpR[i].index(), res_ids[i]);
            BOOST_CHECK_EQUAL(vpD[i], res_dists[i]);
        }

    }

    BOOST_AUTO_TEST_CASE(bhtree_test){
        int nx = 1000;
        int dim = 2;
        std::unique_ptr<float[]> rnd_f = std::unique_ptr<float[]>(new float[nx*dim]);
        simile::float_rand(rnd_f.get(), nx*dim, 1988);
        std::unique_ptr<double[]> rnd_d = std::unique_ptr<double[]>(new double[nx*dim]);

        for(int i = 0; i < nx*dim; i++){
            rnd_d.get()[i] = static_cast<double>(rnd_f.get()[i]);
        }

        tsne::BarnesHutTree<double> tree(nx, 2, rnd_d.get());

        std::unique_ptr<double[]> ng_f = std::unique_ptr<double[]>(new double[dim]);
        double s_q = 0.;
        tree.computeNonEdgeForces(rnd_d.get(), 0.5, ng_f.get(), s_q);


        std::unique_ptr<double[]> ng_ff = std::unique_ptr<double[]>(new double[dim]);
        double s_qq = 0.;
        SPTree stree(dim, rnd_d.get(), nx);
        stree.computeNonEdgeForces(0, 0.5, ng_ff.get(), &s_qq);

        for(int i = 0; i < dim; i++){
            BOOST_CHECK_EQUAL(ng_f[i], ng_ff[i]);
        }

        BOOST_CHECK_EQUAL(s_q, s_qq);

    }

    BOOST_AUTO_TEST_CASE(tsne_construct_test){

        int nx = 1000;
        int x_dim = 512;
        int y_dim = 2;
        std::unique_ptr<float[]> rnd_x = std::unique_ptr<float[]>(new float[nx*x_dim]);
        simile::float_rand(rnd_x.get(), nx*x_dim, 1988);

        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[nx*y_dim]);
        simile::float_rand(rnd_y.get(), nx*y_dim, 1982);

        tsne::TSNE<float> ts(nx, x_dim, y_dim,rnd_x.get(), rnd_y.get());


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