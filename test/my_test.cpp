#define BOOST_TEST_MODULE MyTest

#include <boost/test/unit_test.hpp>
#include <boost/range/numeric.hpp>
#include <memory>
#include <iostream>
#include <sptree.h>
#include <cfloat>
#include <vptree.h>
// #include <splittree.h>
#include <tsne.h>

#include <tree.hpp>
#include <rand.hpp>
#include <tsne.hpp>
#include <utils.hpp>


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
        std::vector<double*> rnd_dv;
        rnd_dv.reserve(nx * dim);

        for(int i = 0; i < nx*dim; i++){
            rnd_d.get()[i] = static_cast<double>(rnd_f.get()[i]);
        }
        for(int i = 0; i < nx; i++){
            double *t = rnd_d.get() + i * dim;
            rnd_dv.push_back(t);
        }

        tsne::BarnesHutTree<double> tree(nx, dim, rnd_dv.data());

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

    BOOST_AUTO_TEST_CASE(tsne_run){

        int nx = 2000;
        ushort x_dim = 512;
        ushort y_dim = 2;
        std::unique_ptr<float[]> rnd_x = std::unique_ptr<float[]>(new float[nx*x_dim]);
        std::unique_ptr<float[]> y_ret = std::unique_ptr<float[]>(new float[nx*y_dim]);
        simile::float_rand(rnd_x.get(), nx*x_dim, 1988);
        tsne::TSNE<float> ts(x_dim, y_dim);

        ts.run(nx, rnd_x.get(), 30, 0.5, 1000, 250, 250, y_ret.get());

    }

    BOOST_AUTO_TEST_CASE(tsne_generate){

        std::string input_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000.data";
        std::string output_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000_test.results";
        std::vector<std::string> image_uris;
        std::vector<double> features;
        std::vector<double> pca;
        std::vector<double> ys;
        int n = 0;
        int source_dim = 0;
        int target_dim = 0;

        tsne::load_device_features(input_path, &n, &source_dim, &target_dim, features, ys, pca, image_uris);
        if(n == 0){
            fprintf(stderr, "No Initialized Data");
            return ;
        }

        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[n*2]);
        simile::float_rand(rnd_y.get(), n*2, 1982);

        std::unique_ptr<double[]> rnd_dy = std::unique_ptr<double[]>(new double[n*2]);
        std::unique_ptr<double[]> rnd_dyt = std::unique_ptr<double[]>(new double[n*2]);

        for(int i = 0; i < n*2; i++){
            rnd_dy.get()[i] = static_cast<double>(rnd_y.get()[i]);
        }

//        tsne::TSNE<double> ts(n, target_dim, 2, features.data(), rnd_dy.get());
        tsne::TSNE<double> ts(target_dim, 2);
        ts.run(n, features.data(), 30, 0.5, 1000, 200, 200, rnd_dy.get());

        tsne::save_device_features(output_path, n, source_dim, target_dim, features.data(), rnd_dy.get(), pca.data(), image_uris);

    }

    BOOST_AUTO_TEST_CASE(tsne_generate_add){

        std::string fixed_input_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000_test.results";
        std::string var_input_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_362.data";
        std::string output_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000_test_add.results";
        std::vector<std::string> image_uris;
        std::vector<double> features;
        std::vector<double> pca;
        std::vector<double> ys;

        std::vector<std::string> var_image_uris;
        std::vector<double> var_features;
        std::vector<double> var_pca;
        std::vector<double> var_ys;

        int fixed_n = 0;
        int var_n = 0;
        int source_dim = 0;
        int target_dim = 0;
        int full_dim = 0;

        tsne::load_device_features(fixed_input_path, &fixed_n, &source_dim, &target_dim, features, ys, pca, image_uris);
        tsne::load_device_features(var_input_path, &var_n, &source_dim, &full_dim, var_features, var_ys, var_pca, var_image_uris);

//        var_n = 20

        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[var_n*2]);
        simile::float_rand(rnd_y.get(), var_n*2, 1982);

        std::unique_ptr<double[]> rnd_dy = std::unique_ptr<double[]>(new double[(var_n + fixed_n)*2]);
//        std::unique_ptr<double[]> rnd_dyt = std::unique_ptr<double[]>(new double[var_n*2]);

        for(int i = 0; i < var_n*2; i++){
            rnd_dy.get()[i] = static_cast<double>(rnd_y.get()[i]) ;
        }

        tsne::TSNE<double> ts(fixed_n, target_dim, 2, features.data(), ys.data());

        ts.run(var_n, var_features.data(), 30, 0.5, 300, 200, 200, rnd_dy.get());

        features.insert(features.end(), var_features.begin(), var_features.end());
        ys.insert(ys.end(), rnd_dy.get(), rnd_dy.get() + var_n * 2);
        for(int i = 0; i < var_n; i++){
            image_uris.push_back(var_image_uris[i]);
        }

//        tsne::save_device_features(output_path, fixed_n + var_n, source_dim, target_dim, features.data(), rnd_dy.get(), pca.data(), image_uris);
        tsne::save_device_features(output_path, fixed_n + var_n, source_dim, target_dim, features.data(), ys.data(), pca.data(), image_uris);

    }


BOOST_AUTO_TEST_SUITE_END()