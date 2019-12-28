#define BOOST_TEST_MODULE MyTest

#include <boost/test/unit_test.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <memory>
#include <tsne.hpp>
#include "multicore_tsne/tsne.h"
#include "multicore_tsne/splittree.h"
#include "multicore_tsne/vptree.h"


BOOST_AUTO_TEST_SUITE(tsne_test)

    boost::random::mt19937 gen;

    BOOST_AUTO_TEST_CASE(tsne_run){
        int nx = 10000;
        ushort x_dim = 728;
        ushort y_dim = 2;
        std::unique_ptr<double[]> rnd_x = std::unique_ptr<double[]>(new double[nx*x_dim]);
        std::unique_ptr<double[]> y_ret = std::unique_ptr<double[]>(new double[nx*y_dim]);
//        std::unique_ptr<double[]> y_org = std::unique_ptr<double[]>(new double[nx*y_dim]);
        boost::random::uniform_real_distribution<double> dist(0, 1.0);

        for(size_t i = 0; i < nx*x_dim; i++) rnd_x[i] = dist(gen);

        tsne::TSNE<double> ts(x_dim, y_dim, true);

        ts.run(nx, rnd_x.get(), 200, 30, 0.5, 300, 250, 250,
                y_ret.get());

        TSNE<SplitTree, euclidean_distance_squared> tsne;
        tsne.run(rnd_x.get(), nx, x_dim, y_ret.get(), 2, 30, 0.5, -1, 300, 250, 0, false, 1);

    }


    BOOST_AUTO_TEST_CASE(multicore_tsne_run){
        int nx = 10000;
        ushort x_dim = 728;
        ushort y_dim = 2;
        std::unique_ptr<double[]> rnd_x = std::unique_ptr<double[]>(new double[nx*x_dim]);
        std::unique_ptr<double[]> y_ret = std::unique_ptr<double[]>(new double[nx*y_dim]);
        boost::random::uniform_real_distribution<double> dist(0, 1.0);

        for(size_t i = 0; i < nx*x_dim; i++) rnd_x[i] = dist(gen);
        TSNE<SplitTree, euclidean_distance_squared> tsne;
        tsne.run(rnd_x.get(), nx, x_dim, y_ret.get(), 2, 30, 0.5, -1, 300, 250, 0, false, 1);

    }

//    BOOST_AUTO_TEST_CASE(tsne_generate){
//
//        std::string input_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000.data";
//        std::string output_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000_test.results";
//        std::vector<std::string> image_uris;
//        std::vector<double> features;
//        std::vector<double> pca;
//        std::vector<double> ys;
//        int n = 0;
//        int source_dim = 0;
//        int target_dim = 0;
//
//        tsne::load_device_features(input_path, &n, &source_dim, &target_dim, features, ys, pca, image_uris);
//        if(n == 0){
//            fprintf(stderr, "No Initialized Data");
//            return ;
//        }
//
//        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[n*2]);
//        simile::float_rand(rnd_y.get(), n*2, 1982);
//
//        std::unique_ptr<double[]> rnd_dy = std::unique_ptr<double[]>(new double[n*2]);
//        std::unique_ptr<double[]> rnd_dyt = std::unique_ptr<double[]>(new double[n*2]);
//
//        for(int i = 0; i < n*2; i++){
//            rnd_dy.get()[i] = static_cast<double>(rnd_y.get()[i]);
//        }
//
//        tsne::TSNE<double> ts(target_dim, 2, true);
//        ts.run(n, features.data(), 30, 0.5, 1000, 250, 250, rnd_dy.get());
//
//        tsne::save_device_features(output_path, n, source_dim, target_dim, features.data(), rnd_dy.get(), pca.data(), image_uris);
//
//    }
//
//    BOOST_AUTO_TEST_CASE(tsne_generate_add){
//
//        std::string fixed_input_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000_test.results";
//        std::string var_input_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_362.data";
//        std::string output_path = "/home/andy/Data/projects/competitive_intelligent/tsne_coords/phone_2000_test_add.results";
//        std::vector<std::string> image_uris;
//        std::vector<double> features;
//        std::vector<double> pca;
//        std::vector<double> ys;
//
//        std::vector<std::string> var_image_uris;
//        std::vector<double> var_features;
//        std::vector<double> var_pca;
//        std::vector<double> var_ys;
//
//        int fixed_n = 0;
//        int var_n = 0;
//        int source_dim = 0;
//        int target_dim = 0;
//        int full_dim = 0;
//
//        tsne::load_device_features(fixed_input_path, &fixed_n, &source_dim, &target_dim, features, ys, pca, image_uris);
//        tsne::load_device_features(var_input_path, &var_n, &source_dim, &full_dim, var_features, var_ys, var_pca, var_image_uris);
//
////        var_n = 1;
//
//        std::unique_ptr<double[]> rnd_y = std::unique_ptr<double[]>(new double[var_n*2]);
//
//        tsne::TSNE<double> ts(fixed_n, target_dim, 2, features.data(), ys.data(), true);
//
//        ts.run(var_n, var_features.data(), 30, 0.5, 300, 200, 200, rnd_y.get());
//
//        features.insert(features.end(), var_features.begin(), var_features.begin() + var_n*target_dim);
//        ys.insert(ys.end(), rnd_y.get(), rnd_y.get() + var_n * 2);
//        for(int i = 0; i < var_n; i++){
//            image_uris.push_back(var_image_uris[i]);
//        }
//
//        tsne::save_device_features(output_path, fixed_n + var_n, source_dim, target_dim, features.data(), ys.data(), pca.data(), image_uris);
//
//    }


BOOST_AUTO_TEST_SUITE_END()