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

    BOOST_AUTO_TEST_CASE(tsne_construct_test){

        struct NodeHolder{
            size_t idx;
            double val;

            bool operator!=(const NodeHolder &b) const{
                return (this->idx != b.idx || this->val != b.val);
            }
        };

        int nx = 1000;
        int x_dim = 512;
        int y_dim = 2;
        std::unique_ptr<float[]> rnd_x = std::unique_ptr<float[]>(new float[nx*x_dim]);
        simile::float_rand(rnd_x.get(), nx*x_dim, 1988);

        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[nx*y_dim]);
        simile::float_rand(rnd_y.get(), nx*y_dim, 1982);


        std::unique_ptr<double[]> rnd_dx = std::unique_ptr<double[]>(new double[nx*x_dim]);
        std::unique_ptr<double[]> rnd_dy = std::unique_ptr<double[]>(new double[nx*y_dim]);

        for(int i = 0; i < nx*x_dim; i++){
            rnd_dx.get()[i] = static_cast<double>(rnd_x.get()[i]);
        }
        for(int i = 0; i < nx*y_dim; i++){
            rnd_dy.get()[i] = static_cast<double>(rnd_y.get()[i]);
        }

        unsigned int* row_P; unsigned int* col_P; double* val_P;
        TSNE::testGaussianPerplexity(rnd_dx.get(), nx, x_dim, &row_P, &col_P, &val_P, 30, 90);


        size_t *t_row_P; size_t *t_col_P; double *t_val_P;

        tsne::TSNE<double> ts(nx, x_dim, y_dim,rnd_dx.get(), rnd_dy.get());
        ts.testGaussianPerplexity(0, 90, 30, &t_row_P, &t_col_P, &t_val_P);
        for(size_t i = 0; i < nx; i++){

            BOOST_CHECK_EQUAL(row_P[i], t_row_P[i]);
            std::vector<NodeHolder> ls;
            std::vector<NodeHolder> rs;
            for(size_t j = row_P[i]; j < row_P[i + 1]; j ++){
                NodeHolder l;
                l.idx = col_P[j];
                l.val = val_P[j];
                ls.push_back(l);

                NodeHolder r;
                r.idx = t_col_P[j];
                r.val = t_val_P[j];
                rs.push_back(r);
            }
            std::sort(ls.begin(), ls.end(), [](const NodeHolder &a, const NodeHolder &b)->bool{return a.idx > b.idx;});
            std::sort(rs.begin(), rs.end(), [](const NodeHolder &a, const NodeHolder &b)->bool{return a.idx > b.idx;});
            for(size_t k = 0; k < ls.size(); k++){
                BOOST_CHECK_EQUAL(ls[k].idx, rs[k].idx);
                BOOST_CHECK_EQUAL(ls[k].val, rs[k].val);
//                BOOST_CHECK_CLOSE(ls[k].val, rs[k].val, 0.00001);
            }
        }

    }

    BOOST_AUTO_TEST_CASE(tsne_gradient_test){

        int nx = 1000;
        int x_dim = 512;
        int y_dim = 2;
        std::unique_ptr<float[]> rnd_x = std::unique_ptr<float[]>(new float[nx*x_dim]);
        simile::float_rand(rnd_x.get(), nx*x_dim, 1988);

        std::unique_ptr<float[]> rnd_y = std::unique_ptr<float[]>(new float[nx*y_dim]);
        simile::float_rand(rnd_y.get(), nx*y_dim, 1982);


        std::unique_ptr<double[]> rnd_dx = std::unique_ptr<double[]>(new double[nx*x_dim]);
        std::unique_ptr<double[]> rnd_dxt = std::unique_ptr<double[]>(new double[nx*x_dim]);
        std::unique_ptr<double[]> rnd_dy = std::unique_ptr<double[]>(new double[nx*y_dim]);
        std::unique_ptr<double[]> rnd_dyt = std::unique_ptr<double[]>(new double[nx*y_dim]);


        for(int i = 0; i < nx*x_dim; i++){
            rnd_dx.get()[i] = static_cast<double>(rnd_x.get()[i]);
            rnd_dxt.get()[i] = static_cast<double>(rnd_x.get()[i]);
        }
        for(int i = 0; i < nx*y_dim; i++){
            rnd_dy.get()[i] = static_cast<double>(rnd_y.get()[i]);
            rnd_dyt.get()[i] = static_cast<double>(rnd_y.get()[i]);
        }
        double *dY = new double[nx * y_dim];
//        TSNE::testGradient(rnd_dx.get(), rnd_dy.get(), nx, x_dim, y_dim, 30, 0.5, dY);


        tsne::TSNE<double> ts(nx, x_dim, y_dim, rnd_dxt.get(), rnd_dyt.get());
//        double *dYt = new double[nx * y_dim];
//        ts.testGradient(0, 30, 0.5, dYt);
//        for(size_t i = 0; i < nx; i++){
////            BOOST_CHECK_CLOSE(dY[i], dYt[i], 0.00001);
//            BOOST_CHECK_EQUAL(dY[i], dYt[i]);
//        }

        TSNE::run(rnd_dx.get(), nx, x_dim, rnd_dy.get(), y_dim, 30, 0.5, 1988, true, 10, 200, 200);
        ts.run(0, NULL, rnd_dyt.get(), 30, 0.5, false, false, 10, 200, 200);
        for(size_t i = 0; i < nx; i++){
            BOOST_CHECK_CLOSE(rnd_dy[i], rnd_dyt[i], 0.00001);
//            BOOST_CHECK_EQUAL(rnd_dy[i], rnd_dyt[i]);

        }
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

        tsne::TSNE<double> ts(n, target_dim, 2, features.data(), rnd_dy.get());

        ts.run(0, NULL, rnd_dyt.get(), 30, 0.5, false, false, 1000, 200, 200);
        tsne::save_device_features(output_path, n, source_dim, target_dim, features.data(), rnd_dyt.get(), pca.data(), image_uris);

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
//        double * ret_ys = new double[fixed_n + var_n];

        ts.run(var_n, var_features.data(), rnd_dy.get(), 30, 0.5, false, true, 500, 200, 200);

        features.insert(features.end(), var_features.begin(), var_features.end());
//        ys.insert(ys.end(), rnd_dy.get(), rnd_dy.get() + var_n * 2);
        for(int i = 0; i < var_n; i++){
            image_uris.push_back(var_image_uris[i]);
        }

        tsne::save_device_features(output_path, fixed_n + var_n, source_dim, target_dim, features.data(), rnd_dy.get(), pca.data(), image_uris);
//        tsne::save_device_features(output_path, fixed_n + var_n, source_dim, target_dim, features.data(), ys.data(), pca.data(), image_uris);

    }


BOOST_AUTO_TEST_SUITE_END()