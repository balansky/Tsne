//
// Created by andy on 2019-12-11.
//

#ifndef TSNE_UTILS_HPP
#define TSNE_UTILS_HPP

namespace tsne{

    bool load_device_features(std::string data_path, int *n, int *source_dim, int *target_dim,
                              std::vector<double> &features, std::vector<double> &ys, std::vector<double> &pca,
                              std::vector<std::string> &file_uri){
        FILE *h;
        int ny = 0;
        std::string device_data_path = data_path;
        if((h = fopen(device_data_path.c_str(), "r+b")) == NULL) {
            fprintf(stderr, "Load Error: could not open data file(%s).\n", device_data_path.c_str());
            return false;
        }
        fread(n, sizeof(int), 1, h);
        fread(source_dim, sizeof(int), 1, h);
        fread(target_dim, sizeof(int), 1, h);
        fread(&ny, sizeof(int), 1, h);
        features.resize((*n)*(*target_dim));
        fread(features.data(), sizeof(double), (*n)*(*target_dim), h);
        ys.resize((*n)*2);
        if(ny != 0){
            fread(ys.data(), sizeof(double), ny*2, h);
        }
        pca.resize((*source_dim) * (*target_dim));
        fread(pca.data(), sizeof(double), (*source_dim) * (*target_dim), h);
        char c;
        std::vector<char> cs;
        while(! feof(h)){
            fread(&c, sizeof(char), 1, h);
            if(c != '\n')
            {
                cs.push_back(c);
            }
            else if(!cs.empty()){
                std::string s(cs.begin(), cs.end());
                file_uri.push_back(s);
                cs.clear();
            }
        }
        fclose(h);
        return true;
    }

    bool save_device_features(std::string output_path, int n, int source_dim, int target_dim,
                              double *features, double *ys, double *pca, std::vector<std::string> &image_uris){

        std::string result_path = output_path;
        FILE *h;

        if((h = fopen(result_path.c_str(), "w+b")) == NULL) {
            fprintf(stderr, "Save Data Error: could not open data file %s.\n", result_path.c_str());
            return false;
        }

        fwrite(&n, sizeof(int), 1, h);
        fwrite(&source_dim, sizeof(int), 1, h);
        fwrite(&target_dim, sizeof(int), 1, h);
        fwrite(&n, sizeof(int), 1, h);
        fwrite(features, sizeof(double), n*target_dim, h);
        fwrite(ys, sizeof(double), n*2, h);
        fwrite(pca, sizeof(double), source_dim*target_dim, h);
        for(size_t i = 0; i < image_uris.size(); i++){
            std::string p = image_uris[i] + "\n";
            fwrite(p.c_str(), sizeof(char), p.size(), h);
        }
        fclose(h);
        return true;

    }



}

#endif //TSNE_UTILS_HPP
