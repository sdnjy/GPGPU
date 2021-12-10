#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "processing.hpp"

/*
TODO:
* Balancer le tout à _compute_features
* (bon en gros faire image_to_features)
*/

int main() {
    std::string path = "../collective_database/PXL_20211101_175643604.jpg";

    int scale_factor = 1;
    int pool_size = 31;
    int postproc_size = 5;

    image_to_features(path, scale_factor, pool_size, postproc_size);

    return 0;
}
