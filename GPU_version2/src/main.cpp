#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <experimental/filesystem>

#include "processing.hpp"

/*
TODO:
* Balancer le tout à _compute_features
* (bon en gros faire image_to_features)
*/

int main() {
    const int scale_factor = 1;
    const int pool_size = 31;
    const int postproc_size = 5;


    const std::string input_dir_path = "../../input_images/";
    const std::string output_dir_path = "../../output_images/";

    for (const auto& entry : std::experimental::filesystem::directory_iterator(input_dir_path))
    {
        const std::string image_path = entry.path();
        const int i_start_image_name = image_path.find_last_of("/") + 1;
        const std::string image_name = image_path.substr(i_start_image_name,
                                                         image_path.find_last_of(".") - i_start_image_name);

        const std::string output_path = output_dir_path + image_name + "_barcode.jpg";
        image_to_features(image_path, scale_factor, pool_size, postproc_size, output_path);
    }

    return 0;
}
