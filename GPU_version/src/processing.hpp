#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
               
void image_to_features(std::string path,
                       const int scale_factor,
                       const int pool_size,
                       const int postproc_size,
                       const std::string output_path);
