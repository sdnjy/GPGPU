#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


#include "img_features.hpp"


ImgFeatures* _compute_features(cv::Mat& img, const size_t width, const size_t height);
void image_to_features(std::string path, int scale_factor, int pool_size, int postproc_size);

template<typename T>
T abs(T value);
