#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


void _compute_features(unsigned char* img, unsigned char* sobel_x, unsigned char* sobel_y, const size_t height, const size_t width);
void image_to_features(std::string path, int scale_factor, int pool_size, int postproc_size);

template<typename T>
T abs(T value);
