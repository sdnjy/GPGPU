#pragma once

#include <stddef.h>
#include <cstdint>

#include "img_features.hpp"


ImgFeatures<uint8_t>* _compute_features(double** img, const size_t width, const size_t height);

template<typename T>
T abs(T value);
