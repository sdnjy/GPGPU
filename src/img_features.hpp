#pragma once

#include "array.hpp"

template<typename T>
class ImgFeatures
{
public:
    ImgFeatures(const size_t size_dim1, const size_t size_dim2);
    ~ImgFeatures();

    Array<T>* img_feature_x;
    Array<T>* img_feature_y;
};

#include "img_features.tpp"