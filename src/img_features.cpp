#include "img_features.hpp"

ImgFeatures::ImgFeatures(size_t size_dim1, size_t size_dim2, int type)
{
    this->img_feature_x = new cv::Mat(size_dim1, size_dim2, type, 0.0);
    this->img_feature_y = new cv::Mat(size_dim1, size_dim2, type, 0.0);
}

ImgFeatures::~ImgFeatures()
{
    delete this->img_feature_x;
    delete this->img_feature_y;
}
