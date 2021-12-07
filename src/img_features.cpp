#include "img_features.hpp"

template<typename T>
ImgFeatures<T>::ImgFeatures(const size_t size_dim1, const size_t size_dim2)
{
    this->img_feature_x = new Array<T>(size_dim1, size_dim2);
    this->img_feature_y = new Array<T>(size_dim1, size_dim2);
}

template<typename T>
ImgFeatures<T>::~ImgFeatures()
{
    delete this->img_feature_x;
    delete this->img_feature_y;
}