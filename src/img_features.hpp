#pragma once

#include <opencv2/core.hpp>


class ImgFeatures
{
public:
    ImgFeatures(size_t size_dim1, size_t size_dim2, int type);
    ~ImgFeatures();

    cv::Mat* img_feature_x;
    cv::Mat* img_feature_y;
};
