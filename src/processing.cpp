#include <iostream>

#include "processing.hpp"
#include "img_features.hpp"
#include "array.hpp"


ImgFeatures* _compute_features(cv::Mat& img, const size_t width, const size_t height)
{
    const int n_filters = 2;

    ImgFeatures* img_features = new ImgFeatures(width, height, CV_8UC1);
    cv::Mat* img_feature_x = img_features->img_feature_x;
    cv::Mat* img_feature_y = img_features->img_feature_y;
    
    
    float sobelx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float sobely[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (size_t ii = 1; ii < width - 1; ++ii)
    {
        for (size_t jj = 1; jj < height - 1; ++jj)
        {
            float sum_x = 0;
            float sum_y = 0;

            for (size_t i = 0; i < 3; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    auto img_current_pixel = img.at<uint8_t>(ii + i - 1, jj + j - 1);
                    sum_x += sobelx[i][j] * img_current_pixel;
                    sum_y += sobely[i][j] * img_current_pixel;
                }
            }
            // sobel x
            img_feature_x->at<uint8_t>(ii, jj) = abs<float>(sum_x);
            // sobel y
            img_feature_y->at<uint8_t>(ii, jj) = abs<float>(sum_y);
        }
    }

    return img_features;
}

// FIXME return features
void image_to_features(std::string path, int scale_factor, int pool_size, int postproc_size)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
    {
        std::cerr << "Could not read the image" << std::endl;
        exit(1);
    }

    ImgFeatures* img_features = _compute_features(img, img.cols, img.rows);

    size_t patchs_x = img.cols / pool_size;
    size_t patchs_y = img.rows / pool_size;

    
    
}

template<typename T>
T abs(T value)
{
    if (value < 0)
        return -value;
    return value;
}
