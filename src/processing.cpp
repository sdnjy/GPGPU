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

    ImgFeatures* img_features = _compute_features(img, img.rows, img.cols);

    size_t num_patchs_x = img.cols / pool_size;
    size_t num_patchs_y = img.rows / pool_size;

    //save both features
    cv::imwrite("totox.jpg", *img_features->img_feature_x);
    cv::imwrite("totoy.jpg", *img_features->img_feature_y);

    
    cv::Mat* img_feature_x = img_features->img_feature_x;
    cv::Mat* img_feature_y = img_features->img_feature_y;

    // Setup a rectangle to define your region of interest for the crop
    cv::Rect myROI(0, 0, img.cols - img.cols % (pool_size * num_patchs_x), img.rows - img.rows % (pool_size * num_patchs_y));

    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    cv::Mat features_crop_x = (*img_feature_x)(myROI);
    cv::Mat features_crop_y = (*img_feature_y)(myROI);

    int16_t tmp_response[num_patchs_y][num_patchs_x];

    // fill features_patch

    // Allocate table of size heigth/pool_size * weight/pool_size

    for (int i = 0; i < features_crop_x.rows; ++i)
    {
        for (int j = 0; j < features_crop_x.cols; ++j)
        {
            auto f_x = features_crop_x.at<uint8_t>(i, j);
            auto f_y = features_crop_y.at<uint8_t>(i, j);

            auto diff = f_x - f_y;
            tmp_response[i / pool_size][j / pool_size] += diff;
        }
    }

    const int pool_size_squared = pool_size * pool_size;

    uint8_t max_value = 0;
    // clip between 0 and 255
    uint8_t response[num_patchs_y][num_patchs_x];
    for (int i_patch_y = 0; i_patch_y < num_patchs_y; ++i_patch_y)
    {
        for (int i_patch_x = 0; i_patch_x < num_patchs_x; ++i_patch_x)
        {
            int16_t current_value = tmp_response[i_patch_y][i_patch_x] / pool_size_squared;

            if (current_value <= 0)
                response[i_patch_y][i_patch_x] = 0;
            else if (current_value <= 255)
                response[i_patch_y][i_patch_x] = current_value;
            else
                response[i_patch_y][i_patch_x] = 255;

            if (response[i_patch_y][i_patch_x] > max_value)
                max_value = response[i_patch_y][i_patch_x];
        }
    }

    const uint8_t threshold = max_value / 2;
    for (int i_patch_y = 0; i_patch_y < num_patchs_y; ++i_patch_y)
    {
        for (int i_patch_x = 0; i_patch_x < num_patchs_x; ++i_patch_x)
        {
            response[i_patch_y][i_patch_x] = 255 * (response[i_patch_y][i_patch_x] >= threshold);
        }
    }

    
    cv::Mat mat_response(num_patchs_y, num_patchs_x, CV_8UC1, response);

    cv::imwrite("mat_response.jpg", mat_response);
    delete img_features;

}

template<typename T>
T abs(T value)
{
    if (value < 0)
        return -value;
    return value;
}
