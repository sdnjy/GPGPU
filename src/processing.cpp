#include <iostream>

#include "processing.hpp"
#include "img_features.hpp"
#include "array.hpp"


ImgFeatures* _compute_features(cv::Mat& img, const size_t height, const size_t width)
{
    const int n_filters = 2;

    ImgFeatures* img_features = new ImgFeatures(height, width, CV_8UC1);
    cv::Mat* img_feature_x = img_features->img_feature_x;
    cv::Mat* img_feature_y = img_features->img_feature_y;
    
    
    float sobelx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float sobely[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (size_t ii = 1; ii < height - 1; ++ii)
    {
        for (size_t jj = 1; jj < width - 1; ++jj)
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

// Crop image
ImgFeatures* crop(ImgFeatures* img_feat, size_t num_patch_x, size_t num_patch_y, int pool_size)
{
    // Get matrices from struct
    cv::Mat* img_feature_x = img_feat->img_feature_x;
    cv::Mat* img_feature_y = img_feat->img_feature_y;

    // Calculate dim of cropped images
    size_t width = img_feature_x->cols - img_feature_x->cols % (pool_size * num_patch_x);
    size_t height = img_feature_x->rows - img_feature_x->rows % (pool_size * num_patch_y);

    //Create arrays
    ImgFeatures* crop_features = new ImgFeatures(height, width, CV_8UC1);
    cv::Mat* crop_feature_x = crop_features->img_feature_x;
    cv::Mat* crop_feature_y = crop_features->img_feature_y;

    for (size_t i = 0; i < height; ++i)
    {
        for(size_t j = 0; j < width; ++j)
        {
            crop_feature_x->at<uint8_t>(i, j) = img_feature_x->at<uint8_t>(i, j);
            crop_feature_y->at<uint8_t>(i, j) = img_feature_y->at<uint8_t>(i, j);
        }
    }

    // Delete old images
    delete img_feat;

    return crop_features;
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
    //cv::Rect myROI(0, 0, img.cols - img.cols % (pool_size * num_patchs_x), img.rows - img.rows % (pool_size * num_patchs_y));
    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    //cv::Mat features_crop_x = (*img_feature_x)(myROI);
    //cv::Mat features_crop_y = (*img_feature_y)(myROI);

    img_features = crop(img_features, num_patchs_x, num_patchs_y, pool_size);

    cv::Mat features_crop_x = *img_features->img_feature_x;
    cv::Mat features_crop_y = *img_features->img_feature_y;

    cv::imwrite("crop_totox.jpg", features_crop_x);
    cv::imwrite("crop_totoy.jpg", features_crop_y);

    // Allocate table of size heigth/pool_size * weight/pool_size
    int tmp_response[num_patchs_y][num_patchs_x];
    memset(tmp_response, 0, sizeof(tmp_response));

    // fill features_patch

    for (int i = 0; i < features_crop_x.rows; ++i)
    {
        for (int j = 0; j < features_crop_x.cols; ++j)
        {
            int f_x = features_crop_x.at<uint8_t>(i, j);
            int f_y = features_crop_y.at<uint8_t>(i, j);

            int diff = f_x - f_y;
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

    cv::Mat mat_response2(num_patchs_y, num_patchs_x, CV_8UC1, response);
    cv::imwrite("mat_response_before_morpho.jpg", mat_response2);

    // Make copy of response
    uint8_t tmp_morpho[num_patchs_y][num_patchs_x];
    memset(tmp_morpho, 0, sizeof(tmp_morpho));

    // dilation
    for (int i_patch_y = 1; i_patch_y < num_patchs_y - 1; ++i_patch_y)
    {
        for (int i_patch_x = 2; i_patch_x < num_patchs_x - 2; ++i_patch_x)
        {
            uint8_t max = 0;
            //For each pixel (without padding) see if there is a 1 in the 3x5 surronding neighbours
            for (int i_kernel = i_patch_y  - 1; i_kernel <= i_patch_y  + 1; ++i_kernel)
            {
                for (int j_kernel = i_patch_x - 2; j_kernel <= i_patch_x + 2; ++j_kernel)
                {
                    if (response[i_kernel][j_kernel] > max)
                    {
                        max = response[i_kernel][j_kernel];
                    }
                }
            } 
            tmp_morpho[i_patch_y][i_patch_x] = max;
        }
    }

    // erosion
    for (int i_patch_y = 1; i_patch_y < num_patchs_y - 1; ++i_patch_y)
    {
        for (int i_patch_x = 2; i_patch_x < num_patchs_x - 2; ++i_patch_x)
        {
            uint8_t min = 255;
            for (int i_kernel = i_patch_y  - 1; i_kernel <= i_patch_y  + 1; ++i_kernel)
            {
                for (int j_kernel = i_patch_x - 2; j_kernel <= i_patch_x + 2; ++j_kernel)
                {
                    if (tmp_morpho[i_kernel][j_kernel] < min)
                    {
                        min = tmp_morpho[i_kernel][j_kernel];
                    }
                }
            } 
            response[i_patch_y][i_patch_x] = min;
        }
    }

    cv::Mat mat_response_1(num_patchs_y, num_patchs_x, CV_8UC1, response);
    cv::imwrite("mat_response_morphed.jpg", mat_response_1);


    // Use threshold to activate patch
    const uint8_t threshold = max_value / 2;
    for (int i_patch_y = 0; i_patch_y < num_patchs_y; ++i_patch_y)
    {
        for (int i_patch_x = 0; i_patch_x < num_patchs_x; ++i_patch_x)
        {
            response[i_patch_y][i_patch_x] = 255 * (response[i_patch_y][i_patch_x] > threshold);
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
