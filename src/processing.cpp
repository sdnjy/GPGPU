#include <iostream>
#include <stdio.h>
#include "processing.hpp"


void _compute_features(unsigned char* img, unsigned char* sobel_x, unsigned char* sobel_y, const size_t height, const size_t width)
{
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
                    size_t id = (ii + i - 1) * width + (jj + j - 1);
                    auto img_current_pixel = img[id];
                    sum_x += sobelx[i][j] * img_current_pixel;
                    sum_y += sobely[i][j] * img_current_pixel;
                }
            }
            // sobel x
            size_t id_ = ii * width + jj;
            sobel_x[id_] = abs<float>(sum_x);
            // sobel y
            sobel_y[id_] = abs<float>(sum_y);
        }
    }
}

// Crop image
void crop(unsigned char* sobel_x, unsigned char* sobel_y, unsigned char* crop_x, unsigned char* crop_y, size_t rows, size_t cols, size_t base_col)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            size_t id = i * cols + j;
            size_t id_ = i * base_col + j;

            crop_x[id] = sobel_x[id_];
            crop_y[id] = sobel_y[id_];
        }
    }
}

// FIXME return features
void image_to_features(std::string path, int scale_factor, int pool_size, int postproc_size)
{
    // Using opencv to get image
    cv::Mat mat_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(mat_img.empty())
    {
        std::cerr << "Could not read the image" << std::endl;
        exit(1);
    }

    // No more cv::mat
    size_t cols = mat_img.cols;
    size_t rows = mat_img.rows;
    unsigned char* img = mat_img.data;
    unsigned char* sobel_x = (unsigned char*) calloc(cols * rows, sizeof(unsigned char));
    unsigned char* sobel_y = (unsigned char*) calloc(cols * rows, sizeof(unsigned char));

    // Check sobel images
    //cv::imwrite("toto_base.jpg", cv::Mat (rows, cols, CV_8UC1, img));

    _compute_features(img, sobel_x, sobel_y, rows, cols);

    // Check sobel images
    //cv::imwrite("totox.jpg", cv::Mat (rows, cols, CV_8UC1, sobel_x));
    //cv::imwrite("totoy.jpg", cv::Mat (rows, cols, CV_8UC1, sobel_y));

    // Calculate number of patchs
    size_t num_patchs_x = cols / pool_size;
    size_t num_patchs_y = rows / pool_size;
    size_t crop_cols = cols - cols % (pool_size * num_patchs_x);
    size_t crop_rows = rows - rows % (pool_size * num_patchs_y);

    unsigned char* crop_x = (unsigned char*) calloc(crop_cols * crop_rows, sizeof(unsigned char));
    unsigned char* crop_y = (unsigned char*) calloc(crop_cols * crop_rows, sizeof(unsigned char));

    crop(sobel_x, sobel_y, crop_x, crop_y, crop_rows, crop_cols, cols);

    //cv::imwrite("crop_totox.jpg", cv::Mat (crop_rows, crop_cols, CV_8UC1, crop_x));
    //cv::imwrite("crop_totoy.jpg", cv::Mat (crop_rows, crop_cols, CV_8UC1, crop_y));


    // Allocate table of size heigth/pool_size * weight/pool_size
    int tmp_response[num_patchs_y][num_patchs_x];
    memset(tmp_response, 0, sizeof(tmp_response));

    // fill features_patch
    for (int i = 0; i < crop_rows; ++i)
    {
        for (int j = 0; j < crop_cols; ++j)
        {
            size_t id = i * crop_cols + j;
            int f_x = crop_x[id];
            int f_y = crop_y[id];

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

    //cv::imwrite("mat_response_before_morpho.jpg", cv::Mat(num_patchs_y, num_patchs_x, CV_8UC1, response));

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

    //cv::imwrite("mat_response_morphed.jpg", cv::Mat(num_patchs_y, num_patchs_x, CV_8UC1, response));


    // Use threshold to activate patch
    const uint8_t threshold = max_value / 2;
    for (int i_patch_y = 0; i_patch_y < num_patchs_y; ++i_patch_y)
    {
        for (int i_patch_x = 0; i_patch_x < num_patchs_x; ++i_patch_x)
        {
            response[i_patch_y][i_patch_x] = 255 * (response[i_patch_y][i_patch_x] > threshold);
        }
    }

    cv::imwrite("Barcode.jpg", cv::Mat(num_patchs_y, num_patchs_x, CV_8UC1, response));

    // Free all allocations
    free(sobel_x);
    free(sobel_y);
    free(crop_x);
    free(crop_y);
}

template<typename T>
T abs(T value)
{
    if (value < 0)
        return -value;
    return value;
}
