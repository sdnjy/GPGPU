#include "processing.hpp"
#include <iostream>
#include <stdio.h>

//Compute feature GPU
__global__ void compute_features(unsigned char* img, unsigned char* sobel_x, unsigned char* sobel_y, 
                                const size_t width, const size_t height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_x = 0;
    float sum_y = 0;
    int size = width * height;

    while (id < size)
    {
        // Padding on border
        if (id <= width || id >= width * (height - 1) 
            || id % width == 0 || id % width == width - 1 )
        {
            // Do nothing
        }
        else
        {
            // Calculate sobel x
            sum_x += -1 * img[id - width - 1];
            sum_x += 1 * img[id - width + 1];
            sum_x += -2 * img[id - 1];
            sum_x += 2 * img[id + 1];
            sum_x += -1 * img[id + width - 1];
            sum_x += 1 * img[id + width + 1];

            // Calculate sobel y
            sum_y += -1 * img[id - width - 1];
            sum_y += -2 * img[id - width];
            sum_y += -1 * img[id - width + 1];
            sum_y += 1 * img[id + width - 1];
            sum_y += 2 * img[id + width];
            sum_y += 1 * img[id + width + 1];

            // Apply kernel
            sobel_x[id] = (sum_x > 0) ? sum_x : -sum_x;
            sobel_y[id] = (sum_y > 0) ? sum_y : -sum_y;
        }

        // Go to next pixel to treat by this thread
        id += blockDim.x * gridDim.x;
    }
}

// Crop the image using GPU
__global__ void crop(unsigned char* sobel_x, unsigned char* sobel_y, unsigned char* crop_x, unsigned char* crop_y, 
                    const size_t width, const size_t height, size_t diff_width)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int count_border = 0;
    int size = width * height;

    while (id < size)
    {
        int id_ = id + count_border * diff_width;

        // If true then we need to skip all remaining pixel of this line (crop occurs)
        if (id / width > count_border)
        {
            count_border += 1;
        }

        crop_x[id] = sobel_x[id_];
        crop_y[id] = sobel_y[id_];

        id += blockDim.x * gridDim.x;
    }
}

// Fill patches with mean features
__global__ void fill(int* int_response,  unsigned char* crop_x, unsigned char* crop_y, const int pool_size, 
                    const size_t width, const size_t height, const size_t num_patchs_y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        // Get i and j (not very efficient)
        int i = id / width;
        int j = id % width;
        int tmp_id = (i / pool_size) * num_patchs_y + (j / pool_size);

        int diff = crop_x[id] - crop_y[id];

        int_response[tmp_id] += diff;
        id += blockDim.x * gridDim.x;
    }
}

// Clip between 255 and 0
__global__ void clip(int* int_response, unsigned char* response, const size_t width, 
                    const size_t height, const int pool_size_squared)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        int current_value = int_response[id] / pool_size_squared;

        if (current_value <= 0)
            response[id] = 0;
        else if (current_value <= 255)
            response[id] = current_value;
        else
            response[id] = 255;

        id += blockDim.x * gridDim.x;
    }
}

// Apply morpho
__global__ void morpho(unsigned char* tmp_morpho, unsigned char* response, const size_t width,
                        const size_t height, bool dilation)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        // Max value
        uint8_t best = 255;

        if (dilation)
        {
            // Min value
            best = 0;
        }

        // Check that id is not first or last row and two first or two last coll
        if (id <= width || id >= width * (height - 1) 
            || id % width == 0 || id % width == 1 
            || id % width == width -1 || id % width == width - 2)
        {
            // Do nothing
        }
        else
        {
            // Check 3x5 kernel
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -2; j <= 2; ++j)
                {
                    int tmp_id = id + (i * width) + j;

                    if (dilation)
                    {
                        // Search maximum
                        if (response[tmp_id] > best)
                        {
                            best = response[tmp_id];
                        }
                    }
                    else
                    {
                        // Search minimum
                        if (response[tmp_id] < best)
                        {
                            best = response[tmp_id];
                        }
                    }
                    
                }
            }

            // Apply kernel result
            tmp_morpho[id] = best;
        }
        
        id += blockDim.x * gridDim.x;
    }
}

// Main function (will need refacto)
void image_to_features(std::string path, const int scale_factor, const int pool_size,
                        const int postproc_size, const std::string output_path)
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
    size_t img_size = cols * rows * sizeof(unsigned char);
    unsigned char *img = mat_img.data;
    unsigned char *sobel_x = (unsigned char*) calloc(cols * rows, sizeof(unsigned char));
    unsigned char *sobel_y = (unsigned char*) calloc(cols * rows, sizeof(unsigned char));

    // Malloc for GPU
    unsigned char *img_gpu;
    unsigned char *sobelx_gpu;
    unsigned char *sobely_gpu;
    
    cudaMalloc(&img_gpu, img_size);
    cudaMalloc(&sobelx_gpu, img_size);
    cudaMalloc(&sobely_gpu, img_size);

    cudaMemcpy(img_gpu, img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(sobelx_gpu, sobel_x, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(sobely_gpu, sobel_y, img_size, cudaMemcpyHostToDevice);

    // Check images
    cv::imwrite("toto_base.jpg", cv::Mat (rows, cols, CV_8UC1, img));

    // Initialize thread
    int blockSize = 10;
    int gridSize = 20;

    // Compute Sobel
    compute_features<<<gridSize, blockSize>>>(img_gpu, sobelx_gpu, sobely_gpu, rows, cols);
    cudaDeviceSynchronize();

    // Check sobel images
    cudaMemcpy(sobel_x, sobelx_gpu, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(sobel_y, sobely_gpu, img_size, cudaMemcpyDeviceToHost);
    cv::imwrite("totox.jpg", cv::Mat (rows, cols, CV_8UC1, sobel_x));
    cv::imwrite("totoy.jpg", cv::Mat (rows, cols, CV_8UC1, sobel_y));

    // Calculate number of patchs
    size_t num_patchs_x = cols / pool_size;
    size_t num_patchs_y = rows / pool_size;
    size_t crop_cols = cols - cols % (pool_size * num_patchs_x);
    size_t crop_rows = rows - rows % (pool_size * num_patchs_y);

    unsigned char* crop_x = (unsigned char*) calloc(crop_cols * crop_rows, sizeof(unsigned char));
    unsigned char* crop_y = (unsigned char*) calloc(crop_cols * crop_rows, sizeof(unsigned char));

    // Malloc for GPU
    img_size = crop_cols * crop_rows * sizeof(unsigned char);
    unsigned char *cropx_gpu;
    unsigned char *cropy_gpu;
    
    cudaMalloc(&cropx_gpu, img_size);
    cudaMalloc(&cropy_gpu, img_size);

    cudaMemcpy(cropx_gpu, crop_x, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cropy_gpu, crop_y, img_size, cudaMemcpyHostToDevice);

    // Crop images
    int diff_width = cols - crop_cols;
    crop<<<gridSize, blockSize>>>(sobelx_gpu, sobely_gpu, cropx_gpu, cropy_gpu, crop_rows, crop_cols, diff_width);
    cudaDeviceSynchronize();

    // Check crop
    cudaMemcpy(crop_x, cropx_gpu, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(crop_y, cropy_gpu, img_size, cudaMemcpyDeviceToHost);
    cv::imwrite("crop_totox.jpg", cv::Mat (crop_rows, crop_cols, CV_8UC1, crop_x));
    cv::imwrite("crop_totoy.jpg", cv::Mat (crop_rows, crop_cols, CV_8UC1, crop_y));

    int *int_response = (int*) calloc(num_patchs_y * num_patchs_x, sizeof(int));
    unsigned char *response = (unsigned char*) calloc(num_patchs_y * num_patchs_x, sizeof(unsigned char));
    
    // Malloc for GPU
    img_size = num_patchs_y * num_patchs_x;
    int *int_resp_gpu;
    unsigned char *resp_gpu;
    
    cudaMalloc(&int_resp_gpu, img_size * sizeof(int));
    cudaMalloc(&resp_gpu, img_size * sizeof(unsigned char));

    cudaMemcpy(int_resp_gpu, int_response, img_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(resp_gpu, response, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Fill patches with mean sobel differences
    fill<<<gridSize, blockSize>>>(int_resp_gpu,  cropx_gpu, cropy_gpu, pool_size, crop_cols, crop_rows, num_patchs_y);
    cudaDeviceSynchronize();

    // Clip values
    const int pool_size_squared = pool_size * pool_size;
    clip<<<gridSize, blockSize>>>(int_resp_gpu, resp_gpu, num_patchs_x,  num_patchs_y, pool_size_squared);
    cudaDeviceSynchronize();

    // Return response in CPU
    cudaMemcpy(response, resp_gpu, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Check clip
    cv::imwrite("clip_toto.jpg", cv::Mat (num_patchs_y, num_patchs_x, CV_8UC1, response));

    // Get max value
    int max_value = 0;
    for (int i = 0; i < num_patchs_y * num_patchs_x; ++i)
    {
        if (max_value < response[i])
        {
            max_value = response[i];
        }
    }

    // Allocate CPU and GPU temporary morpho
    unsigned char *tmp_morpho = (unsigned char*) calloc(num_patchs_y * num_patchs_x, sizeof(unsigned char));
    unsigned char *morpho_gpu;
    cudaMalloc(&morpho_gpu, img_size * sizeof(unsigned char));
    cudaMemcpy(morpho_gpu, tmp_morpho, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Apply morpho dilation
    morpho<<<gridSize, blockSize>>>(morpho_gpu, resp_gpu, num_patchs_x,  num_patchs_y, true);
    cudaDeviceSynchronize();

    // Apply morpho erosion
    morpho<<<gridSize, blockSize>>>(resp_gpu, morpho_gpu, num_patchs_x,  num_patchs_y, false);
    cudaDeviceSynchronize();

    // Get response back to CPU
    cudaMemcpy(response, resp_gpu, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Check final barcode prediction
    cv::imwrite("barcode.jpg", cv::Mat (num_patchs_y, num_patchs_x, CV_8UC1, response));

    // Free all allocations
    free(sobel_x);
    free(sobel_y);
    free(crop_x);
    free(crop_y);
    free(int_response);
    free(response);
    free(tmp_morpho);

    cudaFree(img_gpu);
    cudaFree(sobelx_gpu);
    cudaFree(sobely_gpu);
    cudaFree(cropx_gpu);
    cudaFree(cropy_gpu);
    cudaFree(int_resp_gpu);
    cudaFree(resp_gpu);
    cudaFree(morpho_gpu);
}