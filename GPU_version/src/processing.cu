#include processing.hpp
#include <iostream>
#include <stdio.h>


//Compute feature GPU
__global__ void compute_features(unsigned char* img, unsigned char* sobel_x, unsigned char* sobel_y, 
                                const size_t width, const size_t height, const size_t thread_block_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_x = 0;
    float sum_y = 0;

    while (id < width * height)
    {
        // Padding on border
        if (id <= width || id % width == 0)
        {
            // Do nothing
        }
        else
        {
            // Calculate sobel x
            sum_x += -1 * img[id - width - 1]
            sum_x += 1 * img[id - width + 1]
            sum_x += -2 * img[id - 1]
            sum_x += 2 * img[id + 1]
            sum_x += -1 * img[id + width - 1]
            sum_x += 1 * img[id + width + 1]

            // Calculate sobel y
            sum_y += -1 * img[id - width - 1]
            sum_y += -2 * img[id - width]
            sum_y += -1 * img[id - width + 1]
            sum_y += 1 * img[id + width - 1]
            sum_y += 2 * img[id + width]
            sum_y += 1 * img[id + width + 1]

            // Apply kernel
            sobel_x[id] = (sum_x > 0) ? sum_x : 0;
            sobel_y[id] = (sum_y > 0) ? sum_y : 0;
        }

        // Go to next pixel to treat by this thread
        id += thread_block_size;
    }

}

// Crop the image using GPU
__global__ void crop(unsigned char* sobel_x, unsigned char* sobel_y, unsigned char* crop_x, unsigned char* crop_y, 
                    const size_t width, const size_t height, size_t diff_width, const size_t thread_block_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int count_border = 0

    while (id < rows * cols)
    {
        id += thread_block_size;
        id_ = id + count_border * diff_width;

        // If true then we need to skip all remaining pixel of this line (crop occurs)
        if (id % width > count_border)
        {
            count_border += 1;
        }

        crop_x[id] = sobel_x[id_];
        crop_y[id] = sobel_y[id_];
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
    size_t img_size = cols * rows * sizeof(unsigned char);
    size_t cols = mat_img.cols;
    size_t rows = mat_img.rows;
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

    // Check sobel images
    cv::imwrite("toto_base.jpg", cv::Mat (rows, cols, CV_8UC1, img));

    // Initialize thread
    int blockSize = 10;
    int gridSize = 20;
    int thread_block_size = blockSize * gridSize;

    compute_features<<<gridSize, blockSize>>>(img_gpu, sobelx_gpu, sobely_gpu, rows, cols, thread_block_size);

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
    img_size = crop_cols * crop_rows;
    unsigned char *cropx_gpu;
    unsigned char *cropy_gpu;
    
    cudaMalloc(&cropx_gpu, img_size);
    cudaMalloc(&cropy_gpu, img_size);

    cudaMemcpy(cropx_gpu, crop_x, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cropy_gpu, crop_y, img_size, cudaMemcpyHostToDevice);

    int diff_width = cols - crop_cols;
    crop<<<gridSize, blockSize>>>(sobelx_gpu, sobely_gpu, cropx_gpu, cropy_gpu, crop_rows, crop_cols, diff_width, thread_block_size);

    // Check crop
    cudaMemcpy(crop_x, cropx_gpu, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(crop_y, cropy_gpu, img_size, cudaMemcpyDeviceToHost);
    cv::imwrite("crop_totox.jpg", cv::Mat (crop_rows, crop_cols, CV_8UC1, crop_x));
    cv::imwrite("crop_totoy.jpg", cv::Mat (crop_rows, crop_cols, CV_8UC1, crop_y));

    // Free all allocations
    free(sobel_x);
    free(sobel_y);
    free(crop_x);
    free(crop_y);

    cudaFree(img_gpu);
    cudaFree(sobelx_gpu);
    cudaFree(sobely_gpu);
    cudaFree(cropx_gpu);
    cudaFree(cropy_gpu);
}