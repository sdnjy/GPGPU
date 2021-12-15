#include "processing.hpp"
#include <iostream>
#include <stdio.h>

//Compute feature GPU
__global__ void compute_features(unsigned char* img, unsigned char* sobel_x, unsigned char* sobel_y, 
                                const size_t width, const size_t height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        float sum_x = 0;
        float sum_y = 0;

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
    int size = width * height;

    while (id < size)
    {
        int id_ = id + (id / width) * diff_width;

        // If true then we need to skip all remaining pixel of this line (crop occurs)

        crop_x[id] = sobel_x[id_];
        crop_y[id] = sobel_y[id_];

        id += blockDim.x * gridDim.x;
    }
}

// Fill patches with mean features
__global__ void fill(int* int_response,  unsigned char* crop_x, unsigned char* crop_y, const int pool_size, 
                    const size_t width, const size_t height, const size_t image_width)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        int sum = 0;

        int base = id / width * pool_size * image_width 
            + pool_size * (id % width);

        for (int i = 0; i < pool_size; ++i)
        {
            for (int j = 0; j < pool_size; ++j)
            {
                int id_ = (base + i) + j * image_width;

                sum += (int)crop_x[id_] - (int)crop_y[id_];
            }
        }

        int_response[id] = sum;
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
__global__ void morpho_erosion(unsigned char* tmp_morpho, unsigned char* response, const size_t width,
                        const size_t height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        // Max value
        uint8_t best = 255;

        // Check that id is not first or last row and two first or two last coll
        if (!(id <= width || id >= width * (height - 1) 
            || id % width == 0 || id % width == 1 
            || id % width == width -1 || id % width == width - 2))
        {
            // Check 3x5 kernel
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -2; j <= 2; ++j)
                {
                    int tmp_id = id + (i * width) + j;

                    // Search minimum
                    if (response[tmp_id] < best)
                        best = response[tmp_id];
                }
            }

            // Apply kernel result
            tmp_morpho[id] = best;
        }
        
        id += blockDim.x * gridDim.x;
    }
}

// Apply morpho
__global__ void morpho_dilation(unsigned char* tmp_morpho, unsigned char* response, const size_t width,
                        const size_t height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        // Max value
        uint8_t best = 0;

        // Check that id is not first or last row and two first or two last coll
        if (!(id <= width || id >= width * (height - 1) 
            || id % width == 0 || id % width == 1 
            || id % width == width -1 || id % width == width - 2))
        {
            // Check 3x5 kernel
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -2; j <= 2; ++j)
                {
                    int tmp_id = id + (i * width) + j;

                    // Search maximum
                    if (response[tmp_id] > best)
                        best = response[tmp_id];
                }
            }

            // Apply kernel result
            tmp_morpho[id] = best;
        }
        
        id += blockDim.x * gridDim.x;
    }
}


__global__ void threshold(unsigned char *response, uint8_t threshold, const size_t width, const size_t height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        response[id] = 255 * (response[id] > threshold);
        id += blockDim.x * gridDim.x;
    }
}

__global__ void resize(unsigned char *response, unsigned char *crop_x, const size_t width, const size_t height, 
                        const size_t num_patchs_x, const size_t pool_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    while (id < size)
    {
        // Get i and j (not very efficient)
        int i = id / width;
        int j = id % width;
        int patch_id = (i / pool_size) * num_patchs_x + (j / pool_size);

        crop_x[id] = response[patch_id];

        id += blockDim.x * gridDim.x;
    }
}


__global__ void myMax(const unsigned char* input, const int size, int* maxOut)
{
    __shared__ int sharedMax;
  
    if (0 == threadIdx.x)
    {
        sharedMax = 0;
    }
  
    __syncthreads();
  
    int localMax = 0;
  
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        const int val = input[i];
  
        if (localMax < val)
        {
            localMax = val;
        }
    }
  
    atomicMax(&sharedMax, localMax);
  
    __syncthreads();
  
    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
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
    const size_t cols = mat_img.cols;
    const size_t rows = mat_img.rows;
    size_t img_size = cols * rows * sizeof(unsigned char);
    unsigned char *img = mat_img.data;

    // Malloc for GPU
    unsigned char *img_gpu;
    unsigned char *sobelx_gpu;
    unsigned char *sobely_gpu;
    
    cudaMalloc(&img_gpu, img_size);
    cudaMalloc(&sobelx_gpu, img_size);
    cudaMalloc(&sobely_gpu, img_size);

    cudaMemcpy(img_gpu, img, img_size, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Initialize thread
    int blockSize = deviceProp.maxThreadsDim[1];
    int gridSize = mat_img.rows < deviceProp. maxGridSize[1] ? mat_img.rows : deviceProp. maxGridSize[1];

    // Compute Sobel
    compute_features<<<gridSize, blockSize>>>(img_gpu, sobelx_gpu, sobely_gpu, cols, rows);
    cudaDeviceSynchronize();

    // Calculate number of patchs
    size_t num_patchs_x = cols / pool_size;
    size_t num_patchs_y = rows / pool_size;

    // Malloc for GPU
    img_size = num_patchs_y * num_patchs_x;
    int* int_resp_gpu;
    unsigned char* resp_gpu;
    
    cudaMalloc(&int_resp_gpu, img_size * sizeof(int));
    cudaMalloc(&resp_gpu, img_size * sizeof(unsigned char));

    // Fill patches with mean sobel differences
    fill<<<gridSize, blockSize>>>(int_resp_gpu,  sobelx_gpu, sobely_gpu, pool_size, num_patchs_x, num_patchs_y, cols);
    cudaDeviceSynchronize();

    // Clip values
    const int pool_size_squared = pool_size * pool_size;
    clip<<<gridSize, blockSize>>>(int_resp_gpu, resp_gpu, num_patchs_x,  num_patchs_y, pool_size_squared);
    cudaDeviceSynchronize();

    // Get max value
    int max_value = 0;
    int* d_max_value;
    cudaMalloc(&d_max_value, sizeof(int));
    cudaMemcpy(d_max_value, &max_value, sizeof(int), cudaMemcpyHostToDevice);

    myMax<<<gridSize, blockSize>>>(resp_gpu, img_size, d_max_value);
    cudaDeviceSynchronize();

    cudaMemcpy(&max_value, d_max_value, sizeof(int), cudaMemcpyDeviceToHost);

    // Allocate CPU and GPU temporary morpho
    unsigned char *tmp_morpho = (unsigned char*) calloc(num_patchs_y * num_patchs_x, sizeof(unsigned char));
    unsigned char *morpho_gpu;
    cudaMalloc(&morpho_gpu, img_size * sizeof(unsigned char));

    // Apply morpho dilation
    morpho_dilation<<<gridSize, blockSize>>>(morpho_gpu, resp_gpu, num_patchs_x,  num_patchs_y);
    cudaDeviceSynchronize();

    // Apply morpho erosion
    morpho_erosion<<<gridSize, blockSize>>>(resp_gpu, morpho_gpu, num_patchs_x,  num_patchs_y);
    cudaDeviceSynchronize();


    // Apply threshold
    const uint8_t thresh = max_value / 2;
    threshold<<<gridSize, blockSize>>>(resp_gpu, thresh, num_patchs_x,  num_patchs_y);
    cudaDeviceSynchronize();

    resize<<<gridSize, blockSize>>>(resp_gpu, sobelx_gpu, cols, rows, num_patchs_x, pool_size);
    cudaDeviceSynchronize();


    unsigned char *res = (unsigned char*) calloc(cols * rows, sizeof(unsigned char));
    // Get response back to CPU
    cudaMemcpy(res, sobelx_gpu, cols * rows * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Check final barcode prediction
    cv::imwrite(output_path, cv::Mat (rows, cols, CV_8UC1, res));

    // Free all allocations
    free(res);
    free(tmp_morpho);

    cudaFree(d_max_value);
    cudaFree(img_gpu);
    cudaFree(sobelx_gpu);
    cudaFree(sobely_gpu);
    cudaFree(int_resp_gpu);
    cudaFree(resp_gpu);
    cudaFree(morpho_gpu);
}