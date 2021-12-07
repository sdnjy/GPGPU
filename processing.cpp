#include <stddef.h>
#include <cstdint>


uint8_t*** _compute_features(double** img, const size_t width, const size_t height)
{
    int n_filters = 2;
    uint8_t*** img_features = npZeros<uint8_t>(width, height, n_filters);
    
    float sobelx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float sobely[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (size_t ii = 1; ii < width - 1; ++ii)
    {
        for (size_t jj = 1; jj < height - 1; ++jj)
        {
            float sum_x = 0;
            float sum_y = 0;

            for 


            patch = img[ii-1:ii+2,jj-1:jj+2];
            // sobel x
            img_features[ii,jj,0] = abs(sum_x);
            // sobel y
            img_features[ii,jj,1] = abs(sum_y);
        }
    }

    return img_features;
}


template<typename T>




template<typename T>
T*** npZeros(const size_t width, const size_t height, const size_t dim3)
{
    T*** matrice  = new T[width][height][dim3]{0};
    //std::fill(*matrice, *matrice + length * height, 0);
    return matrice;
}

template<typename T>
T abs(T value)
{
    if (value < 0)
        return -value;
    return value;
}

//pour compute_feature, sobelx * patch est de dimension (3, 3)
template<typename T>
T npSum(const T** matrice, const size_t width, const size_t height)
{
    T sum = 0;
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            sum += T[i][j];
        }
    }
    return sum;
}

