#include "processing.hpp"
#include "img_features.hpp"
#include "array.hpp"


ImgFeatures<uint8_t>* _compute_features(double** img, const size_t width, const size_t height)
{
    const int n_filters = 2;

    ImgFeatures<uint8_t>* img_features = new ImgFeatures<uint8_t>(width, height);
    Array<uint8_t>* img_feature_x = img_features->img_feature_x;
    Array<uint8_t>* img_feature_y = img_features->img_feature_y;
    
    
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
                    sum_x += sobelx[i][j] * img[ii + i - 1][jj + j - 1];
                    sum_y += sobely[i][j] * img[ii + i - 1][jj + j - 1];
                }
            }
            // sobel x
            img_feature_x->at(ii, jj) = abs<float>(sum_x);
            // sobel y
            img_feature_y->at(ii, jj) = abs<float>(sum_y);
        }
    }

    return img_features;
}

template<typename T>
T abs(T value)
{
    if (value < 0)
        return -value;
    return value;
}
