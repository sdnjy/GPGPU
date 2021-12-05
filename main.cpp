#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>




int main() {
    
    //std::string image_path = cv::samples::findFile("collective_database/PXL_20211101_175643604-GT.png");
    cv::Mat img = cv::imread("../collective_database/PXL_20211101_175643604-GT.png", cv::IMREAD_COLOR);
    //cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }
    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display window", img);
    //cv::waitKey(0);


    int scale_factor = 1;
    int pool_size = 31;
    int postproc_size = 5;



    return 0;
}