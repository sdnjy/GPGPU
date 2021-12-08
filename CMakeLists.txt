cmake_minimum_required(VERSION 3.10)

project(GPGPU)


file(GLOB main_SRC
    "src/*.cpp"
)

add_executable(main ${main_SRC})


find_package( OpenCV REQUIRED )

set(OPENCV_BUILD_3RDPARTY_LIBS "ON")
set(WITH_GTK "ON")
set(OPENCV_INCLUDE_DIR /usr/local/include/opencv4/)
set(OPENCV_LIB_DIR /usr/local/lib/)


message(STATUS "OpenCV library path: ${OPENCV_LIB_DIR}")

# set include directory
target_include_directories(main PUBLIC "${OPENCV_INCLUDE_DIR}")

# set library directory
target_link_directories(main PUBLIC "${OPENCV_LIB_DIR}")

# link specific object files we need
target_link_libraries(main opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui)