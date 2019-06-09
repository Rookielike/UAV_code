cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11 -lpthread -o0 -g)
project(UAV)

find_package(OpenCV REQUIRED)
include_directories(/home/shunya/caffe/include
             /usr/include
             /usr/local/include
            ${OpenCV_INCLUDE_DIRS}
            )

aux_source_directory(. DIR_SRCS)

add_executable(Demo ${DIR_SRCS})

target_link_libraries(Demo 
     /home/shunya/caffe/build/lib/libcaffe.so
#     /usr/local/lib/libopencv_core.so 
 #    /usr/local/lib/libopencv_imgproc.so
 #    /usr/local/lib/libopencv_imgcodecs.so 
 #   /usr/local/lib/libopencv_highgui.so
#    /usr/local/lib/libopencv_videoio.so
    /usr/lib/aarch64-linux-gnu/libpthread.so
    /usr/lib/aarch64-linux-gnu/libgflags.so    
    /usr/lib/aarch64-linux-gnu/libglog.so   
    /usr/lib/aarch64-linux-gnu/libprotobuf.so 
    /usr/lib/aarch64-linux-gnu/libboost_system.so
    ${OpenCV_LIBS}
    )


