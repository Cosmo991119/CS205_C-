cmake_minimum_required(VERSION 3.10)
project(Matrix_Operation)

set(CMAKE_CXX_STANDARD 11)


#
#set(OpenCV_DIR }F:/opencv/mingw-build)
set(CMAKE_PREFIX_PATH F:/opencv/mingw-build)
##
##set(OpenCV_DIR"${CMAKE_SOURCE_DIR}F:\\opencv\\mingw-build")
#
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(Matrix_Operation Matrix.cpp Matrix.h Matrix_test.cpp)
add_executable(Matrix_adtest Matrix.cpp Matrix.h matrix_advanced_test.cpp)

FIND_PACKAGE(OpenCV 4.0.1 REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(Matrix_Operation ${OpenCV_LIBS})

ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)

