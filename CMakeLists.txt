cmake_minimum_required(VERSION 3.10)
project(Matrix_Operation)

set(CMAKE_CXX_STANDARD 11)

add_executable(Matrix_Operation Matrix.cpp Matrix.h Matrix_test.cpp)
add_executable(Matrix_adtest Matrix.cpp Matrix.h matrix_advanced_test.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(Matrix_Operation ${OpenCV_LIBS})
    TARGET_LINK_LIBRARIES(Matrix_adtest ${OpenCV_LIBS})

ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)

