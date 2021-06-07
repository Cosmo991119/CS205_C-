cmake_minimum_required(VERSION 3.10)
project(Matrix_Operation)

set(CMAKE_CXX_STANDARD 11)

add_executable(Matrix_test Matrix.cpp Matrix.h Matrix_test.cpp)