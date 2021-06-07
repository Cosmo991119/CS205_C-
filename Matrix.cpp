//#include <iostream>
//#include "Matrix.h"
//
//namespace MATRIX {
//    template<typename T>
//    T Matrix<T>::max(int row_i,int row_f,int col_i,int col_f) {
//        //maybe check size first?
//        T MAX = Mat[row_i][col_i];
//        for (int i = row_i; i < row_f+1; i++) {
//            for (int j = col_i; j < col_f+1; j++) {
//                if (Mat[i][j] > MAX)
//                    MAX = Mat[i][j];
//            }
//        }
//        return MAX;
//    }
//
//    template<typename T>
//    T Matrix<T>::min() {
//        //maybe check size first?
//        T MIN = Mat[0][0];
//        for (int i = 0; i < Cols; i++) {
//            for (int j = 0; j < Rows; j++) {
//                if (Mat[i][j] < MIN)
//                    MIN = Mat[i][j];
//            }
//        }
//        return MIN;
//    }
//
//    template<typename T>
//    T Matrix<T>::sum() {
//        //maybe check size first?
//        T SUM; //maybe need initialize?
//        for (int i = 0; i < Cols; i++) {
//            for (int j = 0; j < Rows; j++) {
//                if (Mat[i][j] < SUM)
//                    SUM += Mat[i][j];
//            }
//        }
//        return SUM;
//    }
//
//    template<typename T>
//    void Matrix<T>::show() {
//        for(int i=0;i<Rows;i++) {
//            std::cout<<"[ ";
//            for (int j = 0; j < Cols; j++) {
//                std::cout << Mat[i][j]<<' ';
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<"]";
//    }
//};
//
//
//
