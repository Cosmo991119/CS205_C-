//
// Created by Penson on 2021/6/8.
//
#include <iostream>
#include "Matrix.h"

using namespace MATRIX;

int main(){
    double mat[2][2] = {{1,2},{3,4}};
    double rec_mat[2][3] = {{1,2,3},{1,3,2}};
    //int (*p)[2] = mat;
    Matrix<double> m(2, 2, *mat);
    Matrix<double> m1(2, 2);

    Matrix<double> rec_m(2, 3, *rec_mat);
    Matrix<double> rec_m1(2, 3);

    try {
        cout<<"Show m:"<<endl;
        m.ShowMatrix();
        cout<<"inverse of m:"<<endl;
        m1=m.inverse();
        m1.ShowMatrix();

//
//        cout<<"Change item:"<<endl;
//        m.ChangeItem(0,0,10);
//        m.ShowMatrix();






    }catch(const char* msg) {
        cerr << msg << endl;
    }
    return 0;
}
