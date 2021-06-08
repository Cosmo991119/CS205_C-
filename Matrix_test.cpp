//
// Created by Penson on 2021/6/8.
//
#include <iostream>
#include "Matrix.h"
using namespace std;
using namespace MATRIX;
int main(){
    double mat[2][2] = {{1,2},{3,4}};
    //int (*p)[2] = mat;
    Matrix<double> m(2, 2, *mat);
    try {
        cout << "The matrix: " << endl;
        m.ShowMatrix();
        cout << "max: " << m.max(0, 1, 0, 1) << endl;
        cout << "min: " << m.min(0, 1, 0, 1) << endl;
        cout << "sum: " << m.sum(0, 1, 0, 1) << endl;
        cout << "avg: " << m.avg(0, 1, 0, 1) << endl;
        
    }catch(const char* msg) {
        cerr << msg << endl;
    }
    return 0;
}
