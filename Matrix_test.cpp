//
// Created by Penson on 2021/6/8.
//
#include <iostream>
#include "Matrix.h"
using namespace std;
using namespace MATRIX;
int main(){
    int mat[2][2] = {{1,2},{3,4}};
    //int (*p)[2] = mat;
    Matrix<int> m(2,2, *mat);
    cout<<"The matrix: "<<endl;
    m.show();
    cout<<"max: "<<m.max(0,1,0,1)<<endl;
    cout<<"min: "<<m.min(0,1,0,1)<<endl;
    cout<<"sum: "<<m.sum(0,1,0,1)<<endl;
    return 0;
}
