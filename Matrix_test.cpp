//
// Created by Penson on 2021/6/8.
//
#include <iostream>
#include "Matrix.h"
using namespace std;
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
        cout << "The square matrix: " << endl;
        m.ShowMatrix();
        cout << "max: " << m.max(0, 1, 0, 1) << endl;
        cout << "min: " << m.min(0, 1, 0, 1) << endl;
        cout << "sum: " << m.sum(0, 1, 0, 1) << endl;
        cout << "avg: " << m.avg(0, 1, 0, 1) << endl;
        cout<<"Copy Matrix: "<<endl;
        m1=m;
        m1.ShowMatrix();
        cout<<"Add result: "<<endl;
        m = m+m;
        m.ShowMatrix();
        cout<<"Minus result: "<<endl;
        m = m-m1;
        m.ShowMatrix();
        cout<<"* result: "<<endl;
        m = m*m1;
        m.ShowMatrix();
//        cout<<"Remainder: "<<endl;
//        (m.remainder(0,1)).ShowMatrix();
        cout<<"Det: "<<endl;
        cout<<m.Det()<<endl;
        cout<<"Det: "<<endl;
        double det[3][3] = {{3,5,7},{7,11,13},{2,4,7}};
        Matrix<double> det_m(3, 3, *det);
        det_m.ShowMatrix();
        (det_m.Remainder(0,0)).ShowMatrix();
        cout<<det_m.Det()<<endl;

//        cout << "The rectangle matrix: " << endl;
//        rec_m.ShowMatrix();
//        cout << "max: " << rec_m.max(0, 1, 0, 2) << endl;
//        cout << "min: " << rec_m.min(0, 1, 0, 2) << endl;
//        cout << "sum: " << rec_m.sum(0, 1, 0, 2) << endl;
//        cout << "avg: " << rec_m.avg(0, 1, 0, 2) << endl;
//        cout<<"Matrix tran: "<<endl;
//        Matrix<double> rec_m_tran(3,2);
//        rec_m_tran=rec_m.tran();
//        rec_m_tran.ShowMatrix();
//        cout<<"Copy Matrix: "<<endl;
//        rec_m1=rec_m;
//        rec_m1.ShowMatrix();
//        cout<<"Add result: "<<endl;
//        rec_m = rec_m+rec_m;
//        rec_m.ShowMatrix();
//        cout<<"Minus result: "<<endl;
//        rec_m = rec_m-rec_m1;
//        rec_m.ShowMatrix();
//        cout<<"Multiply result: "<<endl;
//        (rec_m*rec_m1.tran()).ShowMatrix();
    }catch(const char* msg) {
        cerr << msg << endl;
    }
    return 0;
}
