//
// Created by Penson on 2021/6/8.
//
#include <iostream>
#include "Matrix.h"

using namespace MATRIX;

int main() {
    double mat[2][2] = {{1, 2},
                        {3, 4}};

    double rec_mat[2][3] = {{1, 2, 3},
                            {1, 3, 2}};
    double conv[3][3] = {{1, 2, 3},
                         {1, 2, 3},
                         {1, 2, 3}};
    //int (*p)[2] = mat;
    Matrix<double> m(2, 2, *mat);
    Matrix<double> m1(2, 2);

    Matrix<double> rec_m(2, 3, *rec_mat);
    Matrix<double> rec_m1(2, 3);

    Matrix<double> fliter(3, 3, *conv);

    try {
        cout << "Show m:" << endl;
        m.ShowMatrix();
        cout << "inverse of m:" << endl;
        m1 = m.inverse();
        m1.ShowMatrix();

        cout << "reshape of m:" << endl;
        (m.reshape(1,4)).ShowMatrix();

        cout << "convolution(same) of rec_mat:" << endl;
        rec_m1 = rec_m.conv_same(fliter);
        rec_m1.ShowMatrix();


        cout << "Slice:" << endl;
        double conv[3][3];
        Matrix<double> slice_a(2, 2);
        slice_a = rec_m.slice(0, 2, 0, 2);
        slice_a.ShowMatrix();

        cout << "Slice: slice rows" << endl;
        Matrix<double> slice_b(1, 3);
        slice_b = rec_m.slice(0, 1, -1);
        slice_b.ShowMatrix();

        cout << "Slice: slice cols" << endl;
        Matrix<double> slice_c(2, 2);
        slice_c = rec_m.slice(0, 2, -2);
        slice_c.ShowMatrix();

        cout<<"Change item:"<<endl;
        m.ChangeItem(0,0,10);
        m.ShowMatrix();

        auto mat1 = imread("./water.jpg", 0);
        auto vec = Mat2Vec(mat1);
        auto mat2 = vec.Vec2Mat();
        imshow("image", mat2);
        waitKey(0);
        destroyAllWindows();

    } catch (const char *msg) {
        cerr << msg << endl;
    }
    return 0;
}
