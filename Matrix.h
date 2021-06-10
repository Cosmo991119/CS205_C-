//
// Created by zhang on 2021/5/15.
//

#ifndef WSL_MATRIX_H
#define WSL_MATRIX_H

#include <complex>
//opencv
#include <math.h>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

//structure
namespace MATRIX {//lab9
    using namespace std;
    using namespace cv;


    template<typename T>
    class Matrix {
    private:
        int Cols;
        int Rows;
        int size;
        //other properity
        T **Matrixs;//lab12, a pointer to point the matrix
    public:
        //hbx：我直接改了，先rows再cols
        Matrix(int rows, int cols) : Rows(rows), Cols(cols), size(rows * cols) {
            T a = '\0';
            Matrixs = new T *[rows];
            for (int i = 0; i < rows; i++) {
                Matrixs[i] = new T[cols];
                for (int j = 0; j < cols; j++)
                    Matrixs[i][j] = a;//得显式初始化一下，不然乘法会不正常。（我也不知道为什么但实践得）
            }
        };

        Matrix(int rows, int cols, T *p) : Rows(rows), Cols(cols), size(rows * cols) {
            Matrixs = new T *[rows];
            T *pr = p;
            for (int i = 0; i < rows; i++) {
                Matrixs[i] = new T[cols];
                for (int j = 0; j < cols; j++) {
                    Matrixs[i][j] = *(pr++);
                }
            }
        }; //one dimension array

        Matrix(int rows, int cols, T **p) : Rows(rows), Cols(cols), size(rows * cols) {
            Matrixs = new T *[rows];
            for (int i = 0; i < rows; i++) {
                Matrixs[i] = new T[cols];
                for (int j = 0; j < cols; j++) {
                    Matrixs[i][j] = p[i][j];
                }
            }
        }; //two dimension array

        Matrix(const Matrix &mat) : Rows(mat.Rows), Cols(mat.Cols), size(mat.size) {
            Matrixs = new T *[mat.Rows];

            for (int i = 0; i < mat.Rows; i++) {
                Matrixs[i] = new T[mat.Rows];
                for (int j = 0; j < mat.Cols; j++) {
                    Matrixs[i][j] = mat.Matrixs[i][j];
                }
            }
        };//copy constructor, the size maybe different



        ~Matrix() {
            for (int i = 0; i < Rows; i++) {
                delete[] Matrixs[i];
            }
            delete[] Matrixs;
        }// deconstructor

        //get some values of matrix
        int GetCols() const {
            return Cols;
        };

        int GetRows() const {
            return Rows;
        };

        int GetSize() const {
            return size;
        };

        void ShowMatrix() {
            for (int i = 0; i < Rows; i++) {
                cout << "[ ";
                for (int j = 0; j < Cols; j++) {
                    cout << Matrixs[i][j] << " ";
                }
                cout << "]" << endl;
            }

        };


        T **GetMatrix() {
            return Matrixs;
        };

        T GetItem(int col, int row) {
            return Matrixs[col][row];
        };

        void ChangeItem(int col, int row, T val) {
            Matrixs[col][row] = val;
        };

        T *GetColum(int col) {
            T *arr[Rows];
            for (int i = 0; i < Rows; ++i) {
                arr[i] = Matrixs[col][i];
            }
            return arr;
        };

        T *GetRow(int row) {
            T *arr[Cols];
            for (int i = 0; i < Cols; ++i) {
                arr[i] = Matrixs[i][row];
            }
            return arr;
        };

        //find special value
        //column/row initial, column/row final(0~Cols-1)
        //可做局部比大小、加和
        T max(int row_i, int row_f, int col_i, int col_f) {
            T MAX = Matrixs[row_i][col_i];
            for (int i = row_i; i < row_f + 1; i++) {
                for (int j = col_i; j < col_f + 1; j++) {
                    if (Matrixs[i][j] > MAX)
                        MAX = Matrixs[i][j];
                }
            }
            return MAX;
        }

        T min(int row_i, int row_f, int col_i, int col_f) {
            T MIN = Matrixs[row_i][col_i];
            for (int i = row_i; i < row_f + 1; i++) {
                for (int j = col_i; j < col_f + 1; j++) {
                    if (Matrixs[i][j] < MIN)
                        MIN = Matrixs[i][j];
                }
            }
            return MIN;
        }

        T sum(int row_i, int row_f, int col_i, int col_f) {
            T SUM = '\0'; //maybe need initialize?
            for (int i = row_i; i < row_f + 1; i++) {
                for (int j = col_i; j < col_f + 1; j++) {
                    SUM += Matrixs[i][j];
                }
            }
            return SUM;
        }


        T avg(int row_i, int row_f, int col_i, int col_f) {
            return sum(row_i, row_f, col_i, col_f) / ((row_f - row_i + 1) * (col_f - col_i + 1));
        }

//        std::complex<double> avg(int row_i, int row_f, int col_i, int col_f){
//
//        }
        Matrix<T> tran() {
            Matrix TRAN(Cols, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    TRAN.Matrixs[j][i] = Matrixs[i][j];
            return TRAN;
        }

        T trace() {
            if (Cols != Rows)
                throw "\033[31msquare Error: \033[0mthe matrixs must be a square matrix.";
            T TRACE;
            for (int i = 0; i < Rows; i++)
                TRACE += Matrixs[i][i];
            return TRACE;
        }

        //要分清运行时exception和编译器报错的区别。后者可不是exception，是直接提醒使用者就是不能用。
        Matrix<T> MatrixConj() {
            Matrix result(Rows, Cols);
            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Cols; j++) {
                    result.Matrixs[i][j] = conj(Matrixs[i][j]);
                }
            }
            return result;
        }

        Matrix<T> Remainder(int i, int j) {
            Matrix<T> remain_mat(Rows - 1, Cols - 1);
            int m_i = 0;
            for (int r_i = 0; r_i < remain_mat.Rows; r_i++) {
                int m_j = 0;
                for (int r_j = 0; r_j < remain_mat.Cols; r_j++) {
                    if (m_j == j)
                        m_j++;
                    if (m_i == i)
                        m_i++;
                    remain_mat.Matrixs[r_i][r_j] = Matrixs[m_i][m_j];
                    m_j++;
                }
                m_i++;
                //remain_mat.ShowMatrix();
            }
            //remain_mat.ShowMatrix();
            return remain_mat;

        }

        T Det() {
            if (Cols != Rows)
                throw "\033[31msquare Error: \033[0mthe matrixs must be a square matrix.";
            T det_val;
            //Matrix<T> mat_mid(Rows,Cols);
            if (Rows == 2)
                return Matrixs[0][0] * Matrixs[1][1] - Matrixs[0][1] * Matrixs[1][0];
            else {
                for (int i = 0; i < Rows; i++)
                    det_val += pow(-1, i) * Matrixs[i][0] * Remainder(i, 0).Det();
            }
            return det_val;

        }

        void QR_fact(Matrix<T> *A, Matrix<T> *Q, Matrix<T> *R) {
            T sum;
            for(int i=0;i<Rows;i++){
                sum += pow(A->Matrixs[i][0],2);
            }
            for(int i=0;i<Cols;i++){
                Q->Matrixs[i][0] = A->Matrixs[i][0]/sqrt(sum);
            }
            for (int i = 1; i < Rows; i++) {
                for(int j=0;j<Cols;j++){
                    Q->Matrixs[j][i] = A->Matrixs[j][i];
                }
                for(int j=0;j<i;j++){
                    for(int k=0;k<i;k++){
                        
                    }
                }
            }
        }

        Matrix<T> eigV(int times = 100) {
            if (Cols != Rows)
                throw "\033[31mSize Error: \033[0mThe matrix is not square matrix!";//need to do throw exception
            if (Det() == 0)
                throw "Cannot use QR method. Not a Full-rank Matrix!";

            Matrix<T> Q(Rows, Cols);
            Matrix<T> R(Rows, Cols);
            Matrix<T> A(Rows, Cols, *this);
            for (int t = 0; t < times; t++) {
                QR_fact
            }
        }

//        T EigenValue() {
//
//
//
//        }

        T EigenVector();//vector

        //advanced operator implement
        void reshape(int cols, int rows) {
            if (cols * rows != Cols * Rows) {
                throw "\033[31mSize Error: \033[0mthe matrixs must has same size.";//need to do throw exception
            }

            Cols = cols;
            Rows = rows;

        };

        //Gauss-Jordan Elimination Method
        Matrix<T> inverse() {
            if (Cols != Rows) {
                throw
                        "\033[31msquare Error: \033[0mthe matrixs must be a square matrix.";
            }

            if (Cols == 1) {//1*1
                T *pt;
                T ans[1];
                pt = ans;
                ans[0] = 1 / Matrixs[0][0];
                return Matrix(1, 1, pt);
            } else {//
                //Inverted triangle
                T **IMatrix = new T *[Rows];
                T **t = new T *[Rows];

                for (int i = 0; i < Rows; i++) {
                    t[i] = new T[Rows];
                    IMatrix[i] = new T[Rows];

                    for (int j = 0; j < Cols; j++) {
                        t[i][j] = Matrixs[i][j];

                        if (i == j)
                            IMatrix[i][j] = 1;
                        else
                            IMatrix[i][j] = 0;
                    }
                }


                for (int i = 0; i < Rows; i++) {
                    int privot = i;
                    T max = t[i][i];
                    if (max < 0)
                        max = -max;


                    for (int j = i + 1; j < Rows; j++) {
                        T tmp = t[j][j];
                        if (tmp < 0)
                            tmp = -tmp;//abs

                        if (max < tmp) {
                            max = tmp;
                            privot = j;
                        }
                    }

                    //can't inverse
                    if (max == 0) {
                        throw "\033[31mIrreversible Error: \033[0mthe matrixs is  irreversible.";
                    }

                    //change two rows
                    if (privot != i) {
                        for (int k = 0; k < Rows; k++) {
                            T tmp;
                            tmp = t[i][k];
                            t[i][k] = t[privot][k];
                            t[privot][k] = tmp;

                            T Inv;
                            Inv = IMatrix[i][k];
                            IMatrix[i][k] = IMatrix[privot][k];
                            IMatrix[privot][k] = Inv;
                        }
                    }

                    //make mat[i][i] be 1,others 0
                    for (int j = i + 1; j < Rows; j++) {
                        T f = t[j][i] / t[i][i];

                        for (int k = 0; k < Rows; k++) {
                            t[j][k] -= t[i][k] * f;
                            IMatrix[j][k] -= IMatrix[i][k] * f;

                        }
                    }
                }

                for (int i = Rows - 1; i >= 0; i--) {
                    T f = t[i][i];
                    if (f == 0) {
                        throw
                                "\033[31mIrreversible Error: \033[0mthe matrixs is  irreversible.";
                    }

                    for (int j = 0; j < Rows; j++) {
                        t[i][j] /= f;
                        IMatrix[i][j] /= f;
                    }

                    for (int j = 0; j < i; j++) {
                        T m = t[j][i];
                        for (int k = 0; k < Rows; k++) {
                            t[j][k] -= m * t[i][k];
                            IMatrix[j][k] -= m * IMatrix[i][k];
                        }
                    }

                }

                return Matrix(Cols, Rows, IMatrix);
            }

        }

        //mid
        Matrix<T>  conv_same(const Matrix<T> &kernal) {
            int k_rows = kernal.Rows;
            int k_cols = kernal.Cols;
            //kernal's pos
            int k_x = (int) k_rows / 2;
            int k_y = (int) k_cols / 2;
//
//            T ans[Rows][Cols];



            T **ans = new T *[Rows];
//
            for (int i = 0; i < Rows; i++) {
                ans[i] = new T[Rows];
                for (int j = 0; j < Cols; j++) {
                    ans[i][j] = '\0';

                }

            }

            for (int i = 0; i < Rows; i++) {

                for (int j = 0; j < Cols; j++) {
                    //calculate ans
                    for (int k = 0; k < k_rows; k++) {
                        for (int l = 0; l < k_cols; l++) {
                            int p_x = i - (k_x - k);
                            int p_y = j - (k_y - l);

                            if (p_x < 0 || p_x >= Rows || p_y < 0 || p_y > Cols) {
                                continue;
                            } else {
                                ans[i][j] += Matrixs[p_x][p_y] * kernal.Matrixs[k][l];
                            }
                        }
                    }
                }
            }

//            Matrix<T> convSameMode(Rows, Cols, *ans);
//            convSameMode.ShowMatrix();

            return Matrix(Rows, Cols, ans);

        };


        //[a,b:c,d] not include d,b
        Matrix<T> slice(int a, int b, int c, int d) {
            if (a >= b || c >= d || a < 0 || b > Rows || c < 0 || d > Cols) {
                throw "\033[31mSlice Error!\033[31m";
            }

//            T ans[b - a][d - c];
            int ans_row = b - a;
            int ans_col = d - c;
            T **ans_slice = new T *[ans_row];
//            T ans[ans_row][ans_col];


            for (int i = 0; i < ans_row; i++) {
                ans_slice[i] = new T[ans_row];
                for (int j = 0; j < ans_col; j++) {
                    ans_slice[i][j] = '\0';

                }

            }

            for (int i = 0; i < ans_row; i++) {
                for (int j = 0; j < ans_col; j++) {
                    ans_slice[i][j] = Matrixs[a + i][c + j];
                }
            }

//            Matrix<T> sliceMatrix(Rows, Cols, *ans);
//            sliceMatrix.ShowMatrix();

            return Matrix(Rows, Cols, ans_slice);
        }

        //[a,b:]
        Matrix<T> slice(int a, int b, int type) {
            if (a >= b) {
                throw "\033[31mSlice Error!\033[31m";
            }

            int ans_size = b - a;

            if (type == -1) {
                if (a < 0 || b > Rows) {
                    throw "\033[31mSlice Error!\033[31m";
                }

                T **ans_slic = new T *[ans_size];

                for (int i = 0; i < ans_size; i++) {
                    ans_slic[i] = new T[ans_size];
                    for (int j = 0; j < Cols; j++) {
                        ans_slic[i][j] = '\0';

                    }

                }

                for (int i = 0; i < ans_size; i++) {
                    for (int j = 0; j < Cols; j++) {
                        ans_slic[i][j] = Matrixs[a + i][j];
                    }
                }

                return Matrix<T>(ans_size, Cols, ans_slic);

            } else if (type == -2) {
                if (a < 0 || b > Cols) {
                    throw "\033[31mSlice Error!\033[31m";
                }

                T **ans_type = new T *[Rows];

                for (int i = 0; i < Rows; i++) {
                    ans_type[i] = new T[Rows];
                    for (int j = 0; j < ans_size; j++) {
                        ans_type[i][j] = '\0';

                    }

                }


                for (int i = 0; i < Rows; i++) {
                    for (int j = 0; j < ans_size; j++) {
                        ans_type[i][j] = Matrixs[i][a + j];
                    }
                }

                return Matrix<T>(Rows, ans_size, ans_type);
            }else  {
                throw "\033[31mSlice Type Error!\033[31m";
            }
        }

//        //[a,:]

//        Matrix<T> slice(int a,int type){
//
//
//            if (type==-1){
//                if (a<=-Rows || a>=Rows){
//                    throw "\033[31mSlice Error!\033[31m";
//                }
//                int row_size=a;
//
//                for (int i = 0; i < b-a+1; i++) {
//                    for (int j = 0; j < Cols; j++) {
//                        ans[i][j]=Mat[a+i][j];
//                    }
//                }
//
//                return Matrix<T>(b-a+1,Cols,*ans);
//
//            } else if (type==-2){
//                T ans[Rows][b-a+1];
//                for (int i = 0; i < Rows; i++) {
//                    for (int j = 0; j < b-a+1; j++) {
//                        ans[i][j]=Mat[i][a+j];
//                    }
//                }
//
//                return Matrix<T>(Rows,b-a+1,*ans);
//            }i
//        }


        //openCV

        Mat Vec2Mat() {

            Mat img(Rows, Cols, CV_8UC1);
            uchar *ptmp = NULL;
            for (int i = 0; i < Rows; ++i) {
                ptmp = img.ptr<uchar>(i);

                for (int j = 0; j < Cols; ++j) {
                    ptmp[j] = Matrixs[i][j];
                }
            }

            return img;
        }

        //mat to matrixs
        Matrix<T> Mat2Vec(string pic)//read pic
        {
            Mat img = imread(pic, 0);
            if (img.empty()) {
                throw "\033[31mRead Picture Failed!\033[31m";
            }


            uchar **array = new uchar *[img.rows];
            for (int i = 0; i < img.rows; i++)
                array[i] = new uchar[img.cols];

            uchar *ptmp = NULL;

            for (int i = 0; i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                    array[i][j] = img.at<uchar>(i, j);
                }
            }

            return array;
        }

        Matrix<T> Mat2Vec(Mat img)//input Mat
        {
//            Mat img = imread(pic,0);
            if (img.empty()) {
                throw "\033[31mRead Picture Failed!\033[31m";
            }


            uchar **array = new uchar *[img.rows];
            for (int i = 0; i < img.rows; i++)
                array[i] = new uchar[img.cols];

            uchar *ptmp = NULL;

            for (int i = 0; i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                    array[i][j] = img.at<uchar>(i, j);
                }
            }

            return array;
        }


        //居然要重写等号，虽然我不知道为什么，不写赋值就会有问题。
        Matrix<T> operator=(const Matrix<T> &other) const {
            //TODO: more specific
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot assign value!\033[31m";
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Matrixs[i][j] = other.Matrixs[i][j];
            //result.ShowMatrix();
            return *this;
        }

        Matrix<T> operator+(const Matrix<T> &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot plus!\033[31m";
            Matrix<T> result(Rows, Cols);//constructor反了，搞到这里直接转置了hhh
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Matrixs[i][j] = Matrixs[i][j] + other.Matrixs[i][j];
            //result.ShowMatrix();
            return result;
        }

        Matrix<T> operator-(const Matrix<T> &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot minus!\033[31m";
            Matrix<T> result(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Matrixs[i][j] = Matrixs[i][j] - other.Matrixs[i][j];
            //result.ShowMatrix();
            return result;
        }


        Matrix<T> operator*(const Matrix<T> &other) const {
            if (Rows != other.Cols || Cols != other.Rows)
                throw "\033[31mSize does not match! Cannot multiply!\033[31m";
            Matrix<T> result(Rows, other.Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < other.Cols; j++)
                    for (int k = 0; k < other.Rows; k++) {
                        //cout<<"**********"<<endl;
                        //result.ShowMatrix();
                        result.Matrixs[i][j] += Matrixs[i][k] * other.Matrixs[k][j];
                        //result.ShowMatrix();
                    }
            //result.ShowMatrix();
            return result;
        }

        Matrix<T> eleWiseMul(const Matrix<T> &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot element-wize multiply!\033[31m";
            Matrix<T> result(Rows, other.Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < other.Cols; j++)
                    result.Matrixs[i][j] = Matrixs[i][j] * other.Matrixs[i][j];
            return result;
        }

        Matrix<T> operator*(const T k) const {
            Matrix<T> result(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Matrixs[i][j] = k * result.Matrixs[i][j];
            return result;
        }

        Matrix<T> operator/(const T k) const {
            Matrix<T> result(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Matrixs[i][j] = result.Matrixs[i][j] / k;
            return result;
        }

        Matrix<T> CrossRow(const Matrix<T> &other) const {//行向量乘行向量,返回3维向量
            if (Rows != 1 || other.Rows != 1)
                throw "Not vectors! Cannot done cross product";
            Matrix<T> result(1, 3);
            if (Cols == 1) {
                result.Matrixs[0][0] = 0;
                result.Matrixs[0][1] = 0;
                result.Matrixs[0][2] = 0;
            } else if (Cols == 2) {
                result.Matrixs[0][0] = 0;
                result.Matrixs[0][0] = 0;
                result.Matrixs[0][0] = Matrixs[0][0] * other.Matrixs[0][1] - Matrixs[0][1] * other.Matrixs[0][0];
            } else if (Cols == 3) {
                result.Matrixs[0][0] = Matrixs[0][1] * other.Matrixs[0][2] - Matrixs[0][2] * other.Matrixs[0][1];
                result.Matrixs[0][0] = Matrixs[0][2] * other.Matrixs[0][0] - Matrixs[0][0] * other.Matrixs[0][2];
                result.Matrixs[0][0] = Matrixs[0][0] * other.Matrixs[0][1] - Matrixs[0][1] * other.Matrixs[0][0];
            }
            return result;
        }


    };

    template<typename T>
    class SparseMatrix {
    private:
        int Rows;
        int Cols;
        int Items;
        T *row;
        T *col;
        T *val;
        int itemMax;
    public:
        SparseMatrix(int rows, int cols) : Rows(rows), Cols(cols), itemMax(rows * cols) {
            T a = '\0';
            Items = 0;
            row = new T[itemMax];
            col = new T[itemMax];
            val = new T[itemMax];
            for (int i = 0; i < itemMax; i++) {
                row[i] = 0;
                col[i] = 0;
                val[i] = a;
            }
        }

        SparseMatrix(int rows, int cols, int items, int *row_in, int *col_in, T *val_in) : Rows(rows), Cols(cols),
                                                                                           Items(items),
                                                                                           itemMax(rows * cols) {
            row = new T[itemMax];
            col = new T[itemMax];
            val = new T[itemMax];
            for (int i = 0; i < items; i++) {
                row[i] = row_in[i];
                col[i] = col_in[i];
                val[i] = val_in[i];
            }
        }

        void setItems(int a) {
            Items = a;
        }

        int getItems() {
            return Items;
        }

        Matrix<T> Sparse2Norm() {
            T arr[Rows * Cols];
            for (int i = 0; i < Rows * Cols; i++)
                arr[i] = 0;
            Matrix<T> result(Rows, Cols, arr);

            for (int i = 0; i < Items; i++)
                result.ChangeItem(row[i], col[i], val[i]);
            return result;
        }

        void ShowSparseMatrix() {
            Sparse2Norm().ShowMatrix();
        }

        SparseMatrix<T> operator=(const SparseMatrix &other) {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot assign value!\033[31m";
            for (int i = 0; i < other.Items; i++) {
                row[i] = other.row[i];
                col[i] = other.col[i];
                val[i] = other.val[i];
                Items = other.Items;
            }
            return *this;
        }

        SparseMatrix<T> operator+(const SparseMatrix &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot plus!\033[31m";
            //int item = Items + other.Items;
            SparseMatrix<T> result(Rows, Cols);
            result = *this;
//            for (int i = 0; i < Items; i++) {
//                result.row[i] = row[i];
//                result.col[i] = col[i];
//                result.val[i] = val[i];
//            }
            int index = Items;
            for (int i = 0; i < other.Items; i++) {//取后者
                for (int j = 0; j < Items; j++) {//搜前者
                    if (row[j] == other.row[i] && col[j] == other.col[i]) {
                        //item--;
                        result.val[j] += other.val[i];
                        break;
                    } else if (j == Items - 1) {
                        result.row[index] = other.row[i];
                        result.col[index] = other.col[i];
                        result.val[index] = other.val[i];
                        index++;
                    }

                }
            }
            result.Items = index;
            return result;

        }

        SparseMatrix<T> operator-(const SparseMatrix &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot minus!\033[31m";
            //int item = Items + other.Items;
            SparseMatrix<T> result(Rows, Cols);
            for (int i = 0; i < Items; i++) {
                result.row[i] = row[i];
                result.col[i] = col[i];
                result.val[i] = val[i];
            }
            int index = Items;
            for (int i = 0; i < other.Items; i++) {
                for (int j = 0; j < Items; j++) {
                    if (row[j] == other.row[i] && col[j] == other.col[i]) {
                        //item--;
                        result.val[j] -= other.val[i];
                        break;
                    } else if (j == Items - 1) {
                        result.row[index] = other.row[i];
                        result.col[index] = other.col[i];
                        result.val[index] -= other.val[i];
                        index++;
                    }

                }
            }
            result.Items = index;
            return result;

        }


    };


};

#endif //WSL_MATRIX_H
