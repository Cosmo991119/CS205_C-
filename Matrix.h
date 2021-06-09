//
// Created by zhang on 2021/5/15.
//

#ifndef WSL_MATRIX_H
#define WSL_MATRIX_H

#include <complex>

//structure
namespace MATRIX {//lab9
    using namespace std;


    template<typename T>
    class Matrix {
    private:
        int Cols;
        int Rows;
        int size;
        //other properity
        T **Mat;//lab12, a pointer to point the matrix
    public:
        //hbx：我直接改了，先rows再cols
        Matrix(int rows, int cols) : Rows(rows), Cols(cols), size(rows * cols) {
            T a;
            Mat = new T *[rows];
            for (int i = 0; i < rows; i++) {
                Mat[i] = new T[cols];
                for (int j = 0; j < cols; j++)
                    Mat[i][j] = a;//得显式初始化一下，不然乘法会不正常。（我也不知道为什么但实践得）
            }
        };

        Matrix(int rows, int cols, T *p) : Rows(rows), Cols(cols), size(rows * cols) {
            Mat = new T *[rows];
            T *pr = p;
            for (int i = 0; i < rows; i++) {
                Mat[i] = new T[cols];
                for (int j = 0; j < cols; j++) {
                    Mat[i][j] = *(pr++);
                }
            }
        }; //one dimension array

        Matrix(int rows, int cols, T **p) : Rows(rows), Cols(cols), size(rows * cols) {
            Mat = new T *[rows];
            for (int i = 0; i < rows; i++) {
                Mat[i] = new T[cols];
                for (int j = 0; j < cols; j++) {
                    Mat[i][j] = p[i][j];
                }
            }
        }; //two dimension array

        Matrix(const Matrix &mat) : Rows(mat.Rows), Cols(mat.Cols), size(mat.size) {
            Mat = new T *[mat.Rows];

            for (int i = 0; i < mat.Rows; i++) {
                Mat[i] = new T[mat.Rows];
                for (int j = 0; j < mat.Cols; j++) {
                    Mat[i][j] = mat.Mat[i][j];
                }
            }
        };//copy constructor, the size maybe different

        ~Matrix() {
            for (int i = 0; i < Rows; i++) {
                delete[] Mat[i];
            }
            delete[] Mat;
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
                    cout << Mat[i][j] << " ";
                }
                cout << "]" << endl;
            }

        };


        T **GetMatrix() {
            return Mat;
        };

        T GetItem(int col, int row) {
            return Mat[col][row];
        };

        void ChangeItem(int col, int row, T val) {
            Mat[col][row] = val;
        };

        T *GetColum(int col) {
            T *arr[Rows];
            for (int i = 0; i < Rows; ++i) {
                arr[i] = Mat[col][i];
            }
            return arr;
        };

        T *GetRow(int row) {
            T *arr[Cols];
            for (int i = 0; i < Cols; ++i) {
                arr[i] = Mat[i][row];
            }
            return arr;
        };

        //find special value
        //column/row initial, column/row final(0~Cols-1)
        //可做局部比大小、加和
        T max(int row_i, int row_f, int col_i, int col_f) {
            T MAX = Mat[row_i][col_i];
            for (int i = row_i; i < row_f + 1; i++) {
                for (int j = col_i; j < col_f + 1; j++) {
                    if (Mat[i][j] > MAX)
                        MAX = Mat[i][j];
                }
            }
            return MAX;
        }

        T min(int row_i, int row_f, int col_i, int col_f) {
            T MIN = Mat[row_i][col_i];
            for (int i = row_i; i < row_f + 1; i++) {
                for (int j = col_i; j < col_f + 1; j++) {
                    if (Mat[i][j] < MIN)
                        MIN = Mat[i][j];
                }
            }
            return MIN;
        }

        T sum(int row_i, int row_f, int col_i, int col_f) {
            T SUM; //maybe need initialize?
            for (int i = row_i; i < row_f + 1; i++) {
                for (int j = col_i; j < col_f + 1; j++) {
                    SUM += Mat[i][j];
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
                    TRAN.Mat[j][i] = Mat[i][j];
            return TRAN;
        }

        T trace() {
            if (Cols != Rows)
                throw "\033[31msquare Error: \033[0mthe matrixs must be a square matrix.";
            T TRACE;
            for (int i = 0; i < Rows; i++)
                TRACE += Mat[i][i];
            return TRACE;
        }

        //要分清运行时exception和编译器报错的区别。后者可不是exception，是直接提醒使用者就是不能用。
        Matrix<T> MatrixConj() {
            Matrix result(Rows, Cols);
            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Cols; j++) {
                    result.Mat[i][j] = conj(Mat[i][j]);
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
                    remain_mat.Mat[r_i][r_j] = Mat[m_i][m_j];
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
                return Mat[0][0] * Mat[1][1] - Mat[0][1] * Mat[1][0];
            else {
                for (int i = 0; i < Rows; i++)
                    det_val += pow(-1, i) * Mat[i][0] * Remainder(i, 0).Det();
            }
            return det_val;

        }


        T EigenValue();

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
                int *ans[1];
                ans[0] = 1 / Mat[0][0];
                return Matrix(1, 1, ans);
            } else if (Cols == 2) {//2*2
                int **ans[2][2];
                ans[0][0] = Mat[1][1]/ this->Det();
                ans[1][1] = Mat[0][0]/ this->Det();
                ans[0][1] = -Mat[0][1]/ this->Det();
                ans[1][0] = -Mat[1][0]/ this->Det();
                return Matrix(2, 2, ans);
            } else {//
                //Inverted triangle
                T **IMatrix = new T *[Rows];
                T **t = new T *[Rows];

                for (int i = 0; i < Rows; i++) {
                    t[i] = new T[Rows];
                    IMatrix[i] = new T[Rows];

                    for (int j = 0; j < Cols; j++) {
                        t[i][j] = Mat[i][j];

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
                        t[i][j]/=f;
                        IMatrix[i][j]/=f;
                    }

                    for (int j = 0; j < i; j++) {
                        T m = t[j][i];
                        for (int k = 0; k < Rows; k++) {
                            t[j][k]-=m*t[i][k];
                            IMatrix[j][k]-=m*IMatrix[i][k];
                        }
                    }

                }

                return IMatrix;
            }


        }

        //mid
        Matrix<T> conv_same(const Matrix<T> &kernal) {
            int k_rows=kernal.Rows;
            int k_cols=kernal.Cols;
            //kernal's pos
            int k_x=(int) k_rows/2+1;
            int k_y=(int) k_cols/2+1;;

            T ans[Rows][Cols];

            for (int i = 0; i < Rows; i++) {

                for (int j = 0; j <Cols; j++) {
                    //calculate ans
                    ans[i][j]=0;
                    for (int k = 0; k < k_rows; k++) {
                        for (int l = 0; l < k_cols; ++l) {
                            int p_x=i-(k_x-k);
                            int p_y=j-(k_y-l);

                            if (p_x<0 || p_x>=Rows || p_y<0||p_y>Cols){
                                continue;
                            } else{
                                ans[i][j]+=Mat[p_x][p_y]*kernal.Mat[k][l];
                            }
                        }
                    }
                }
            }

            return Matrix<T>(Rows,Cols,*ans);
        };


        //居然要重写等号，虽然我不知道为什么，不写赋值就会有问题。
        Matrix<T> operator=(const Matrix<T> &other) const {

            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot assign value!\033[31m";
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Mat[i][j] = other.Mat[i][j];
            //result.ShowMatrix();
            return *this;
        }

        Matrix<T> operator+(const Matrix<T> &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot plus!\033[31m";
            Matrix<T> result(Rows, Cols);//constructor反了，搞到这里直接转置了hhh
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Mat[i][j] = Mat[i][j] + other.Mat[i][j];
            //result.ShowMatrix();
            return result;
        }

        Matrix<T> operator-(const Matrix<T> &other) const {
            if (Rows != other.Rows || Cols != other.Cols)
                throw "\033[31mSize does not match! Cannot minus!\033[31m";
            Matrix<T> result(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Mat[i][j] = Mat[i][j] - other.Mat[i][j];
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
                        result.Mat[i][j] += Mat[i][k] * other.Mat[k][j];
                        //result.ShowMatrix();
                    }
            //result.ShowMatrix();
            return result;
        }

        Matrix<T> Cross(const Matrix<T> &other) const {
            Matrix<T> result(Rows, other.Cols);
            return result;
        }


    };

};

#endif //WSL_MATRIX_H
