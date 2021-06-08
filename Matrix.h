//
// Created by zhang on 2021/5/15.
//

#ifndef WSL_MATRIX_H
#define WSL_MATRIX_H
//structure
namespace MATRIX {//lab9

    template<typename T>
    class Matrix {
    private:
        int Cols;
        int Rows;
        int size;
        //other properity
        T **Mat;//lab12, a pointer to point the matrix
    public:
        Matrix() {}

        Matrix(int cols, int rows) : Cols(cols), Rows(rows), size(rows * cols) {
            Mat = new T *[rows];
            for (int i = 0; i < rows; i++) {
                Mat[i] = new T[cols];
            }
        };//default matrix

        Matrix(int cols, int rows, T *p) : Cols(cols), Rows(rows), size(rows * cols) {
            Mat = new T *[rows];

            T *pr = p;

            for (int i = 0; i < rows; i++) {
                Mat[i] = new T[cols];
                for (int j = 0; j < cols; j++) {

                    Mat[i][j] = *(pr++);
                }
            }
        }; //one dimension array

        Matrix(int cols, int rows, T **p) : Cols(cols), Rows(rows), size(rows * cols) {
            Mat = new T *[rows];

            for (int i = 0; i < rows; i++) {
                Mat[i] = new T[cols];
                for (int j = 0; j < cols; j++) {
                    Mat[i][j] = p[i][j];
                }
            }
        }; //two dimension array

        Matrix(const Matrix &mat) : Cols(mat.Cols), Rows(mat.Rows), size(mat.size) {
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
        }

        int GetCols(){
            return Cols;
        };

        int GetRows(){
            return Rows;
        };

        int GetSize(){
            return size;
        };

        void ShowMatrix(){

        };

        //get some values of matrix
//        T *Mat;
//
//        GetMatrix();

        T GetSingleVal(int col, int row);

        T ChangeSingleVal(int col, int row, T val);

        T *GetColum(int col);

        T *GetRow(int row);

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

        T trace();

        T Det();

        T avg();

        T EigenValue();

        T EigenVector();//vector

        /*编译会报错我先注释掉了
        //advanced operator implement
        T *Mat

        inverse();

        T *Mat

        convlve();//
        T *Mat

        reshape();

        //basic operator implement +,-,/,=,>>,
*/
        void show() {
            for (int i = 0; i < Rows; i++) {
                std::cout << "[ ";
                for (int j = 0; j < Cols; j++) {
                    std::cout << Mat[i][j] << ' ';
                }
                std::cout << "]" << std::endl;
            }
        }

    };

};

#endif //WSL_MATRIX_H
