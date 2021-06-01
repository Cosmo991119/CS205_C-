//
// Created by zhang on 2021/5/15.
//

#ifndef WSL_MATRIX_H
#define WSL_MATRIX_H
//structure
namespace matrix {//lab9

    template<typename T>
    class Matrix {
    private:
        int Cols;
        int Rows;
        int size;
        //other properity
        T **Mat;//lab12, a pointer to point the matrix
    public:
        Matrix();

        Matrix(int cols, int rows):Cols(cols),Rows(rows),size(rows*cols),Mat(new T[rows*cols]){};//default matrix
        Matrix(int cols, int rows, int lens,T *val):Cols(cols),Rows(rows),size(rows*cols),Mat(new T[rows*cols]){
            T *ptr;
            ptr=Mat;
            if (size>=lens){
                for (int i=0;i<lens;i++){
                    *ptr=*val;
                    ptr++;
                    val++;
                }
            } else{
                for (int i=0;i<size;i++){
                    *ptr=*val;
                    ptr++;
                    val++;
                }
            }

        }; //one dimension array

        Matrix(int cols, int rows, T **val):Cols(cols),Rows(rows),size(rows*cols){

        }; //two dimension array

        Matrix(const Matrix &mat);//copy constructor, the size maybe different
        ~Matrix();//de

        int GetCols();

        int GetRows();

        int GetSize();

        void ShowMatrix();

        //get some values of matrix
        T *Mat

        GetMatrix();

        T GetSingleVal(int col, int row);

        T ChangeSingleVal(int col, int row, T val);

        T *GetColum(int col);

        T *GetRow(int row);

        //find special value
        T max();

        T min();

        T sum();

        T trace();

        T Det();

        T avg();

        T EigenValue();

        T EigenVector();//vector

        //advanced operator implement
        T *Mat

        inverse();

        T *Mat

        convlve();//
        T *Mat

        reshape();

        //basic operator implement +,-,/,=,>>,


    };

};

#endif //WSL_MATRIX_H
