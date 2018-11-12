#include <iostream>
#include <thread>
#include <vector>
#include <cassert>

#include <time.h>
#include <stdlib.h>

#define SIZE 100
#define MODULO 2
#define DONT_FILL_MATRIX 0
#define FILL_MATRIX 1

/* ========== Main matrix class ========== */
class Matrix {
public:
    int rows = 0;
    int cols = 0;
    int **data = nullptr;

    Matrix();
    Matrix(int _rows, int _cols, int fill) {
        rows = _rows;
        cols = _cols;

        make_matrix();

        if (fill)
            fill_matrix();
    }
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    ~Matrix() {
        if (data != nullptr) {
            for (int i = 0; i < rows; ++i) 
                delete[] data[i];

            delete[] data;
        }
    };

private:
    void fill_matrix() {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = (rand() % (MODULO * 2 + 1)) - MODULO;
    }
    
    void make_matrix() {
        data = new int*[rows];
        for (int i = 0; i < rows; ++i)
            data[i] = new int[cols];
    }
};

std::ostream& operator<<(std::ostream& os, const Matrix& m)  
{  
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0 ; j < m.cols; ++j)
			os << m.data[i][j] << " ";
		os << std::endl;
	}
    return os;  
}  

/* ========== Vector with threads ========== */
std::vector<std::thread> vt;

/* ========== Function declarations ========== */
void do_multiplication(Matrix *a, Matrix *b, int i, int j, int *out)
{
	int result = 0;
    /* Guard */
    assert(a->cols == b->rows);

	for (int k = 0; k < a->cols; ++k) {
		result += (a->data[i][k] * b->data[k][j]);
	}

	*out = result;
}

int calculate_multiplications(int a_cols, int a_rows, int b_cols)
{
    return (a_cols * a_rows * b_cols);
}

/* Result: -1 -> (A * B) * C
 *          0 -> equal
 *          1 -> A * (B * C)
 */
int optimize_multiplications(Matrix *a, Matrix *b, Matrix *c)
{
    int result = 0;
    int muls_with_c;
    int muls_with_a;

    std::cout << "Calculating multiplications with A first: " << std::endl;
    muls_with_a = calculate_multiplications(a->cols, a->rows, b->cols);
    muls_with_a += calculate_multiplications(b->cols, a->rows, c->cols);
    std::cout << muls_with_a << " multiplications" << std::endl;

    std::cout << "Calculating multiplications with C first: " << std::endl;
    muls_with_c = calculate_multiplications(b->cols, b->rows, c->cols);
    muls_with_c += calculate_multiplications(a->cols, a->rows, c->cols);
    std::cout << muls_with_c << " multiplications" << std::endl;


    if (muls_with_a < muls_with_c)
        result = -1;
    else if (muls_with_a > muls_with_c)
        result = 1;
    /* Otherwise are equal */

    return result;
}

void spawn_threads(Matrix *res, Matrix *a, Matrix *b)
{
    for (int i = 0; i < res->rows; ++i) {
        for (int j = 0; j < res->cols; ++j) {
            vt.push_back(std::thread(do_multiplication, a, b, i, j, &(res->data[i][j])));
        }
    }

    for (auto &thread : vt)
        if (thread.joinable())
            thread.join();
}

void multiply_matrixes(Matrix **res, Matrix *a, Matrix *b, Matrix *c)
{
    Matrix *temp = nullptr;
    int optimize_result = optimize_multiplications(a, b, c);

    *res = new Matrix(a->rows, c->cols, DONT_FILL_MATRIX);

    if (optimize_result <= 0) {
        /* Start with A (by default) */
        std::cout << "Starting with A" << std::endl;
        temp = new Matrix(a->rows, b->cols, DONT_FILL_MATRIX);
        spawn_threads(temp, a, b);
        spawn_threads(*res, temp, c);
    }
    else{
        /* Start with C */
        std::cout << "Starting with C" << std::endl;
        temp = new Matrix(b->rows, c->cols, DONT_FILL_MATRIX);
        spawn_threads(temp, b, c);
        spawn_threads(*res, a, temp);
    }
}

int main()
{
    srand(time(NULL));

    constexpr int A_ROWS = 101;
    constexpr int A_COLS = 99;
    constexpr int B_ROWS = 99;
    constexpr int B_COLS = 102;
    constexpr int C_ROWS = 102;
    constexpr int C_COLS = 98;

	Matrix *a = new Matrix(A_ROWS, A_COLS, FILL_MATRIX);
	Matrix *b = new Matrix(B_ROWS, B_COLS, FILL_MATRIX);
	Matrix *c = new Matrix(C_ROWS, C_COLS, FILL_MATRIX);
	Matrix *d = nullptr;

    multiply_matrixes(&d, a, b, c);

    std::cout << "D = " << std::endl;
    std::cout << *d;

    delete a;
    delete b;
    delete c;
    delete d;

	return 0;
}
