package com.company;

interface IMatrix {
    int getCols();
    int getRows();
    void setVal(int r, int c, int val) throws MatrixIndexOutOfBoundsException;
    int getVal(int r, int c) throws MatrixIndexOutOfBoundsException;
    void transpose();
    IMatrix multiply(IMatrix b);
}

public class Matrix implements IMatrix {
    private int rows;
    private int cols;
    private int data[][];

    //  CTORS

    Matrix(int r, int c) {
        data = new int[r][c];
        rows = r;
        cols = c;
    }

    Matrix(Matrix oth) {
        rows = oth.getRows();
        cols = oth.getCols();
        data = new int[rows][cols];

        try {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    setVal(i, j, oth.getVal(i, j));
        } catch (MatrixIndexOutOfBoundsException e) {
            e.printStackTrace();
        }
    }

    Matrix(int [][]vals) {
        rows = vals.length;
        cols = vals[0].length;
        data = new int[rows][cols];

        try {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    setVal(i, j, vals[i][j]);
        } catch (MatrixIndexOutOfBoundsException e) {
            e.printStackTrace();
        }
    }

    // IMPLEMENTATION

    public IMatrix multiply(IMatrix b) {
        if (getCols() != b.getRows()) {
            System.err.println("array sizes don't match");
            return null;
        }

        Matrix c = new Matrix(getRows(), b.getCols());

        try {
            for (int row = 0; row < c.getRows(); ++row) {
                for (int col = 0; col < c.getCols(); ++col) {
                    int sum = 0;
                    for (int k = 0; k < getCols(); ++k) {
                        sum += (getVal(row, k) * b.getVal(k, col));
                    }
                    c.setVal(row, col, sum);
                }
            }
        } catch (MatrixIndexOutOfBoundsException e) {
            e.printStackTrace();
        }

        return c;
    }

    public void transpose() {
        int [][]newData = new int[cols][rows];
        try {
            for (int i = 0; i < getRows(); i++)
                for (int j = 0; j < getCols(); j++)
                    newData[j][i] = getVal(i, j);
        } catch (MatrixIndexOutOfBoundsException e) {
            e.printStackTrace();
        }
        data = newData;
        int tmp = rows;
        rows = cols;
        cols = tmp;
    }

    // UTILITIES

    @Override
    public String toString() {
        String str = "";
        try {
            for (int i = 0; i < rows - 1; ++i) {
                str += "[ ";
                for (int j = 0; j < cols - 1; ++j) {
                    str += getVal(i, j) + ", ";
                }
                str += getVal(i, cols - 1) + " ]\n";
            }

            str += "[ ";
            for (int j = 0; j < cols - 1; ++j) {
                str += getVal(rows - 1, j) + ", ";
            }
            str += getVal(rows - 1, cols - 1) + " ]";
        } catch (MatrixIndexOutOfBoundsException e) {
            e.printStackTrace();
        }

        return str;
    }

    // GETTERS & SETTERS

    public int getCols() {
        return cols;
    }

    public int getRows() {
        return rows;
    }

    public void setVal(int r, int c, int val) throws MatrixIndexOutOfBoundsException {
        if ((c >= 0) && (c < cols)) {
            if ((r >= 0) && (r < rows)) {
                data[r][c] = val;
            } else {
                throw new MatrixIndexOutOfBoundsException(r, rows, false);
            }
        } else {
            throw new MatrixIndexOutOfBoundsException(c, cols, true);
        }
    }

    public int getVal(int r, int c) throws MatrixIndexOutOfBoundsException {
        if ((c >= 0) && (c < cols)) {
            if ((r >= 0) && (r < rows)) {
                return data[r][c];
            } else {
                throw new MatrixIndexOutOfBoundsException(r, rows, false);
            }
        } else {
            throw new MatrixIndexOutOfBoundsException(c, cols, true);
        }
    }
}
