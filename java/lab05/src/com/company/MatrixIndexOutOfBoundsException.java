package com.company;

public class MatrixIndexOutOfBoundsException extends Exception {
    public MatrixIndexOutOfBoundsException(int x, int max, boolean isCol) {
        super("MatrixIndexOutOfBoundsException: " + (isCol ? "col " : "row ") + x + "/" + max);
    }
}
