package com.company;

public class MenuModel {
    private IMatrix a;
    private IMatrix b;
    private IMatrix c;

    private MenuView view;

    public void init() {
        view = new MenuView();
        // nothing
    }

    public void initA() {
        view.printText("Input matrix A", true);
        setA(view.inputMatrix());
    }

    public void initB() {
        view.printText("Input matrix B", true);
        setB(view.inputMatrix());
    }

    public void multiply() {
        if (isAvalid() && isBvalid())
            c = a.multiply(b);
        else
            view.printTextErr("Input matrices are null");
    }

    public void transposeA() {
        if (isAvalid())
            a.transpose();
        else
            view.printTextErr("A is null");
    }

    public void transposeB() {
        if (isBvalid())
            b.transpose();
        else
            view.printTextErr("B is null");
    }

    public void transposeC() {
        if (isCvalid())
            c.transpose();
        else
            view.printTextErr("C is null");
    }

    public boolean isAvalid() {
        return a != null;
    }

    public boolean isBvalid() {
        return b != null;
    }

    public boolean isCvalid() {
        return c != null;
    }

    public IMatrix getA() {
        return a;
    }

    public void setA(IMatrix _a) {
        a = _a;
    }

    public IMatrix getB() {
        return b;
    }

    public void setB(IMatrix _b) {
        b = _b;
    }

    public IMatrix getC() {
        return c;
    }

    public void setC(IMatrix _c) {
        c = _c;
    }
}
