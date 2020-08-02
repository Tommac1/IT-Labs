package com.company;

import java.util.Scanner;

public class MenuView {
    public void printText(String txt, boolean newLine) {
        if (newLine) {
            System.out.println(txt);
        } else {

        }
    }

    public void printTextErr(String txt) {
        System.err.println(txt);
    }

    public void printMatrix(String label, IMatrix m) {
        System.out.println(label);
        System.out.println(m);
    }

    public int mainMenu() {
        int sel;
        Scanner input = new Scanner(System.in);

        printMenu(Vals.Menus.MAIN_MENU);

        sel = input.nextInt();
        return sel;
    }

    public int subMenu(Vals.MainMenuTrans mainMenuIdx) {
        int sel;
        Scanner input = new Scanner(System.in);

        switch (mainMenuIdx) {
            case INPUT_MATRICES:
                // no work here
                break;
            case ARITHMETIC_OPERATIONS:
                printMenu(Vals.Menus.SUB_MENU_APRITHM_OPS);
                break;
            case DISPLAY_MATRICES:
                printMenu(Vals.Menus.SUB_MENU_DISPLAY);
                break;
            case QUIT:
            default:
                break;
        }

        sel = input.nextInt();
        return sel;
    }

    public void printMenu(Vals.Menus menu) {
        String []text;

        switch (menu) {
            case MAIN_MENU:
                text = Vals.MAIN_MENU_TEXT;
                break;
            case SUB_MENU_APRITHM_OPS:
                text = Vals.SUB_MENU_ARITHM_OPS_TEXT;
                break;
            case SUB_MENU_DISPLAY:
                text = Vals.SUB_MENU_DISPLAY_TEXT;
                break;
            default:
                text = new String[1];
                text[0] = "ERROR";
                break;
        }

        for (String line : text) {
            System.out.println(line);
        }
    }

    public IMatrix inputMatrix() {
        String text;
        int rows;
        int cols;
        Scanner input = new Scanner(System.in);

        while (true) {
            System.out.print("Size (NxM): ");
            text = input.nextLine();
            String []parts = text.split("x");
            try {
                rows = Integer.parseInt(parts[0]);
                cols = Integer.parseInt(parts[1]);
                break;
            } catch (NumberFormatException e) {
                System.err.println("input valid size format (eg. 2x3)");
            }
        }

        IMatrix m = new Matrix(rows, cols);
        inputMatrixData(m);
        return m;
    }

    private void inputMatrixData(IMatrix m) {
        System.out.println("Input rows data (lines, values separated by spaces):");
        try {
            int i = 0;
            while (i < m.getRows()) {
                System.out.print("Input " + (i + 1) + " row: ");
                Scanner input = new Scanner(System.in);
                String text = input.nextLine();
                String []parts = text.split(" ");

                if (parts.length == m.getCols()) {
                    for (int j = 0; j < m.getCols(); ++j) {
                        m.setVal(i, j, Integer.parseInt(parts[j]));
                    }
                    i++; // advance row
                } else {
                    System.err.println("to " +
                            (parts.length > m.getCols() ? "many" : "few") +
                            " values, should be " +
                            m.getCols());
                }
            }
        } catch (MatrixIndexOutOfBoundsException e) {
            e.printStackTrace();
        }
    }
}
