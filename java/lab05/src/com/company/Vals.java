package com.company;

public class Vals {
    public enum Menus {
        MAIN_MENU,
        SUB_MENU_APRITHM_OPS,
        SUB_MENU_DISPLAY,
    }

    public enum MainMenuTrans {
        INPUT_MATRICES,
        ARITHMETIC_OPERATIONS,
        DISPLAY_MATRICES,
        QUIT,
    }

    public enum SubMenuArithmOpsTrans {
        TRANSPOSE_INPUT_MATRICES,
        MULTIPLY_INPUT_MATRICES,
        TRANSPOSE_OUTPUT_MATRIX,
        BACK,
    }

    public enum SubMenuDisplayTrans {
        DISPLAY_INPUT_MATRICES,
        DISPLAY_OUTPUT_MATRIX,
        BACK,
    }

    public final static String []MAIN_MENU_TEXT = {
        "Menu",
        "-------------------------",
        "1 - Input matrices",
        "2 - Arithmetic operations",
        "3 - Display matrices",
        "4 - Quit",
    };

    public final static String []SUB_MENU_ARITHM_OPS_TEXT = {
        "Arithmetic Operations",
        "-------------------------",
        "1 - Transpose input matrices",
        "2 - Multiply input matrices",
        "3 - Transpose output matrix",
        "4 - Back",
    };

    public final static String []SUB_MENU_DISPLAY_TEXT = {
        "Display Matrices",
        "-------------------------",
        "1 - Display input matrices",
        "2 - Display output matrix",
        "3 - Back",
    };
}
