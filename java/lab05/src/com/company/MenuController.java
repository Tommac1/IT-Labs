package com.company;

import java.util.Scanner;

public class MenuController {
    private enum StateMachineState {
        INIT,
        MAIN_MENU,
        INPUT_MATRICES,
        SUB_MENU_ARITHM_OPS,
        SUB_MENU_DISPLAY,
        QUIT,
    }

    private MenuView view;
    private MenuModel model;

    private StateMachineState state = StateMachineState.INIT;

    public void init() {
        model = new MenuModel();
        view = new MenuView();
        model.init();
    }

    public boolean execute() {
        boolean isExit = false;

        switch (state) {
            case INIT:
                state = stateInit();
                break;
            case MAIN_MENU:
                state = stateMainMenu();
                break;
            case INPUT_MATRICES:
                state = stateInputMatrices();
                break;
            case SUB_MENU_ARITHM_OPS:
                state = stateArithmOps();
                break;
            case SUB_MENU_DISPLAY:
                state = stateDisplay();
                break;
            case QUIT:
            default:
                isExit = true;
                break;
        }

        return isExit;
    }

    private StateMachineState stateInit() {
        return StateMachineState.MAIN_MENU; // no init for now
    }

    private StateMachineState stateMainMenu() {
        StateMachineState ret;
        int sel = view.mainMenu();

        switch (Vals.MainMenuTrans.values()[sel - 1]) {
            case INPUT_MATRICES:
                ret = StateMachineState.INPUT_MATRICES;
                break;
            case ARITHMETIC_OPERATIONS:
                ret = StateMachineState.SUB_MENU_ARITHM_OPS;
                break;
            case DISPLAY_MATRICES:
                ret = StateMachineState.SUB_MENU_DISPLAY;
                break;
            case QUIT:
            default:
                ret = StateMachineState.QUIT;
                break;
        }

        return ret;
    }

    private StateMachineState stateInputMatrices() {
        model.initA();
        model.initB();

        return StateMachineState.MAIN_MENU; // from input always return to main menu
    }

    private StateMachineState stateArithmOps() {
        int sel = view.subMenu(Vals.MainMenuTrans.ARITHMETIC_OPERATIONS);

        switch (Vals.SubMenuArithmOpsTrans.values()[sel - 1]) {
            case TRANSPOSE_INPUT_MATRICES: {
                model.transposeA();
                model.transposeB();
                break;
            }
            case MULTIPLY_INPUT_MATRICES: {
                model.multiply();
                break;
            }
            case TRANSPOSE_OUTPUT_MATRIX: {
                model.transposeC();
                break;
            }
            case BACK:
            default:
                break;
        }

        return StateMachineState.MAIN_MENU; // from disp always return to main menu
    }

    private StateMachineState stateDisplay() {
        int sel = view.subMenu(Vals.MainMenuTrans.DISPLAY_MATRICES);

        switch (Vals.SubMenuDisplayTrans.values()[sel - 1]) {
            case DISPLAY_INPUT_MATRICES:
                view.printMatrix("A = ", model.getA());
                view.printMatrix("B = ", model.getB());
                break;
            case DISPLAY_OUTPUT_MATRIX:
                view.printMatrix("C = ", model.getC());
                break;
            case BACK:
            default:
                break;
        }

        return StateMachineState.MAIN_MENU; // from disp always return to main menu
    }
}
