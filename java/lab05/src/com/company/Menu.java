package com.company;

import java.util.Scanner;

public class Menu {
    private MenuController ctrl;

    void run() {
        ctrl = new MenuController();
        boolean isFinished = false;

        ctrl.init();

        while (!isFinished) {
            isFinished = ctrl.execute();
        }
    }
}
