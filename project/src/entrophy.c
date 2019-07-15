#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#define TAB_SIZE 510

static int COLS = 0;
static int ROWS = 0;
static float COEFF = 0.0;
static char FILE_NAME[256] = { 0 };
static int N = 0;

void count_p(unsigned int *tab_e, float *tab_p)
{
    for (int i = 0; i < TAB_SIZE; i++)
        tab_p[i] = (float) tab_e[i] / (COLS * ROWS);
}

float sum(float *tab_p)
{
    float s = 0;
    float tmp = 0;

    for (int i = 0; i < TAB_SIZE; i++) {
        if (tab_p[i] == 0) {
            tmp = 0;
        }
        else {
            tmp = tab_p[i] * log2f(tab_p[i]);
        }
        s = s + tmp;
    }

    return -s;
}

void parse_args(int argc, char *argv[])
{
    strncpy(FILE_NAME, argv[1], 256);
    ROWS = atoi(argv[2]);
    COLS = atoi(argv[3]);
    COEFF = atof(argv[4]);
    N = COLS * ROWS;
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        printf("usage: %s err_img h w coeff\n", argv[0]);
        return 1;
    }

    parse_args(argc, argv);

    unsigned int *tab_En = malloc(TAB_SIZE * sizeof(int));
    float *tab_P = malloc(TAB_SIZE * sizeof(float));
    unsigned short tmp = 0;
    float Hs = 0;
    float Lsr = 0;
    FILE *ptr = fopen(FILE_NAME, "rb"); // r for read, b for binary

    if (ptr == NULL) {
        fprintf(stderr, "error opening file: %s\n", FILE_NAME);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        // read one pixel and increment histogram
        if (1 != fread(&tmp, sizeof(tmp), 1, ptr)) {
            fprintf(stderr, "error reading file: %s\n", FILE_NAME);
            return 1;
        }
        tab_En[tmp]++;
    }

    count_p(tab_En, tab_P); //licze P
    Hs = sum(tab_P);        //-suma P*log2P
    Lsr = COEFF + Hs;       //dla kazdego Wk

    printf("%s\nLsr = %f\n", FILE_NAME, Lsr);

    fclose(ptr); //zamykam plik metrics dla danego Wk
    return 0;
}
