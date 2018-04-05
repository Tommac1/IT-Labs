#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

void init_env(int argc, char *argv[]);
void print_board(int *board);
int count_collisions(int *board);
void n_queens_solve();
int *init_board();
void move_queen(int *board, int col);
int abs(int x);
int *copy_board(int *dest, int *src);


static int N_COLS = 0;
static int N_ROWS = 0;
static int GENERATED = 0;
static int K_MAX = 10000;

int main(int argc, char *argv[])
{
    clock_t begin = 0;
    clock_t end = 0;
    float time_spent = 0.0f;

    init_env(argc, argv);

    begin = clock();
    n_queens_solve();
    end = clock();

    printf("Generated boards: %d\n", GENERATED);
    time_spent = ((double) (end - begin)) / CLOCKS_PER_SEC;
    printf("Time spent: %.4fs\n", time_spent);

    return 0;
}


void n_queens_solve()
{
    int *board = init_board();
    int *new_board = init_board();
    new_board = copy_board(new_board, board);
    int k = 0;
    int collisions = -1;
    int collisions_after = 0;
    int queen_to_move = 0;

    do {
        queen_to_move = rand() % N_COLS; 
        collisions = count_collisions(board); 
        move_queen(new_board, queen_to_move);
        collisions_after = count_collisions(new_board);
        if (collisions > collisions_after)
            board = copy_board(board, new_board);

        ++k;
    } while ((k < K_MAX) && (collisions != 0));

    if (k < K_MAX) {
        printf("Solution: \n");
        print_board(board);
    }
    else {
        printf("Solution not found within %d moves.\n", K_MAX);
    }

    GENERATED = k;
    free(board);
    free(new_board);
}

int *copy_board(int *dest, int *src)
{
    memcpy(dest, src, sizeof(int) * N_COLS);
    return dest;
}

void move_queen(int *board, int col)
{
    int row = board[col];
    int new_row;

    do {
        new_row = rand() % N_COLS;
    } while (new_row == row);

    board[col] = new_row;
}

int count_collisions(int *board)
{
    int collisions = 0;
    int i;
    int j;

    // Check verts, horiz and diags
    for (i = 0; i < N_COLS; i++) {
        for (j = 0; j < N_COLS; j++) {
            if (i == j)
                continue;

            // Same row
            if (board[i] == board[j])
                collisions++;

            // Diags
            // Two points lay on a diagonal if:
            // |x1 - x2| = |y1 - y2|
            if (abs(i - j) == abs(board[i] - board[j])) 
                collisions++;
        }
    }

    // Collision is counted for each queen
    // So we must divide it by two
    collisions >>= 1;

    return collisions;
}

void print_board(int *board)
{
    int i;
    int j;

    for (i = 0; i < N_ROWS; ++i) {
        for (j = 0; j < N_COLS; ++j) {
            if (board[j] == i)
                printf("Q ");
            else
                printf("_ ");
        }
        
        printf("\n");
    }
}

void init_env(int argc, char *argv[])
{
    if (2 > argc) {
        printf("%s N\n", argv[0]);
        printf("\tN - N queens to place (and size of board)\n");
        exit(EXIT_FAILURE);
    }

    if (3 == argc)
        K_MAX = atoi(argv[2]);
    else
        printf("Number of moves defaulting to %d.\n", K_MAX);

    N_COLS = atoi(argv[1]);
    N_ROWS = N_COLS;

    srand(time(NULL));
}

int *init_board()
{
    int i;
    int *board = calloc(N_COLS, sizeof(int));
    assert(board);

    for (i = 0; i < N_COLS; ++i)
        board[i] = rand() % N_ROWS;
    
    return board;
}

int abs(int x)
{
    return (x < 0 ? -x : x);
}
