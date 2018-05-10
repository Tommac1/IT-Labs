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
int abs(int x);
int *copy_board(int *dest, int *src);
int **init_population();
int calculate_population_collisions(int **population,  int *collisions);
void destroy_population(int **pop);
void my_swap(int *x, int *y);
int min(int *v, int n);
int avg(int *v, int n);
int *init_ff_vector();
void print_ff_vectors(int *best_ff, int *avg_ff);


int **selection(int **old_population, int *collisions);
int **crossing(int **old_population);
void mutation(int **population);


static int N_COLS = 0;
static int N_ROWS = 0;
static int POP_SIZE = 0;
static int MAX_GEN = 0;
static double PROB_CROSS = 0.0;
static double PROB_MUT = 0.0;

int main(int argc, char *argv[])
{
    clock_t begin = 0;
    clock_t end = 0;
    float time_spent = 0.0f;

    init_env(argc, argv);

    begin = clock();
    n_queens_solve();
    end = clock();

    time_spent = ((double) (end - begin)) / CLOCKS_PER_SEC;
    printf("Time spent: %.4fs\n", time_spent);

    return 0;
}


void n_queens_solve()
{
    int solution_position = -1;
    int generation = 0;
    int **population = init_population();
    int *collisions = calloc(POP_SIZE, sizeof(int));
    int *best_ff = init_ff_vector();
    int *avg_ff = init_ff_vector();

    solution_position = calculate_population_collisions(population, collisions);
    best_ff[generation] = min(collisions, POP_SIZE);
    avg_ff[generation] = avg(collisions, POP_SIZE);

    while ((solution_position == -1) && (generation < MAX_GEN)) {
        generation++;

        /* Selection */
        population = selection(population, collisions);

        /* Crossing */
        population = crossing(population);

        /* Mutation */
        mutation(population);

        /* Calculate FF */
        solution_position = calculate_population_collisions(population, collisions);
        best_ff[generation] = min(collisions, POP_SIZE);
        avg_ff[generation] = avg(collisions, POP_SIZE);
    }


    print_ff_vectors(best_ff, avg_ff);
    printf("Generations of population: %d\n", generation);

    if (solution_position != -1) {
        printf("Solution:\n");
        print_board(population[solution_position]);
    }
    else {
        printf("Solution not found.\nBest in population:\n");
        print_board(population[best_ff[generation]]);
    }

    free(best_ff);
    free(avg_ff);
    free(collisions);

    destroy_population(population);
}

int *init_ff_vector()
{
    int *v = malloc(sizeof(int) * MAX_GEN); 
    assert(v);
    memset(v, 0xFF, sizeof(int) * MAX_GEN);

    return v;
}

void print_ff_vectors(int *best_ff, int *avg_ff)
{
    int i = 0;

    printf("Best FF: ");
    while ((i < MAX_GEN) && (best_ff[i] != -1)) {
        printf("%d ", best_ff[i]);
        i++;
    }

    printf("\n");
    i = 0;

    printf("Avg FF: ");
    while ((i < MAX_GEN) && (best_ff[i] != -1)) {
        printf("%d ", avg_ff[i]);
        i++;
    }

    printf("\n");
}

void mutation(int **population)
{
    double calc_pm = 0.0;
    int mutated_gene = 0;
    int i;

    /* Calculate probability for each individual
     * and if mutation occurs - change it's one random gene (col)*/
    for (i = 0; i < POP_SIZE; ++i) {
        calc_pm = ((double)rand()/(double)RAND_MAX);

        if (calc_pm < PROB_MUT) {
            mutated_gene = rand() % N_COLS;

            population[i][mutated_gene] = rand() % N_ROWS;
        }
    }
}

int **crossing(int **old_population)
{
    int i;
    int j;
    int **new_population = init_population();
    double calc_pc = 0.0;

    for (i = 0; i < POP_SIZE; i += 2) {
        calc_pc = ((double)rand()/(double)RAND_MAX);

        if (calc_pc < PROB_CROSS) {
            /* Parent crossing */
            for (j = 0; j < N_COLS; ++j) {
                if (0 == (rand() % 1)) {
                    /* Swap the genes between parents */
                    my_swap(&(old_population[i][j]), &(old_population[i + 1][j]));
                }
            }
        }

        /* Parents move to new generation */
        new_population[i] = copy_board(new_population[i], 
                old_population[i]);
        new_population[i + 1] = copy_board(new_population[i + 1], 
                old_population[i + 1]);
    }

    destroy_population(old_population);

    return new_population;
}

void my_swap(int *x, int *y)
{
    int tmp = *x;
    *x = *y;
    *y = tmp;
}

int **selection(int **old_population, int *collisions)
{
    int i;
    int **new_population = init_population();
    int opponent1;
    int opponent2;
    int opponent1_collisions = 0;
    int opponent2_collisions = 0;

    for (i = 0; i < POP_SIZE; ++i) {
        opponent1 = rand() % POP_SIZE;
        opponent2 = rand() % POP_SIZE;

        if (opponent1 == opponent2) {
            /* Don't fight with yourself */
            --i;
            continue;
        }

        opponent1_collisions = count_collisions(old_population[opponent1]);
        opponent2_collisions = count_collisions(old_population[opponent2]);

        /* Tournament! */
        if (opponent1_collisions < opponent2_collisions)
            new_population[i] = copy_board(new_population[i],
                    old_population[opponent1]);
        else 
            new_population[i] = copy_board(new_population[i],
                    old_population[opponent2]);

    }

    destroy_population(old_population);

    return new_population;
}

int calculate_population_collisions(int **population,  int *collisions)
{
    int ret = -1;
    int i;

    for (i = 0; i < POP_SIZE; ++i) {
        collisions[i] = count_collisions(population[i]);
        if (collisions[i] == 0)
            ret = i;
    }

    /*  Returned vaule is -1, if solution not found or 
     *  index of solution in population array */
    return ret;
}

int *copy_board(int *dest, int *src)
{
    memcpy(dest, src, sizeof(int) * N_COLS);
    return dest;
}

void destroy_population(int **pop)
{
    int i;

    for (i = 0; i < POP_SIZE; ++i)
        free(pop[i]);

    free(pop);
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

int *init_board()
{
    int i;
    int *board = calloc(N_COLS, sizeof(int));
    assert(board);

    for (i = 0; i < N_COLS; ++i)
        board[i] = rand() % N_ROWS;
    
    return board;
}

int **init_population()
{
    int i;
    int **ret = malloc(sizeof(int *) * (POP_SIZE));
    assert(ret);

    for (i = 0; i < POP_SIZE; ++i) 
        ret[i] = init_board();

    return ret;
}

void init_env(int argc, char *argv[])
{
    if (6 > argc) {
        printf("%s N POP_SIZE MAX_GEN PC PM\n", argv[0]);
        printf("\tN - N queens to place (and size of board)\n");
        printf("\tPOP_SIZE - max population size [10..100]\n");
        printf("\tMAX_GEN - max generations number [100..10000]\n");
        printf("\tPC - probabillity of crossing [0.7..1.0]\n");
        printf("\tPM - probabillity of mutation (0..0.2)\n");
        exit(EXIT_FAILURE);
    }

    N_COLS = atoi(argv[1]);
    N_ROWS = N_COLS;

    POP_SIZE = atoi(argv[2]);
    MAX_GEN = atoi(argv[3]);
    PROB_CROSS = atof(argv[4]);
    PROB_MUT = atof(argv[5]);

    if (POP_SIZE > 100)
        POP_SIZE = 100;

    if (POP_SIZE < 10)
        POP_SIZE = 10;

    if (MAX_GEN > 10000)
        MAX_GEN = 10000;

    if (MAX_GEN < 100)
        MAX_GEN = 100;

    if (PROB_CROSS > 1.0)
        PROB_CROSS = 1.0;

    if (PROB_CROSS < 0.7)
        PROB_CROSS = 0.7;

    if (PROB_MUT >= 2.0)
        PROB_MUT = 1.9;

    if (PROB_MUT <= 0.0)
        PROB_MUT = 0.1;

    srand(time(NULL));
}

int abs(int x)
{
    return (x < 0 ? -x : x);
}

int min(int *v, int n)
{
    int i;
    int ret = v[0];

    for (i = 0; i < n; ++i)
        if (ret > v[i])
            ret = v[i];

    return ret;
}

int avg(int *v, int n)
{
    int i;
    int ret = 0;

    for (i = 0; i < n; ++i)
        ret += v[i];

    ret /= n;

    return ret;
}

