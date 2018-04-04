#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

typedef struct sList {
    int **board;
    struct sList *next;
} List;

List *initList();
void pushFront(List **head, int **data);
void pushBack(List **head, int **data);
void printList(List *head);
List *popFront(List **head);
List *createNode(int **data); 
List *popBack(List **head);
int **copyArray(int **data);
void printNode(List *node);
void n_queens_solve();
void placeQueen(int **board, int col, int row);
void print_board(int **board);
int **init_board();
int is_empty(List *head);
int count_queens(int **board);
void generate_boards(int **board);
void delete_node(List *node);
void delete_list(List *head);



List *queue = NULL;

int N_COLS = 4;
int N_ROWS = 4;

int GENERATED = 0;
int DFS = 0;

int main(int argc, char *argv[])
{
    clock_t begin = 0;
    clock_t end = 0;
    float time_spent = 0.0f;

    if (3 != argc) {
        printf("%s N M\n", argv[0]);
        printf("\tN - N queens to place (and size of board)\n");
        printf("\tM - mode (0 = DFS, 1 = BFS)\n");
        exit(EXIT_FAILURE);
    }

    N_COLS = atoi(argv[1]);
    N_ROWS = N_COLS;
    DFS = atoi(argv[2]);

    begin = clock();
    n_queens_solve();
    end = clock();

    printf("Generated boards: %d\n", GENERATED);
    time_spent = ((double) (end - begin)) / CLOCKS_PER_SEC;
    printf("Time spent: %.4fs\n", time_spent);

    delete_list(queue);

    return 0;
}

void n_queens_solve()
{
   int **board = init_board();
   List *node = NULL;
   pushBack(&queue, board);
   GENERATED++;

   int found = 0;
   int nqueens = 0;

   do {
       if (!is_empty(queue)) {
           node = popFront(&queue);
           board = node->board;
           nqueens = count_queens(board);
           
           if (nqueens == N_COLS) { 
                   found = 1;
           }
           else {
               generate_boards(board);
               delete_node(node);
           }
       }
       else {
           break;
       }
   } while (!found);

   if (found) {
       printf("Solve: \n");
       print_board(board);
   }
   else {
       printf("Solution not found.\n");
   }
}

void generate_boards(int **board)
{
    int i;
    int j;
    int **new;

    for (i = 0; i < N_ROWS; ++i) {
        for (j = 0; j < N_COLS; ++j) {
            if (board[i][j] == 0) {
                new = copyArray(board);
                placeQueen(new, i, j);

                if (DFS == 1)
                    pushFront(&queue, new);
                else
                    pushBack(&queue, new);

                GENERATED++;
            }
        }
    }
}

void placeQueen(int **board, int col, int row)
{
    int i;

    // Block vert and horiz
    for (i = 0; i < N_ROWS; ++i) {
        board[col][i] = 1;
        board[i][row] = 1;
    }

    // Block diags
    for (i = 0; i < N_ROWS; ++i) {
        if (col - i >= 0) {
            if (row - i >= 0)
                board[col - i][row - i] = 1;
            if (row + i < N_ROWS)
                board[col - i][row + i] = 1;
        }

        if (col + i < N_COLS) {
            if (row - i >= 0)
                board[col + i][row - i] = 1;
            if (row + i < N_ROWS)
                board[col + i][row + i] = 1;
        }
    }

    board[col][row] = 2;
}

int count_queens(int **board)
{
    int i;
    int j;
    int ret = 0;

    for (i = 0; i < N_COLS; ++i) {
        for (j = 0; j < N_ROWS; ++j) {
            if (board[i][j] == 2)
                ret++;
        }
    }

    return ret;
}

int check_collision(int **board, int row, int col)
{
    int ret = 1;
    int i;
    int j;

    // Check vert and horiz
    for (i = 0, j = 0; j < N_COLS; j++, i++) {
        if (j == col) 
            continue;

        if ((board[col][j] > 0) || (board[i][row] > 0)) {
            ret = 0;
        }
    }

    return ret;
}

int **init_board()
{
    int i;
    int **ret = calloc(N_COLS, sizeof(int *));

    for (i = 0; i < N_COLS; ++i) {
        ret[i] = calloc(N_ROWS, sizeof(int));
    }

    return ret;
}

void printNode(List *node)
{
    int i;
    int j;

    for (i = 0; i < N_COLS; ++i) {
        for (j = 0; j < N_ROWS; ++j) {
            printf("%d ", node->board[i][j]);
        }
        printf("\n");
    }
}

void print_board(int **board)
{
    int i;
    int j;

    for (i = 0; i < N_ROWS; ++i) {
        for (j = 0; j < N_COLS; ++j) 
            printf("%d ", board[i][j]);
        
        printf("\n");
    }
}


List *initList()
{
    return NULL;
}

void pushFront(List **head, int **data)
{
    List *new = createNode(data);

    if (NULL != *head)
    {
        new->next = *head; 
    }

    *head = new;
}

void delete_list(List *head)
{
    List *tmp;

    while (head != NULL) {
        tmp = head;
        head = head->next;
        delete_node(tmp);
    }
}

void pushBack(List **head, int **data)
{
    List *tmp = *head;
    List *prev = NULL;

    while (NULL != tmp)
    {
        prev = tmp;
        tmp = tmp->next; 
    }

    tmp = createNode(data);
    if (NULL != prev) {
        prev->next = tmp;
    }
    else {
        *head = tmp;
    }
}

List *popBack(List **head)
{
    List *tmp = *head;
    List *prev = NULL;

    while (NULL != tmp->next) {
        prev = tmp;
        tmp = tmp->next;
    }

    if (NULL != prev) {
        prev->next = NULL;
    }

    return tmp;
}

List *popFront(List **head)
{
    List *tmp = *head;

    if (NULL != tmp) {
        *head = tmp->next;
    }

    return tmp;
}

int is_empty(List *head)
{
    return (head == NULL);
}

void printList(List *head)
{
    while (NULL != head) {
        print_board(head->board);
        printf("\n");
        head = head->next;
    }
}

List *createNode(int **data) 
{
    List *list = malloc(sizeof(List));
    assert(list);
    list->next = NULL;
    list->board = copyArray(data);

    return list;
}

void delete_node(List *node)
{
    int i;
    if (node != NULL) {
        for (i = 0; i < N_COLS; ++i)
            free(node->board[i]);

        free(node->board);
        free(node);
    }
}

int **copyArray(int *data[])
{
    int i;
    int **ret = malloc(sizeof(int *) * N_COLS);
    for(i = 0; i < N_COLS; ++i) {
        ret[i] = malloc(sizeof(int) * N_ROWS);
        memcpy(ret[i], data[i], N_COLS * sizeof(int));

    }

    //ret = data;

    return ret;
}
