#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

int getMSB(int num);
void cutString(char *str, int msb);
void adjustStringArg(char *arg);
void run(int argc, char **argv);
void initChildBuffer(char **child, char *arg, int arg_len);


int main(int argc, char *argv[])
{
    if (2 > argc) {
        printf("Usage %s string\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (2 == argc) {
        /*  Only the maser of the masters process */
        adjustStringArg(argv[1]);
    }

    run(argc, argv);

    return 0;
}

void run(int argc, char **argv)
{
    int wstatus = 0;
    pid_t pid = 0;
    pid_t pid2 = 0;
    int arg_len = strlen(argv[argc - 1]);
    int child_buf_len = (arg_len / 2);
    char *child1_buf;
    char *child2_buf;
   
    printf("%d\n", arg_len);

    initChildBuffer(&child1_buf, argv[argc - 1], child_buf_len);
    initChildBuffer(&child2_buf, argv[argc - 1] + child_buf_len, child_buf_len);

    printf("%s\n%s\n", child1_buf, child2_buf);

    if (1 < arg_len) {
        if (0 != (pid = fork())) {
            // We are in the parent
            printf("1st child pid: %d\n", pid);

            if (0 != (pid2 = fork())) {
                // We are in the parent
                printf("2nd child pid: %d\n", pid2);
                pid = getpid();
                printf("parent pid: %d\n", pid);
                /* 
                if (0 > wait(&wstatus)) {
                    perror("wait");
                }
                printf("wstatus: %d\n", wstatus);
                */
            }
            else {
                // We are in the second child
                pid = getpid();
                printf("2nd child pid: %d\n", pid);
            }
        }
        else {
            // We are in the first child.
            pid = getpid();
            printf("1st child pid: %d\n", pid);
        }
    }

    wait(NULL);
    printf("dupa\n");
}

void initChildBuffer(char **child, char *arg, int arg_len) {
    *child = malloc(sizeof(char) * (arg_len + 1));
    assert(child);
    *child = strncpy(*child, arg, arg_len);
    child[arg_len] = '\0';
}


/* Cut the string to be length of the biggest power of two 
 * lower than original length */
void adjustStringArg(char *arg)
{
    int strl;
    int msb;
    strl = strlen(arg);
    if (1 < strl) {
        msb = getMSB(strl);
        cutString(arg, msb);
    }
}

int getMSB(int num)
{
    int msb_pos = 0;

    while (0 != num) {
        msb_pos++;
        num >>= 1;
    }
   
    // Zero indexed.
    return msb_pos - 1;
}

void cutString(char *str, int msb)
{
    str[(1 << msb)] = '\0';
}
