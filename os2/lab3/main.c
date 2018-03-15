#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#define FIRST_CHILD  1
#define SECOND_CHILD 2


int getMSB(int num);
void cutString(char *str, int msb);
void adjustStringArg(char *arg);
void run(int argc, char **argv);
void initChildBuffer(char **child, char *arg, int arg_len); 
char **argvCopy(int argc, char **argv);
void printOutput(int argc, char **argv);
void run_child(int argc, char **argv, int which_child);

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
    int last_arg_len = strlen(argv[argc - 1]);

    if (1 < last_arg_len) {

        if (0 != (pid = fork())) {
            // We are in the parent

            if (0 != (pid = fork())) {
                // We are int the parent
                while (0 < (pid = wait(&wstatus))) {
                    printf("pid %d ended status: %d\n", pid, wstatus);
                   if (WIFEXITED(wstatus)) {
                       printf("exited, status=%d\n", WEXITSTATUS(wstatus));
                   } else if (WIFSIGNALED(wstatus)) {
                       printf("killed by signal %d\n", WTERMSIG(wstatus));
                   } else if (WIFSTOPPED(wstatus)) {
                       printf("stopped by signal %d\n", WSTOPSIG(wstatus));
                   } else if (WIFCONTINUED(wstatus)) {
                       printf("continued\n");
                   }
                }

            }
            else {
                // We are in the second child
                run_child(argc, argv, SECOND_CHILD);
            }
        }
        else {
            // We are in the first child.
            run_child(argc, argv, FIRST_CHILD);
        }
    }
    
    printOutput(argc, argv);

    //free(child1_buf);
    //free(child2_buf);
    //free(argv);
}

void run_child(int argc, char **argv, int which_child)
{
    char *child_buf;
    char **argv_copy;
    int last_arg_len = strlen(argv[argc - 1]);
    int child_len = (last_arg_len / 2);

    if (FIRST_CHILD == which_child) {
        initChildBuffer(&child_buf, argv[argc - 1], child_len);
    }
    else {
        initChildBuffer(&child_buf, argv[argc - 1] + child_len, child_len);
    }

    argv_copy = argvCopy(argc, argv);
    argv_copy[argc] = child_buf;

    printf("child %d las_arg %d child_len %d ", which_child, last_arg_len, child_len);
    int i;
    for (i = 0; i <= argc; ++i)
        printf("%s ", argv_copy[i]);

    printf("\n");

    execv(argv[0], argv_copy);
}

char **argvCopy(int argc, char **argv)
{
    int i;
    char **ret = malloc(sizeof (char *) * (argc + 2));
    ret[argc + 1] = NULL;
    for (i = 0; i < argc; ++i)
        memcpy(&ret[i], &argv[i], strlen(argv[i]));

    return ret;
}

void printOutput(int argc, char **argv)
{
    pid_t pid;
    int i;

    pid = getpid();
    printf("%d ", pid);
    for (i = 1; i < argc; ++i)
        printf("%s ", argv[i]);

    printf("\n");
}

void initChildBuffer(char **child, char *arg, int arg_len) 
{
    *child = malloc(sizeof(char) * (arg_len + 1));
    assert(child);
    *child = strncpy(*child, arg, arg_len);
    (*child)[arg_len] = '\0';
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
