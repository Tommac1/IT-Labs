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
char **argvCopyAndExtend(int argc, char **argv);
void printOutput(int argc, char **argv);
void run_child(int argc, char **argv, int which_child);

int main(int argc, char *argv[])
{
    char **argv_copy = NULL;
    if (2 > argc) {
        printf("Usage %s string\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    argv_copy = argvCopyAndExtend(argc, argv);

    if (2 == argc) {
        /*  Only the maser of the masters process */
        adjustStringArg(argv_copy[1]);
    }

    run(argc, argv_copy);

    return 0;
}

void run(int argc, char **argv)
{
    pid_t pid = 0;
    int last_arg_len = strlen(argv[argc - 1]);

    if (1 < last_arg_len) {

        if (0 != (pid = fork())) {
            // We are in the parent

            if (0 != (pid = fork())) {
                // We are int the parent
                while (0 < (pid = wait(NULL))) {
                    // No error handling.
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
    int last_arg_len = strlen(argv[argc - 1]);
    int child_len = (last_arg_len / 2);

    if (FIRST_CHILD == which_child) {
        initChildBuffer(&child_buf, argv[argc - 1], child_len);
    }
    else {
        initChildBuffer(&child_buf, argv[argc - 1] + child_len, child_len);
    }

    // Append half of the last argument.
    argv[argc] = child_buf;

    execve(argv[0], argv, NULL);
}

char **argvCopyAndExtend(int argc, char **argv)
{
    int i;
    size_t length = 0;
    char **ret = malloc(sizeof (char *) * (argc + 2));
    ret[argc + 1] = NULL;
    ret[argc] = NULL;
    for (i = 0; i < argc; ++i) {
        length = strlen(argv[i]) + 1;
        ret[i] = malloc(length);
        memcpy(ret[i], argv[i], length);
    }

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
