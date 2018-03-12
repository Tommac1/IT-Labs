/*
 * This program lists actie logged users using prepared 
 * shared library libwho.so
 * Available opts:
 *  -g  lists all groups the user belongs to
 *  -h  show users remote host
 *
 *  For example:
 *   $./lab2 -gh
 *   tomek [users, virtual, student] (82.145.204.2)
 *   andrzej [users, virtual, lecturer] (82.205.182.45)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <utmp.h>
#include <assert.h>
#include <limits.h>
#include <dlfcn.h>
#include <pwd.h>

#define FLAG_OFF    0
#define FLAG_ON     1
#define GROUPS_LIB_PATH "./libwho.so"
#define LIB_MAIN_FUNC   "main_lib"

// Host flag
int H_FLAG = FLAG_OFF;
// Groups flag
int G_FLAG = FLAG_OFF;

struct utmp *initUtmp();
void processOptions(int argc, char *argv[]);
void getEntries(struct utmp *ut);
void processEntry(struct utmp *ut);
void printUserGroups(gid_t *groups, int ngroups);
void removeUtmp(struct utmp *ut);
gid_t *getUserGroups(const char *user, int *ngroups);
uid_t getUserUid(const char *user);
char *loadGroupsLib(uid_t uid);

char *(*main_lib)(uid_t);


int main(int argc, char *argv[])
{
    struct utmp *ut = NULL;

    if (1 != argc)
        processOptions(argc, argv);

   ut = initUtmp(); 
   getEntries(ut);
   removeUtmp(ut);

   return 0;
}

struct utmp *initUtmp()
{
    struct utmp *ut = malloc(sizeof(struct utmp));
    assert(ut);
    return ut;
}

void removeUtmp(struct utmp *ut)
{
    if (NULL != ut) {
        free(ut);
        ut = NULL;
    }
}

void getEntries(struct utmp *ut)
{
    setutent();

    while (NULL != (ut = getutent())) {
        if (USER_PROCESS == ut->ut_type) {
            processEntry(ut);
        }
    }

    endutent();
}

void processEntry(struct utmp *ut)
{
    char *groups = NULL;
    uid_t uid = 0;

    printf("%s ", ut->ut_user);

    if (1 == G_FLAG) {
        uid = getUserUid(ut->ut_user);
        groups = loadGroupsLib(uid);
        if (NULL != groups) {
            printf("%s", groups);
            free(groups);
        }
    }

    if (1 == H_FLAG) {
        printf(" (%s)", ut->ut_host);
    }

    printf("\n");
}

uid_t getUserUid(const char *user)
{
    struct passwd *pw; 
    uid_t uid = 0;
    setpwent();

    pw = getpwnam(user);
    uid = pw->pw_uid;

    endpwent();

    return uid;
}

char *loadGroupsLib(uid_t uid)
{
    char *buf = NULL;

    void *handler = dlopen(GROUPS_LIB_PATH, RTLD_LAZY);
    
    if (NULL != handler) {
        main_lib = dlsym(handler, LIB_MAIN_FUNC);
        if (main_lib) {
            buf = main_lib(uid);
        }
        else {
            buf = dlerror();
            printf("dlsym: %s\n", buf);
        }
        dlclose(handler);
    }

    return buf;
}

void processOptions(int argc, char *argv[])
{
    int ret = 0;

    while (-1 != (ret = getopt(argc, argv, "hg"))) {
        switch (ret) {
            case 'g': G_FLAG = FLAG_ON; break;
            case 'h': H_FLAG = FLAG_ON; break;
            case '?':
                fprintf(stderr, "Unknown option: -%c\n", optopt);
                break;
            default: abort(); break;
        }
    }
}









