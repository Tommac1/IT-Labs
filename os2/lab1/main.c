/*
 * This program lists acitve logged users.
 * Available opts:
 *  -g  lists all groups the user belongs to
 *  -h  show users remote host
 *
 * For example:
 *  $ ./lab1 -gh
 *  tomek [users, virtual, students] (82.145.204.2)
 *  andrzej [users, virtual, lecturers] (82.205.182.45)
 */



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <utmp.h>
#include <assert.h>
#include <grp.h>
#include <pwd.h>
#include <limits.h>

#define FLAG_OFF    0
#define FLAG_ON     1


// Host flag
int H_FLAG = FLAG_OFF;
// Groups flag
int G_FLAG = FLAG_OFF;


struct utmp *initUtmp();
void processOptions(int argc, char *argv[]);
void getEntries(struct utmp *ut);
void processEntry(struct utmp *ut); 
gid_t *getUsersGroups(char *user);
void printUserGroups(gid_t *groups, int ngroups);
gid_t *getUserGroups(const char *user, int *ngroups);
void removeUtmp(struct utmp *ut);


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
    struct utmp *ut= malloc(sizeof (struct utmp));
    assert(ut);
    return ut;
}

void removeUtmp(struct utmp *ut)
{
    if (NULL != ut)
    {
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
    gid_t *groups = NULL;
    int ngroups = NGROUPS_MAX;

    printf("%s ", ut->ut_user);

    if (1 == G_FLAG) {
        groups = getUserGroups(ut->ut_user, &ngroups);
        printUserGroups(groups, ngroups);
        free(groups);
    }

    if (1 == H_FLAG) {
        printf(" (%s)", ut->ut_host);
    }

    printf("\n");
}

void printUserGroups(gid_t *groups, int ngroups)
{
    struct group *gr;
    int i;

    printf("[");
    for (i = 0; i < ngroups; ++i) {
        gr = getgrgid(groups[i]);
        printf("%s%s", gr->gr_name, ((i == ngroups - 1) ? "" : ", "));
    }
    printf("]");
}

gid_t *getUserGroups(const char *user, int *ngroups)
{
    struct passwd *pw;
    gid_t *groups = NULL;

    groups = malloc(sizeof(gid_t) * NGROUPS_MAX);
    assert(groups);

    pw = getpwnam(user);
    assert(pw);

    if (-1 == getgrouplist(user, pw->pw_gid, groups, ngroups)) {
        fprintf(stderr, "getgrouplist() error, ngroups: %d\n", *ngroups);
        free(groups);
        exit(EXIT_FAILURE);
    }

    return groups;
}

void processOptions(int argc, char *argv[])
{
    int ret = 0;

    while (-1 != (ret = getopt(argc, argv, "hg"))) {
        switch (ret) {
            case 'g': G_FLAG = FLAG_ON; break;
            case 'h': H_FLAG = FLAG_ON; break;
            case '?':
                fprintf(stderr, "Unknown option: -%c.\n", optopt);
                break;
            default: abort(); break;
        }
    }
}
