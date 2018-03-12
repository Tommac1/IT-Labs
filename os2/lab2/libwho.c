#include <grp.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#define MAX_DATA 1024

char *storeUserGroups(gid_t *groups, int ngroups);
gid_t *getUserGroups(uid_t uid, int *ngroups);

char *main_lib(uid_t uid)
{
    gid_t *groups = NULL;
    int ngroups = NGROUPS_MAX;
    char *buf = NULL;

    if (NULL != (groups = getUserGroups(uid, &ngroups))) {
        buf = storeUserGroups(groups, ngroups);
    }

    return buf;
}

char *storeUserGroups(gid_t *groups, int ngroups)
{
    struct group *gr;
    int i;
    char *temp = malloc(sizeof(char) * MAX_DATA);
    char *buf = malloc(sizeof(char) * MAX_DATA);
    if (!temp || !buf) {
        return NULL;
    }

    sprintf(buf, "[");
    for (i = 0; i < ngroups; ++i) {
        gr = getgrgid(groups[i]);
        sprintf(temp, "%s%s", gr->gr_name, ((i == ngroups - 1) ? "" : ", "));
        strcat(buf, temp);
    }
    strcat(buf, "]");
    free(temp);

    return buf;
}

gid_t *getUserGroups(uid_t uid, int *ngroups)
{
    struct passwd *pw;
    gid_t *groups = NULL;

    groups = malloc(sizeof(gid_t) * NGROUPS_MAX);
    if (!groups) {
        return NULL;
    }

    pw = getpwuid(uid);
    assert(pw);

    if (-1 == getgrouplist(pw->pw_name, pw->pw_gid, groups, ngroups)) {
        fprintf(stderr, "getgroupslist() error, ngroups: %d\n", *ngroups);
        free(groups);
        exit(EXIT_FAILURE);
    }

    return groups;
}

_init()
{

}

_fini()
{

}
