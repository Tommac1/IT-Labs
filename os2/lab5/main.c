#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>

#define FLAG_OFF 0
#define FLAG_ON  1

int D_FLAG = FLAG_OFF;
int L_FLAG = FLAG_OFF;
int F_FLAG = FLAG_OFF;
int BIG_L_FLAG = FLAG_OFF;
int DIR_NUM = 0;
int FILE_NUM = 0;

void my_readdir(DIR *dirp, int level, char *relative_path);
int check_dotted_dirs(const char *name);
DIR *_opendir(const char *path);
char *append_dir_path(char *path, char *dir_name);

int main(int argc, char *argv[])
{
    DIR *dirp;

    if (2 != argc) {
        printf("%s dir\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    dirp = _opendir(argv[1]);

    printf("%s\n", argv[1]);
    my_readdir(dirp, 1, argv[1]);

    printf("%d directories, %d files\n", DIR_NUM, FILE_NUM);
    return 0;
}

void my_readdir(DIR *dirp, int level, char *relative_path)
{
    struct dirent *de;
    DIR *dirp_new;
    char *new_path;
    int i;

    while (NULL != (de = readdir(dirp))) {
        if (0 == check_dotted_dirs(de->d_name)) {
            for (i = 0; i < level; ++i)
                printf("  ");

            printf("%s\n", de->d_name);
            FILE_NUM++;

            if (DT_DIR == de->d_type) {
                new_path = append_dir_path(relative_path, de->d_name);
                FILE_NUM--;
                DIR_NUM++;
                dirp_new = _opendir(new_path);
                my_readdir(dirp_new, level + 1, new_path);
            }
        }
    }
}

char *append_dir_path(char *path, char *dir_name)
{
    const size_t path_len = strlen(path);
    const size_t dir_name_len = strlen(dir_name);
    char *full_path = malloc(sizeof(char) * (path_len + dir_name_len + 1));
    assert(full_path);

    full_path[0] = '\0';
    strcpy(full_path, path);
    full_path[path_len] = '/';
    full_path[path_len + 1] = '\0'; 
    
    strcat(full_path, dir_name);

//    printf("append_dir_path: %s\n", full_path);

    return full_path;
}

DIR *_opendir(const char *path)
{
    DIR *dirp;
    if (NULL == (dirp = opendir(path))) {
        printf("%s ", path);
        perror("opendir");
        exit(EXIT_FAILURE);
    }
    return dirp;
}

int check_dotted_dirs(const char *name)
{
    int ret = 0;

    if ((0 == strcmp(name, ".")) || (0 == strcmp(name, ".."))) {
        ret = 1;
    }

    return ret;
}
