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

void my_closedir(DIR *dirp);
void my_readdir(DIR *dirp, int level, char *relative_path);
void processOptions(int argc, char *argv[]);
int check_dotted_dirs(const char *name);
DIR *my_opendir(const char *path);
char *append_dir_path(char *path, char *dir_name);
void process_dir_entry(struct dirent *de, int level, char *relative_path);
void prepare_and_open_dir(struct dirent *de, char *relative_path, int level);
void process_link(struct dirent *de, char *relative_path, int level);
int check_if_dir(char *path);
void print_name(char *filename, char *relative_path, int level);


int main(int argc, char *argv[])
{
    DIR *dirp;
    int init_level;

    processOptions(argc, argv);
    init_level = BIG_L_FLAG;

    if (NULL == argv[optind]) {
        dirp = my_opendir(".");
        printf("%s\n", ".");
        my_readdir(dirp, init_level, ".");
    }
    else {
        dirp = my_opendir(argv[optind]);
        printf("%s\n", argv[optind]);
        my_readdir(dirp, init_level, argv[optind]);
    }

    my_closedir(dirp);

    printf("\n%d directories, %d files\n", DIR_NUM, FILE_NUM);
    return 0;
}

void my_readdir(DIR *dirp, int level, char *relative_path)
{
    struct dirent *de;

    while (NULL != (de = readdir(dirp))) {
        if (0 == check_dotted_dirs(de->d_name)) {
            process_dir_entry(de, level, relative_path);
        }
    }
}

void process_dir_entry(struct dirent *de, int level, char *relative_path)
{
    char *buf;
    ssize_t len = 0;
    int target_is_dir = 0;

    if (DT_DIR == de->d_type) {
        print_name(de->d_name, relative_path, level);
        printf("\n");

        DIR_NUM++;

        if (1 != level) {
            prepare_and_open_dir(de, relative_path, level);
        }

    }
    else if (FLAG_OFF == D_FLAG) {
        print_name(de->d_name, relative_path, level);

        FILE_NUM++;

        if (DT_LNK == de->d_type) {
            // We do not know, if the target file is dir or reg file
            // Counting will happen in process_link()
            FILE_NUM--;
            process_link(de, relative_path, level);
        }

        printf("\n");

    }
    else if (DT_LNK == de->d_type) {
        buf = malloc(sizeof(char) * PATH_MAX);
        len = readlink(de->d_name, buf, PATH_MAX);
        buf[len] = '\0';

        target_is_dir = check_if_dir(buf);
        if (target_is_dir) {
            DIR_NUM++;

            print_name(de->d_name, relative_path, level);
            printf(" -> %s\n", buf);

            if ((L_FLAG == FLAG_ON) && (1 != level)) {
                prepare_and_open_dir(de, relative_path, level);
            }
        }

        free(buf);
    }
}

void print_name(char *filename, char *relative_path, int level)
{
    int i;
    // Indent
    for (i = BIG_L_FLAG; i >= level; --i)
        printf("    ");

    // Print relative path if flag is on
    if (F_FLAG == FLAG_ON)
        printf("%s/", relative_path);

    printf("%s", filename);
}

void prepare_and_open_dir(struct dirent *de, char *relative_path, int level)
{
    char *new_path;
    DIR *dirp_new;

    new_path = append_dir_path(relative_path, de->d_name);

    dirp_new = my_opendir(new_path);
    my_readdir(dirp_new, level - 1, new_path);
    my_closedir(dirp_new);
}

void process_link(struct dirent *de, char *relative_path, int level)
{
    char *buf;
    int target_is_dir = 0;
    ssize_t len = 0;

    printf(" -> ");
    buf = malloc(sizeof(char) * PATH_MAX);
    len = readlink(de->d_name, buf, PATH_MAX);
    buf[len] = '\0';
    printf("%s", buf);

    target_is_dir = check_if_dir(buf);
    if (target_is_dir) {
        DIR_NUM++;

        if ((L_FLAG == FLAG_ON) && (1 != level)) {
            printf("\n");
            prepare_and_open_dir(de, relative_path, level);
        }
    }
    else {
        FILE_NUM++;
    }

    free(buf);
}

int check_if_dir(char *path)
{
    struct stat buf;
    int ret = 0;
    
    if (0 > stat(path, &buf)) {
        printf("%s ", path);
        perror("stat");
        exit(EXIT_FAILURE);
    }

    if ((buf.st_mode & S_IFMT) == S_IFDIR) {
        ret = 1;
    }

    return ret;
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

    return full_path;
}

DIR *my_opendir(const char *path)
{
    DIR *dirp;
    if (NULL == (dirp = opendir(path))) {
        printf("%s ", path);
        perror("opendir");
        exit(EXIT_FAILURE);
    }
    return dirp;
}

void my_closedir(DIR *dirp)
{
    int ret = 0;
    if (NULL != dirp) {
        ret = closedir(dirp);
        if (0 != ret) {
            perror("closedir");
            exit(EXIT_FAILURE);
        }
    }
}

int check_dotted_dirs(const char *name)
{
    int ret = 0;

    if ((0 == strcmp(name, ".")) || (0 == strcmp(name, ".."))) {
        ret = 1;
    }

    return ret;
}

void processOptions(int argc, char *argv[])
{
    int ret = 0;

    while (-1 != (ret = getopt(argc, argv, "dlfL:"))) {
        switch (ret) {
            case 'd': D_FLAG = FLAG_ON; break;
            case 'l': L_FLAG = FLAG_ON; break;
            case 'f': F_FLAG = FLAG_ON; break;
            case 'L': BIG_L_FLAG = atoi(optarg); break;
            case '?':
                if ('L' == optopt)
                    fprintf(stderr, "Option -%c requires an argument"
                                    "greater than zero.\n", optopt);
                break;
            default: abort(); break;
        }
    }
}

