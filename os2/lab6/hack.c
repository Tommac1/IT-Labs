#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <crypt.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>

#define BILLION  1E9
#define MAX_DATA 8192 

struct hack_env {
    char *shadow_password;
    char *filename;
    int desired_thread_num;
};


int init_env(int argc, char *argv[], struct hack_env *he);
void crack_password(struct hack_env *he);
void calculate_optimal_pthreads_num(struct hack_env *he);
size_t calculate_filesize(const char *filename);
int calculate_processors();
void spawn_workers(int pthread_num, int filesize);
void *worker_main_fun(void *arg);
void extract_salt(char *pass);
void check_password(char *line);

static double FILE_PASSED = 0.0;
static double FILE_PASSED_FIRST_CHECKPOINT = 0.0;
static char *SALT;
static char *PASSWORD_SHADOW;
static char *FILE_BUFF;
static int IS_PASS_FOUND = 0;
static char *PASSWORD_FOUND;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char *argv[])
{
    struct hack_env he;
    int env_inited = 0; 

    env_inited = init_env(argc, argv, &he);
    // env_init returns:
    // -1 -> metric mode (calculate optimal threads number)
    // 0 -> normal mode
    // 1 -> error

    if (env_inited == 0) {
        crack_password(&he);

        if (IS_PASS_FOUND == 1) {
            printf("Bruteforce succeed. Password: %s\n", PASSWORD_FOUND);
            goto exit_main_password_found;
        }
        else {
            printf("Bruteforce failed. Password not found in dictionary \"%s\".\n", he.filename);
        }

        goto exit_main_env_inited;
    }
    else if (env_inited == -1) {
        calculate_optimal_pthreads_num(&he);
        goto exit_main_env_inited;
    }
    else {
        goto exit_main;
    }

exit_main_password_found:
    free(PASSWORD_FOUND);

exit_main_env_inited:
    free(he.shadow_password);
    free(he.filename);
    free(SALT);
    free(PASSWORD_SHADOW);

exit_main:
    return 0;
}

void calculate_optimal_pthreads_num(struct hack_env *he)
{
    struct timespec time_start; 
    struct timespec time_end; 
    double time_spent = 0.0;
    int filesize = 7000;
    int fd;
    int i;

    fd = open(he->filename, O_RDWR);

    if (-1 == fd) {
        perror("open");
        goto calc_exit;
    }

    FILE_BUFF = mmap(0, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    if ((void *)-1 == FILE_BUFF) {
        perror("mmap");
        goto calc_opened;
    }

    // Surpress percent progress output
    FILE_PASSED = 10000.10000;
    FILE_PASSED_FIRST_CHECKPOINT = 10000.10000;

    for (i = 1; i <= 10; ++i) {

        // Calculate time taken by a request
        clock_gettime(CLOCK_REALTIME, &time_start);

        printf("Spawning %d pthread%s. ", i, (i == 1) ? "" : "s");
        spawn_workers(i, filesize);

        clock_gettime(CLOCK_REALTIME, &time_end);

        // Calculate time it took
        time_spent = (time_end.tv_sec - time_start.tv_sec) 
            + (time_end.tv_nsec - time_start.tv_nsec) / BILLION;

        printf("Time spent: %lfs\n", time_spent);
    }

//calc_mapped:
    munmap(FILE_BUFF, filesize);

calc_opened:
    close(fd);

calc_exit:
    return;

}

void crack_password(struct hack_env *he)
{
    int filesize = 0;
    int processors_num = 0;
    int fd;

    filesize = calculate_filesize(he->filename);

    if (-1 == filesize)
        goto crack_exit;

    fd = open(he->filename, O_RDWR);

    if (-1 == fd) {
        perror("open");
        goto crack_exit;
    }

    FILE_BUFF = mmap(0, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    if ((void *)-1 == FILE_BUFF) {
        perror("mmap");
        goto crack_opened;
    }

    processors_num = calculate_processors();

    if (processors_num < he->desired_thread_num) {
        he->desired_thread_num = processors_num;
    }

    if (he->desired_thread_num == 0) {
        he->desired_thread_num = processors_num;
        printf("0 pthread number given. Defaulting to number of processors.\n");
    }

    printf("Spawning %d pthreads.\n", he->desired_thread_num);
    spawn_workers(he->desired_thread_num, filesize);

//crack_mapped:
    munmap(FILE_BUFF, filesize);

crack_opened:
    close(fd);

crack_exit:
    return;
}

void spawn_workers(int pthread_num, int filesize)
{
    pthread_t *workers;
    int pthread_create_ret = 0;
    int i;
    int *args;
    // file_step indicates size of block for each pthread
    int file_step = filesize / pthread_num;
    // Threshold indicates border between two pthreads
    int lower_threshold = 0;
    int upper_threshold = 0;

    workers = malloc(sizeof(pthread_t) * pthread_num);
    assert(workers);

    pthread_mutex_init(&mutex, NULL);

    // Spawn new threads
    for (i = 0; i < pthread_num; ++i) {
        args = malloc(sizeof(int) * 2);

        lower_threshold = upper_threshold;
        upper_threshold += file_step;

        while ((FILE_BUFF[upper_threshold] != '\n') && (upper_threshold < filesize)) {
            upper_threshold++;
        }

        if (upper_threshold > filesize)
            upper_threshold = filesize;

        args[0] = lower_threshold;
        args[1] = upper_threshold;

//        printf("Spawning thread #%d with [%d, %d].\n", i, lower_threshold, upper_threshold);
        pthread_create_ret = pthread_create(&workers[i], NULL, worker_main_fun, args); 
        // args are freed in worker_main();
        
        if (pthread_create_ret != 0) {
            fprintf(stderr, "Cannot create pthread %d.\n", i);
            exit(100);
        }
        upper_threshold++; // Forward after newline char
    }

    // Wait for all threads complete.
    for (i = 0; i < pthread_num; ++i) {
        pthread_join(workers[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    free(workers);
}

void *worker_main_fun(void *arg)
{
    double block_processed = 0.0;
    // Casting to get rid of compiler warnings about dereferencing void ptr
    int begin = ((int *)arg)[0];
    int end = ((int *)arg)[1];
    int last_bytes_read = 0;
    int bytes_read = 0;
    int begin_line = begin;
    int line_len = 0;
    char c;
    //printf("Spawned %lu begin %d end %d first 10 chars %.10s\n", pthread_self(), begin, end, (FILE_BUFF + begin));

    free(arg);

    while (((begin + bytes_read) <= end) && (IS_PASS_FOUND == 0)) {
        ++line_len;
        ++bytes_read;
        c = FILE_BUFF[bytes_read + begin];

        if (c == '\n') {
            //line = strncpy(line, FILE_BUFF + begin_line, line_len - 1);
            pthread_mutex_lock(&mutex);
            FILE_BUFF[begin_line + line_len - 2] = '\0';
            //printf("%lu %s\n", pthread_self(), FILE_BUFF + begin_line);
            pthread_mutex_unlock(&mutex);
            
            check_password(FILE_BUFF + begin_line);

            begin_line = bytes_read + 1 + begin;
            line_len = 0;
        }

        if (bytes_read > last_bytes_read + 5000) {
            block_processed = (double)bytes_read/(double)(end - begin);

            if (abs(FILE_PASSED_FIRST_CHECKPOINT) < 0.0001) {
                FILE_PASSED_FIRST_CHECKPOINT = FILE_PASSED;
            }

            pthread_mutex_lock(&mutex);
            if (block_processed > (FILE_PASSED + (0.1 * FILE_PASSED_FIRST_CHECKPOINT))) {
                printf("File processed: %.4f%%.\n", (block_processed * 100));
                FILE_PASSED = block_processed;
            }
            pthread_mutex_unlock(&mutex);
        
            last_bytes_read = bytes_read;
        }
    }

    return NULL;
}

void check_password(char *line)
{
    struct crypt_data cdata;
    char *pass;
    cdata.initialized = 0;

    pass = crypt_r(line, SALT, &cdata); 

    if (0 == (strcmp(pass, PASSWORD_SHADOW))) {
        pthread_mutex_lock(&mutex);
        IS_PASS_FOUND = 1;
        PASSWORD_FOUND = strdup(line);
        pthread_mutex_unlock(&mutex);
    }
}

size_t calculate_filesize(const char *filename)
{
    struct stat st;
    if (0 != stat(filename, &st)) {
        perror("stat");
        st.st_size = -1;
    }
    return st.st_size;
}

int calculate_processors()
{
    int ret;
    
    ret = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Number of processors found: %d\n", ret);

    return ret;
}

int init_env(int argc, char *argv[], struct hack_env *he)
{
    int ret = 0;
    he->shadow_password = malloc(sizeof(char) * MAX_DATA);
    he->filename = malloc(sizeof(char) * MAX_DATA);

    ret = 0;

    if ((argc == 4) || (argc == 3)) {
        if ((MAX_DATA > strlen(argv[2])) || (MAX_DATA > strlen(argv[1]))) {
            he->shadow_password = strncpy(he->shadow_password, argv[1], MAX_DATA);
            he->filename = strncpy(he->filename, argv[2], MAX_DATA);

            if (argc != 3) {
                he->desired_thread_num = atoi(argv[3]);
            }

            extract_salt(argv[1]);
        }
        else {
            fprintf(stderr, "Too long args. Max is %d.\n", MAX_DATA);
            ret = 1;
        }

        ret = argc - 4;
    }
    else {
        fprintf(stderr, "Usage: %s password filename NUM_THREADS\n", argv[0]);
        ret = 1;
    }

    return ret;
}

void extract_salt(char *pass)
{
    int i = 0;
    int dollars_passed = 0;
    SALT = malloc(sizeof(char) * MAX_DATA);
    PASSWORD_SHADOW = malloc(sizeof(char) * MAX_DATA);

    while (dollars_passed < 3 && pass[i] != '\0') {
        if (pass[i] == '$')
            ++dollars_passed;

        ++i;
    }

    if (pass[i] == '\0') {
        fprintf(stderr, "Error extracting salt. line: %s\n", pass);
        exit(100);
    }

    PASSWORD_SHADOW = strncpy(PASSWORD_SHADOW, pass, strlen(pass) + 1);
    SALT = strncpy(SALT, pass, i - 1);
    SALT[i - 1] = '\0';
}
