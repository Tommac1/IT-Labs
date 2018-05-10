#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <crypt.h>


int main(int argc, char *argv[])
{
    struct crypt_data cdata;
    char *salt = NULL;
    char *pass;

    if (argc != 3) {
        printf("Usage: %s pass salt\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    salt = malloc(sizeof(char) * 128);
    assert(salt);

    salt = strncpy(salt, "$6$", 4);
    salt = strncat(salt, argv[2], 124);

    cdata.initialized = 0;
    pass = crypt_r(argv[1], salt, &cdata);

    printf("%s\n", pass);

    free(salt);

    return 0;
}
