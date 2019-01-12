#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4
#define K 7

static int *code_hamming(int *data, int n, int k)
{
    int *out = malloc(sizeof(int) * k);
    memset(out, 0, sizeof(int) * k);

    out[0] = data[0];
    out[1] = data[1];
    out[2] = data[2];
	out[3] = (out[0] + out[1] + out[2]) % 2;
    out[4] = data[3];
	out[5] = (out[0] + out[1] + out[4]) % 2;
	out[6] = (out[0] + out[2] + out[4]) % 2;

    return out;
}

void code()
{
	int *in = malloc(sizeof(int) * N);
    int *out;
	
	printf("CODING\n insert 4 bits of data: ");
	scanf(" %d %d %d %d", &in[0], &in[1], &in[2], &in[3]);

    out = code_hamming(in, N, K);
	
	printf("after codding: %d %d %d %d %d %d %d.\n", out[0], out[1], out[2],
			out[3], out[4], out[5], out[6]);

    free(in);
    free(out);
}

static void calc_xd(int *in, int *xd)
{
	xd[0] = (in[2] + in[4] + in[6]) % 2;
	xd[1] = (in[2] + in[5] + in[6]) % 2;
	xd[2] = (in[4] + in[5] + in[6]) % 2;
}

static void calc_xk(int *in, int *xd, int *xk)
{
	xk[0] = (xd[0] + in[0]) % 2;
	xk[1] = (xd[1] + in[1]) % 2;
	xk[2] = (xd[2] + in[3]) % 2;
}

static void calc_p(int *xk, int *p)
{
	*p = 1 * xk[0] + 2 * xk[1] + 4 * xk[2];
}

static int *decode_hamming(int *data, int n, int k, int *succ, int *bit_corrected)
{
	int xd[3] = { 0 }; // iks z daszkiem
	int xk[3] = { 0 }; // iks z kreska
    int *out = malloc(sizeof(int) * n);
	int p;

    calc_xd(data, xd);
    calc_xk(data, xd, xk);
    calc_p(xk, &p);
	
	if (p != 0) {
        data[p - 1] = !data[p - 1];
	    *bit_corrected = p;	

        calc_xd(data, xd);
        calc_xk(data, xd, xk);
        calc_p(xk, &p);
	}
	
	if (p == 0) {
        *succ = 1;
		out[0] = data[6];
		out[1] = data[5];
		out[2] = data[4];
		out[3] = data[2];
	}
	else {
        *succ = 0;
	}

    return out;
}

void decode()
{
	int *in = malloc(sizeof(int) * K);
    int *out;
    int succ;
    int bit_corrected = -1; 

	printf("DECODDING\n insert 7 bits of data: ");
	scanf(" %d %d %d %d %d %d %d", &in[6], &in[5], &in[4], &in[3], &in[2], &in[1], &in[0]);
	
    out = decode_hamming(in, N, K, &succ, &bit_corrected);

    if (succ) {
        if (bit_corrected != -1) {
            printf("%d%s bit corrected.\n", bit_corrected,
                    ((bit_corrected == 1) ? "st" : 
                    ((bit_corrected == 2) ? "nd" : 
                    ((bit_corrected == 3) ? "rd" : "th"))));
        }

        printf("after decodding: %d %d %d %d.\n", out[0], out[1], out[2],
                out[3]);
    }
    else {
		printf("more than 1 error: can't decode'\n");
    }
	

    free(in);
    free(out);
}

int main(int argc, char *argv[])
{
    code();

    decode();
	
	return 0;
}
