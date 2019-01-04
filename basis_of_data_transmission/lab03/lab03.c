#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define BYTE_TO_BIN_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BIN(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

struct sine_wave {
    double *values[2];
    int samples;
};

typedef double (*exc_fun)(double);
typedef void (*modulation)(struct sine_wave *, unsigned char);
typedef int (*cmp_fun)(double);

static int plot = 0;
static const double Tb = 1.0;
static const int N_BITS = 8;
static double fn; 
static const double freq_s = 1000.0;
static const double A1 = 1.0;
static const double A2 = 3.0;
static const double ask_h = 1000.0;
static const double psk_h = -100.0;

// =========================================================
// Helper functions
// =========================================================
unsigned char reverse_bits(unsigned char num) 
{ 
    unsigned int NO_OF_BITS = sizeof(num) * 8; 
    unsigned char reverse_num = 0; 
    int i; 
    for (i = 0; i < NO_OF_BITS; i++) 
    { 
        if((num & (1 << i))) 
            reverse_num |= 1 << ((NO_OF_BITS - 1) - i);   
    } 
    return reverse_num; 
} 

void cpy_sine_wave(struct sine_wave *dest, struct sine_wave *src)
{
    memcpy(dest->values[0], src->values[0], src->samples * sizeof(double));
    memcpy(dest->values[1], src->values[1], src->samples * sizeof(double));
}

double sine(double a, double fn, double t, double fi)
{
    return a * sin(2.0 * M_PI * fn * t + fi);
}

static void create_sine(struct sine_wave *sw, int samples)
{
    sw->values[1] = malloc(sizeof(double) * samples);
    sw->values[0] = malloc(sizeof(double) * samples);

    sw->samples = samples;

    assert(sw->values[0] && sw->values[1]);
}

static void generate_san(struct sine_wave *out, struct sine_wave *in)
{
    int i;
    for (i = 0; i < in->samples; ++i) {
        out->values[0][i] = in->values[0][i];
        out->values[1][i] = sine(A2, fn, in->values[0][i], 0.0);
    }
}

static void generate_square(double period, struct sine_wave *out, 
        unsigned char val)
{
    int i = 0;
    double step = 1.0 / freq_s * Tb ;
    double bit_step = Tb / freq_s;
    double bit_val = 0.0;
    double x_val = 0.0;

    val = reverse_bits(val);

    for (i = 0; i < out->samples; ++i) {
        bit_val += bit_step;
        x_val += step;
        if (bit_val > Tb) {
            val >>= 1;
            bit_val = 0.0;
        }
        out->values[0][i] = x_val;
        out->values[1][i] = (val & 1) ? 1.0 : 0.0;
    }
}

static void ask(struct sine_wave *in, unsigned char val)
{
    int i = 0;
    double a = 0.0;

    for (i = 0; i < in->samples; ++i) {
        if (abs(in->values[1][i]) <= 0.01) {
            a = A1;
        }
        else {
            a = A2;
        }

        in->values[1][i] = sine(a, fn, in->values[0][i], 0.0);
    }
}

static void psk(struct sine_wave *in, unsigned char val)
{
    int i = 0;
    double fi = 0.0;

    for (i = 0; i < in->samples; ++i) {
        if (abs(in->values[1][i]) <= 0.01) {
            fi = 0.0;
        }
        else {
            fi = M_PI;
        }

        in->values[1][i] = sine(1.0, fn, in->values[0][i], fi);
    }
}

static void multiply_sines(struct sine_wave *out, struct sine_wave *in,
        struct sine_wave *in2)
{
    int i;
    for (i = 0; i < out->samples; ++i) {
        out->values[0][i] = in->values[0][i];
        out->values[1][i] = in2->values[1][i] * in->values[1][i];
    }
}

static void integrate_sine(struct sine_wave *out, struct sine_wave *in)
{
    int i = 0;
    double bit_step = Tb / freq_s;
    double bit_val = 0.0;
    double val = 0.0;

    for (i = 0; i < out->samples; ++i) {
        bit_val += bit_step;
        if (bit_val > Tb) {
            bit_val = 0.0;
            val = 0.0;
        }
        val += in->values[1][i];
        out->values[0][i] = in->values[0][i];
        out->values[1][i] = val;
    }
}

static void compare_sine(struct sine_wave *out, struct sine_wave *in, cmp_fun cmp)
{
    int i;
    for (i = 0; i < out->samples; ++i) {
        out->values[0][i] = in->values[0][i];
        out->values[1][i] = cmp(in->values[1][i]) ? 1.0 : 0.0;
    }
}

static unsigned char extract_data(struct sine_wave *in)
{
    unsigned char value = 0;
    int i = 0;
    int ones = 0;
    int total = 0;

    for (i = 1; i <= in->samples; ++i) {
        if ((i % (in->samples / N_BITS)) == 0) {
            // next bit
            value <<= 1;
            value |= ((double)ones / total > 0.5) ? 1 : 0;
            ones = 0;
            total = 0;
            continue;
        }

        if (in->values[1][i] > 0.1) 
            ones++;
        total++;
    }
    
    return value;
}

static void free_sine_wave(struct sine_wave *sw)
{
    free(sw->values[0]);
    free(sw->values[1]);
    sw->values[0] = NULL;
    sw->values[1] = NULL;
}

static void print_sine(struct sine_wave *sw, int file, int ftw)
{
    int i = 0;
    FILE *fp;
    char filename[20];

    if (file > 9) {
        strncpy(filename, "./plotxx.csv", 20);
        filename[6] = (file / 10) + '0';
        filename[7] = file + '0';
    } 
    else {
        strncpy(filename, "./plotx.csv", 20);
        filename[6] = file + '0';
    }



    fp = fopen(filename, "w+");
    assert(fp);

    for (i = 0; i < sw->samples; ++i)
        fprintf(fp, "%.15f, %.15f\n", sw->values[0][i], sw->values[1][i]);

    fclose(fp);
}

static void dft_sum(double *out, int k, int N, double *x)
{
    double fi = 0.0;
    int n;

    for (n = 0; n < N; ++n) {
        fi = (2.0 * M_PI * (double)k * (double)n / (double)N);
        out[0] += (x[n] * cos(fi)); // Re
        out[1] += (-x[n] * sin(fi)); // Im
    }
}

static void dft(struct sine_wave *fw, struct sine_wave *sw, double fs)
{
    int i;
    double z[2] = { 0 };

    for (i = 0; i < fw->samples / 2 - 1; ++i) {
        dft_sum(z, i, fw->samples, sw->values[1]);
        fw->values[0][i] = ((double)i * (fs / (double)fw->samples));
        fw->values[1][i] = sqrt(pow(z[0], 2.0) + pow(z[1], 2.0));
        fw->values[1][i] = 10.0 * log10(fw->values[1][i]);
    }
    fw->samples /= 2;
    fw->samples--;
}

int ask_cmp(double val)
{
    return (val > ask_h);
}

int psk_cmp(double val)
{
    return (val < psk_h);
}

unsigned char modem(modulation mod, unsigned char val, cmp_fun cmp)
{
    struct sine_wave sw;
    struct sine_wave san;
    struct sine_wave za;
    struct sine_wave spctr_sw;
    struct sine_wave xt;
    struct sine_wave pt;
    struct sine_wave mt;
    unsigned char result = 0;

    create_sine(&sw, N_BITS * freq_s);
    create_sine(&san, N_BITS * freq_s);
    create_sine(&xt, N_BITS * freq_s);
    create_sine(&pt, N_BITS * freq_s);
    create_sine(&mt, N_BITS * freq_s);
    create_sine(&za, N_BITS * freq_s);
    create_sine(&spctr_sw, N_BITS * freq_s / 2.0);


    // BIT SIGNAL ===========
    generate_square(N_BITS, &sw, val);

    cpy_sine_wave(&za, &sw);


    // MODULATE =============
    mod(&za, val);
    print_sine(&za, plot++, 0);

    dft(&spctr_sw, &za, freq_s);
    print_sine(&spctr_sw, plot++, 0);


    generate_san(&san, &sw);
    
    // x(t)
    multiply_sines(&xt, &za, &san);
    print_sine(&xt, plot++, 0);

    // p(t)
    integrate_sine(&pt, &xt);
    print_sine(&pt, plot++, 0);

    // m(t)
    compare_sine(&mt, &pt, cmp);
    print_sine(&mt, plot++, 0);

    // byte_after
    result = extract_data(&mt);

    free_sine_wave(&sw);
    free_sine_wave(&san);
    free_sine_wave(&xt);
    free_sine_wave(&pt);
    free_sine_wave(&mt);
    free_sine_wave(&za);
    free_sine_wave(&spctr_sw);

    return result;
}

// =========================================================
// Main function
// =========================================================
int main(int argc, char *argv[])
{
    unsigned char byte_before = 0;
    unsigned char byte_after = 0;

    fn = 2.0 / Tb;

    srand(time(NULL));
    byte_before = rand() % 255;
    //byte_before = 114; // 0111 0010 
    

    // TODO: compare_ask() AND compare_psk() functions to compare between h
    // ASK ==================
    printf("Value before ASK modulation: %d, "BYTE_TO_BIN_PATTERN"b\n", 
            byte_before, BYTE_TO_BIN(byte_before));
    byte_after = modem(ask, byte_before, ask_cmp);
    printf("Value after ASK modulation: %d, "BYTE_TO_BIN_PATTERN"b\n", 
            byte_after, BYTE_TO_BIN(byte_after));

    // PSK ==================
    printf("Value before PSK modulation: %d, "BYTE_TO_BIN_PATTERN"b\n", 
            byte_before, BYTE_TO_BIN(byte_before));
    byte_after = modem(psk, byte_before, psk_cmp);
    printf("Value after PSK modulation: %d, "BYTE_TO_BIN_PATTERN"b\n", 
            byte_after, BYTE_TO_BIN(byte_after));


    return 0;
}
