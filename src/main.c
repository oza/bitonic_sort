/*-------------------------------------------------------------------------
 *
 * main.c - 20110612 v0.1.3
 *	  test codes for bitonic sorter (bitonic_sort.c)
 *
 *-------------------------------------------------------------------------
 */

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#include "bitonic_sort.h"
#include "err_utils.h"

#define LAMBDA                  8.0
#define SHUFFLE_RATE            0.8
#define TEST_DATA_SIZE          20
//#define DEFAULT_CHUNK_NUM       8
#define DEFAULT_CHUNK_NUM       1

#define _simd_align(x)   ((((intptr_t)x) + ((0x01 << 4) - 1)) & ~((0x01 << 4) - 1))

static int32_t _init_rand;

inline static int _gen_keyseq(int max) __attribute__((always_inline));
inline static void _replace_uint32(void *a, void *b) __attribute__((always_inline));
static double  _gettimeofday_sec(void);

int 
main(int argc, char **argv)
{
        int32_t         i;
        int32_t         res;
        int32_t         err;
        uint32_t        mul;
        uint32_t        scnt;
        uint32_t        cnt;
        uint32_t        r1;
        uint32_t        r2;
        int           *d;
        int           *d_aligned;
        int           *buf;
        int           *buf_aligned;
        double          srat;
        double          t;

        fprintf(stdout, "Usage: %s [the count of 32-bit ints (2^x)] [shuffling ratio]\n", argv[0]);

        mul = TEST_DATA_SIZE;
        srat = SHUFFLE_RATE;

        while ((res = getopt(argc, argv, "fb")) != -1) {
                switch (res) {
                case 'f':
                        _enable_fast_memcpy = 1;
			printf("enable fast memcpy.\n");
                        break;
                
                case 'b':
                        _enable_bitonic_sort = 1;
			printf("enable bitonic sort.\n");
                        break;
                
                default:
                        break;
                
                }
        }

        if (argc - optind >= 1) 
                mul = atoi(argv[optind]);

        if (argc - optind >= 2)
                srat = atof(argv[optind + 1]);

        if (mul < 16 || mul > 30 || srat < 0.1 || srat > 1.0) {
                fprintf(stdout, "Error: out of required range for these parameters (16<=x<=30 / 0.1<=ratio<=1.0)");
                return EXIT_FAILURE;
        }

        cnt = 1 << mul;
        scnt = cnt * srat;

        fprintf(stdout, "Count:%u Shuffle:%u\n", cnt, scnt);

        /* Generate test data */
        d = malloc(sizeof(int) * cnt + 127);
        buf = malloc(sizeof(int) * cnt + 127);

        if (d == NULL || buf == NULL)
                eoutput("Can't allocate memories");

        d_aligned = (int *)_simd_align(d);
        buf_aligned = (int *)_simd_align(buf);

	printf("%d\n", RAND_MAX);
        for (i = 0, d[0] = 0.0; i < cnt - 1; i++)
                d[i] = rand();

        /* Shuffling */
        for (i = 0; i < scnt; i++) {
                r1 = rand() % cnt;
                r2 = rand() % cnt;

                _replace_uint32(&d[r1], &d[r2]);
        }

        /* Execute bitonic sort */
        t = _gettimeofday_sec();
        bitonic_sort(d_aligned, cnt, buf_aligned, DEFAULT_CHUNK_NUM);
        t = _gettimeofday_sec() - t;

        /* Error chcking */
        err = err_check(d_aligned, cnt);

        if (err < 0)
                eoutput("The returned keys are not completely sorted"); 
        else
                fprintf(stdout, "Sorting Speed: %.10lfmis\n", (double)(cnt / 1000000) / t);

        free(d);
        free(buf);

        return EXIT_SUCCESS;
}

/* --- Intra Functions Below ---*/

int
_gen_keyseq(int max) 
{
        if  (!_init_rand++)
                srand(0);

        //return (int)(max * (-log(1.0 - (int)rand() / UINT32_MAX))) + 1.0;
        return rand() % 100;
}

void 
_replace_uint32(void *a, void *b)
{
        int        *va;
        int        *vb;

        va = (int *)a;
        vb = (int *)b;

        *va ^= *vb;
        *vb ^= *va;
        *va ^= *vb;
}

double 
_gettimeofday_sec()
{
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec + (double)tv.tv_usec*1e-6;
}
