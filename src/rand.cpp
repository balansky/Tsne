#include <rand.hpp>

namespace simile{

RandomGenerator::RandomGenerator (long seed)
    : mt((unsigned int)seed) {}

int RandomGenerator::rand_int ()
{
    return mt() & 0x7fffffff;
}

long RandomGenerator::rand_long ()
{
    return long(rand_int()) | long(rand_int()) << 31;
}

int RandomGenerator::rand_int (int max)
{
    return mt() % max;
}

float RandomGenerator::rand_float ()
{
    return mt() / float(mt.max());
}

double RandomGenerator::rand_double ()
{
    return mt() / double(mt.max());
}

/***********************************************************************
 * Random functions in this C file only exist because Torch
 *  counterparts are slow and not multi-threaded.  Typical use is for
 *  more than 1-100 billion values. */


/* Generate a set of random floating point values such that x[i] in [0,1]
   multi-threading. For this reason, we rely on re-entreant functions.  */
void float_rand (float * x, size_t n, long seed)
{
    // only try to parallelize on large enough arrays
    // const size_t nblock = n < 1024 ? 1 : 1024;
    const size_t nblock = 1;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++)
            x[i] = rng.rand_float ();
    }
}


void float_randn (float * x, size_t n, long seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {
        RandomGenerator rng (a0 + j * b0);

        double a = 0, b = 0, s = 0;
        int state = 0;  /* generate two number per "do-while" loop */

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++) {
            /* Marsaglia's method (see Knuth) */
            if (state == 0) {
                do {
                    a = 2.0 * rng.rand_double () - 1;
                    b = 2.0 * rng.rand_double () - 1;
                    s = a * a + b * b;
                } while (s >= 1.0);
                x[i] = a * sqrt(-2.0 * log(s) / s);
            }
            else
                x[i] = b * sqrt(-2.0 * log(s) / s);
            state = 1 - state;
        }
    }
}


/* Integer versions */
void long_rand (long * x, size_t n, long seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;
        for (size_t i = istart; i < iend; i++)
            x[i] = rng.rand_long ();
    }
}



void rand_perm (int *perm, size_t n, long seed)
{
    for (size_t i = 0; i < n; i++) perm[i] = i;

    RandomGenerator rng (seed);

    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = i + rng.rand_int (n - i);
        std::swap(perm[i], perm[i2]);
    }
}

void byte_rand (uint8_t * x, size_t n, long seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        size_t i;
        for (i = istart; i < iend; i++)
            x[i] = rng.rand_long ();
    }
}

}