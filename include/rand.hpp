#ifndef SIMILE_RAND_H
#define SIMILE_RAND_H

#include <random>
#include <stdint.h>


/**************************************************
 * Random data generation functions
 **************************************************/

/// random generator that can be used in multithreaded contexts

namespace simile{


struct RandomGenerator {

    std::mt19937 mt;

    /// random positive integer
    int rand_int ();

    /// random long
    long rand_long ();

    /// generate random integer between 0 and max-1
    int rand_int (int max);

    /// between 0 and 1
    float rand_float ();

    double rand_double ();

    explicit RandomGenerator (long seed = 1234);
};

/* Generate an array of uniform random floats / multi-threaded implementation */
void float_rand (float * x, size_t n, long seed);
void float_randn (float * x, size_t n, long seed);
void long_rand (long * x, size_t n, long seed);
void byte_rand (uint8_t * x, size_t n, long seed);

/* random permutation */
void rand_perm (int * perm, size_t n, long seed);

}
#endif