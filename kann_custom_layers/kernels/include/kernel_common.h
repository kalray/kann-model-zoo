/***
 * @copyright Copyright (C) 2024 Kalray SA. All rights reserved.
 * This code is Kalray proprietary and confidential.
 * Any use of the code for whatever purpose is subject
 * to specific written permission of Kalray SA.
 ***/

#ifndef COMMON_H
#define COMMON_H

#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"

#define ABS(a) (a<0?-a:a)
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)<(b)?(b):(a))


// FAST SIGMOID IMPLEMENTATION FUNCTION =====================================//
static inline float16_t fast_sigmoid(float16_t value)
{
    // Implementation of y = sigmoid(x)
    //   where sigmoid(x) => 1 / (1 + e(x))   for x <= 0
    //                       1 / (1 + 1/e(x)) for x > 0
    //   and e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4

    float x = ABS(value);
    float x2 = x * x;
    float x4 = x2 * x2;
    float e = 1.0f + x + x2 * 0.555f + x4 * 0.143f;  // 0.2% error
    return 1.0f / (1.0f + (value > 0 ? 1.0f / e : e));
}

// CUSTOM LOGF, EXPF, TANHF             =====================================//

/* method and constants adapted for float16 (aka _Float16):
 in references:
 http://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c
 https://hackage.haskell.org/package/approximate-0.2.2.1/src/cbits/fast.c
 https://nic.schraudolph.org/pubs/Schraudolph99.pdf
 FLOAT16 : b |  |||||  |||||||||||
            15 14  10  9         0
 sign : 1 bit
 exp  : 5 bits
 precision: 10 bits
 coef = 1 << 10 / log(2) = 1477,
 bias << 10 = (01111)b << 10 = 15360
*/

// EXPF FAST --
inline float16_t _expf(float a) {

    /* Function for double precision */
    // union { double d; long long x; } u;
    // u.x = (long long)(6497320848556798LL * a + 0x3fef127e83d16f12LL);
    // return u.d;

    /* Function for float precision */
    union { float d; int x; } u;
    u.x = (int) (12102203 * a + 1065353217);
    return u.d;

    /* Function for float16 precision */
    // union { float16_t d; short x; } u;
    // u.x = (short)(1478 * a + 15360);
    // return u.d;
}

// LOG FAST --
inline float16_t _logf(float a) {

    /* Function for double precision */
    // union { double d; long long x; } u = { a };
    // return (u.x - 4606921278410026770) * 1.539095918623324e-16;

    /* Function for float precision */
    union { float d; int x; } u = { a };
    return (u.x - 1064866805) * 8.262958405176314e-8f;

    /* Function for float16 precision */
    // union { float16_t d; short x; } u = { a };
    // return (u.x - 15360) * 0.0006765899864682003;
}

// FAST TANH IMPLEMENTATION FUNCTION =======================================//
inline float16_t fast_tanhf16(float16_t a) {
    return (2 / (1 + _expf(-2 * a)) - 1);
}

// TANHF --
inline float16_t _tanhf(float16_t x) {
    float16_t num = _expf(x) - _expf(-x);
    float16_t den = _expf(x) + _expf(-x);
    return (num * __builtin_kvx_frecw(den, ".s"));
}

#endif /*COMMON_H*/