/***
 * @copyright Copyright (C) 2024 Kalray SA. All rights reserved.
 * This code is Kalray proprietary and confidential.
 * Any use of the code for whatever purpose is subject
 * to specific written permission of Kalray SA.
 ***/
#ifndef CUSTOM_KERNELS_LIST_H
#define CUSTOM_KERNELS_LIST_H

// For a custom kernel to be available in KaNN, it should be added to this list
// below.
// Kernel name should be of the form mppa_kann_clus_kernel_<suffix>, where only
// the suffix is added to the list.
#define MPPA_CUSTOM_FOREACH_KERNELS(F)                                         \
    F(silu_x8_tf16_tf16)                                                       \

#define CUSTOM_PLUS_ONE(name)  +1
#define KANN_NB_CUSTOM_KERNELS (0 MPPA_CUSTOM_FOREACH_KERNELS(CUSTOM_PLUS_ONE))

#define FNCT_EXTRA(NAME) void mppa_kann_clus_kernel_##NAME();
MPPA_CUSTOM_FOREACH_KERNELS(FNCT_EXTRA)
#undef FNCT_EXTRA

typedef void (*mppa_kann_clus_custom_kernel_fnct_pointer_t)(void *, void *);

typedef struct {
    mppa_kann_clus_custom_kernel_fnct_pointer_t fnct;
    const char *name;
} custom_kernel_fnct_t;

#endif /* CUSTOM_KERNELS_LIST_H */
