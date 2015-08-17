/*-------------------------------------------------------------------------
 *
 * bitonic_sort.h - 20110612 v0.1.3
 *	  A header for bitonic_sort.c
 *
 *-------------------------------------------------------------------------
 */

#include <stdint.h>

#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

extern uint32_t _enable_fast_memcpy;
extern uint32_t _enable_bitonic_sort;

extern void bitonic_sort(int *d, uint32_t s, int *buf, uint32_t chunk_num);
extern int32_t err_check(const int *d, int s);

#endif /* BITONIC_SORT_H */
