/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_NEON_QUAT_INLINE_H
#define HQ_NEON_QUAT_INLINE_H

#include "HQUtilMathCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NEON_ASM

	//the code is contained inside the assmebly

	HQ_UTIL_MATH_API void HQNeonQuatAdd(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result);

	HQ_UTIL_MATH_API void HQNeonQuatSub(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result);

	HQ_UTIL_MATH_API void HQNeonQuatMultiplyScalar(const hqfloat32 *quat1, hqfloat32 f, hqfloat32 *result);

#else//#ifdef NEON_ASM

	HQ_FORCE_INLINE void HQNeonQuatAdd(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	{
		asm volatile(
				 "vld1.32	{d0 , d1} , [%0, : 128]			\n\t"
				 "vld1.32	{d2 , d3} , [%1, : 128]			\n\t"
				 "vadd.f32	q2 , q0 , q1				\n\t"
				 "vst1.32	{d4 , d5} , [%2, : 128]			\n\t"
				 :
				 :"r"(quat1) , "r" (quat2), "r" (result)
				 :"q0" , "q1" , "q2" , "memory"
		);
	}

	HQ_FORCE_INLINE void HQNeonQuatSub(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	{
		asm volatile(
				 "vld1.32	{d0 , d1} , [%0, : 128]			\n\t"
				 "vld1.32	{d2 , d3} , [%1, : 128]			\n\t"
				 "vsub.f32	q2 , q0 , q1				\n\t"
				 "vst1.32	{d4 , d5} , [%2, : 128]			\n\t"
				 :
				 :"r"(quat1) , "r" (quat2), "r" (result)
				 :"q0" , "q1" , "q2" , "memory"
		);
	}

	HQ_FORCE_INLINE void HQNeonQuatMultiplyScalar(const hqfloat32 *quat1, hqfloat32 f, hqfloat32 *result)
	{
		asm volatile(
				 "vld1.32	{d0 , d1} , [%0, : 128]			\n\t"
				 "vmov		s4 ,		%1				\n\t"
				 "vmul.f32	q2 , q0 , d2[0]					\n\t"
				 "vst1.32	{d4 , d5} , [%2, : 128]			\n\t"
				 :
				 :"r"(quat1) , "r" (f), "r" (result)
				 :"q0" , "q1" , "q2" , "memory"
				 );
	}

#endif//#ifdef NEON_ASM

#ifdef __cplusplus
}
#endif

#endif
