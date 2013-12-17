/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_NEON_MATRIX_INLINE_H
#define HQ_NEON_MATRIX_INLINE_H

#include "HQUtilMathCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NEON_ASM
	HQ_UTIL_MATH_API void HQNeonMatrix4Transpose(const hqfloat32 * matrix , hqfloat32 * result);//the code is contained inside the assembly	
#else
	HQ_FORCE_INLINE void HQNeonMatrix4Transpose(const hqfloat32 * matrix , hqfloat32 * result)
	{
		asm volatile(
					 "vld4.32	{d0 , d2 , d4 , d6}	, [%0 , :128]!						\n\t"
					 "vld4.32	{d1 , d3 , d5 , d7}	, [%0 , :128]						\n\t"
					 
					 
					 "vst1.32	{d0 , d1} , [%1  , :128] !								\n\t"
					 "vst1.32	{d2 , d3} , [%1  , :128] !								\n\t"
					 "vst1.32	{d4 , d5} , [%1  , :128] !								\n\t"
					 "vst1.32	{d6 , d7} , [%1  , :128]								\n\t"
					 :"+r" (matrix) , "+r" (result)
					 :
					 :"q0" , "q1" , "q2" , "q3" ,"memory"
					 );
	}
#endif//#ifdef NEON_ASM
	
#ifdef __cplusplus
}
#endif


#endif

