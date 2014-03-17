/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_NEON_VECTOR_H
#define HQ_NEON_VECTOR_H

#include "../../HQUtilMathCommon.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef HQ_NEON_ASM
	float HQNeonVector4Dot(const float* v1 , const float* v2);
	
	float HQNeonVector4Length(const float* v);
		
	void HQNeonVector4Cross(const float* v1 , const float* v2 , float *cross);
	
	void HQNeonVector4Normalize(const float* v , float *normalizedVec);

#else//#ifdef HQ_NEON_ASM
	
//vector in q0
#define HQ_VECTOR_NORMALIZE_ASM_BLOCK\
	"vmul.f32		q1 , q0 , q0			\n\t" /*x^2  y^2 z^2 _*/\
	"vpadd.f32		d2 , d2 , d2			\n\t" /*x^2 + y^2		x^2 + y^2*/\
	"vmov.f32		s5 , s6					\n\t" /*d2 = x^2 + y^2			z^2*/\
	"vpadd.f32		d2 , d2 ,d2				\n\t" /*d2 = x^2 + y^2 + z^2        x^2 + y^2 + z^2*/\
\	
	"vrsqrte.f32	d3 , d2					\n\t"/*d3 ~= 1 / sqrt(x^2 + y^2 + z^2)*/\
	/*Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .*/\
	/*here d2 = x ; d3 = y(n)*/\
	"vmul.f32		d4 , d3 , d3			\n\t"/*y(n) ^ 2*/\
	"vrsqrts.f32	d4 , d2 , d4			\n\t"/*1/2  * (3 - x*y(n)^2))*/\
	"vmul.f32		d3 , d3 , d4			\n\t"/*1/2 *(y(n)*(3-x*y(n)^2))*/\
\
	"vmul.f32		q0 , q0 , d3[0]			\n\t"/*(x  , y  ,  z , w)/ length*/
/*--------------------------------*/	
	
	
	HQ_FORCE_INLINE float HQNeonVector4Dot(const float* v1 , const float* v2)
	{
		register float f;
		
		asm volatile(
					  "vld1.32   {d0 , d1}, [%1 , :128]\n\t"// v1.x v1.y v1.z v1.w
					  "vld1.32   {d2 , d3},[%2 , :128]\n\t"// v2.x v2.y v2.z v2.w
					 
					  "vmul.f32 d0 , d0  , d2\n\t"// v1.x * v2.x		v1.y * v2.y	
					  "vadd.f32 s0 , s0 , s1\n\t"//(v1.x * v2.x + v1.y * v2.y)
					  "vmla.f32 s0 , s2 , s6\n\t"//(v1.x * v2.x + v1.y * v2.y) + (v1.z * v2.z)
					  "vmov  %0   , s0\n\t"
					  :"=r" (f)
					  :"r" (v1) , "r" (v2)  
					  :"q0" , "q1"
		);
			
		return f;
	}
	
	HQ_FORCE_INLINE float HQNeonVector4Length(const float* v)
	{
		register float f;
		
		asm volatile(
					 "vld1.32   {d0 , d1} , [%1 , :128]			\n\t"// v1.x v1.y v1.z v1.w
					 //"flds		s2   , [%1 , #8]	\n\t"// v1.z
					 
					 "vmul.f32		d0 , d0  , d0		\n\t"// v1.x^2		v1.y^2	
					 "vmul.f32		d1 , d1  , d1		\n\t"// v1.z^2		v1.w^2
					 "vpadd.f32		d0 , d0 , d0		\n\t"//(v1.x^2 + v1.y^2)
					 "vadd.f32		d0 ,  d0 ,  d1		\n\t"//(v1.x^2 + v1.y^2) + (v1.z^2)
					 "vsqrt.f32		s0 , s0				\n\t"//length
					 "vmov  %0   , s0					\n\t"
					 :"=r" (f) 
					 :"r" (v)  
					 :"d0" , "d1"
					 );
		
		return f;
	}
	
	/*
	float HQNeonVector4DotNonInline(const float* v1 , const float* v2)  __attribute__ ((noinline));
	
	float HQNeonVector4DotNonInline(const float* v1 , const float* v2) 
	{
		return HQNeonVector4Dot(v1 , v2);
	}
	
	float HQNeonVector4LengthNonInline(const float* v)  __attribute__ ((noinline));
	
	float HQNeonVector4LengthNonInline(const float* v) 
	{
		return HQNeonVector4Length(v);
	}*/
		
	HQ_FORCE_INLINE void HQNeonVector4Cross(const float* v1 , const float* v2 , float *cross)
	{
		asm volatile(
					 "vmov.i64	d5 , 0x0000000000000000		\n\t"//d5 = 0 0
					 "vld1.32	{d1[1]}  , [%0 , :32]			\n\t"//x1
					 "add		%0 , %0 , #4		\n\t"
					 "vld1.32	{d0}  , [%0]	\n\t"//y1 z1
					 "vmov.f32  s2 , s1				\n\t"//q0 = y1 z1 z1 x1
					 
					 "vld1.32	{d2[1]}  , [%1, :32]			\n\t"//x2
					 "add		%1 , %1 , #4		\n\t"
					 "vld1.32	{d3}  , [%1]	\n\t"//y2 z2
					 "vmov.f32  s4 , s7				\n\t"//q1 = z2 x2 y2 z2
					 
					 "vmul.f32	d4   , d0  , d2			\n\t"//d4 = y1z2  z1x2
					 "vmls.f32  d4   , d1  , d3			\n\t"//d4 = y1z2 - z1y2			z1x2 - x1z1
					 
					 "vmul.f32  s10  , s3  , s6			\n\t"//x1y2
					 "vmls.f32  s10  , s0  , s5			\n\t"//x1y2  - y1x2
					 
					 "vst1.32  {d4 , d5}   , [%2 , :128]			\n\t"
					 
					 :"+r" (v1) , "+r" (v2) 
					 : "r" (cross)   
					 :"q0"  , "q1" , "d4" , "s10" , "memory"
					 );
		
	}
#ifdef HQ_ANDROID_PLATFORM
#include <android/log.h>
#endif
	HQ_FORCE_INLINE void HQNeonVector4Normalize(const float* v , float *normalizedVec)
	{
#ifdef HQ_ANDROID_PLATFORM
		// __android_log_print(ANDROID_LOG_DEBUG, "HQNeonVector4Normalize", "HQNeonVector4Normalize(v=%p, normalizedVec=%p)", v, normalizedVec);
#endif
		asm volatile(
					 "vmov.i64		d1  , #0x0000000000000000				\n\t"//d1 = 0	0
					 "vld1.32		{d0} , [%0 , :64]			\n\t" //load x y
					 "add			%0, %0, #8					\n\t"
					 "vld1.32		{d1[0]}	,  [%0 , :32]			\n\t"//load z = > q0 = x y z 0
					 
					 HQ_VECTOR_NORMALIZE_ASM_BLOCK
					 
					 "vst1.32		{d0 , d1} , [%1 , :128]		\n\t"//store x y z 0
					 
					 : "+r"  (v) 
					 :  "r" (normalizedVec) 
					 :"q0"  , "q1" , "d4" , "memory"
					 );
		
#ifdef HQ_ANDROID_PLATFORM
		// __android_log_print(ANDROID_LOG_DEBUG, "HQNeonVector4Normalize", "==>HQNeonVector4Normalize(v=%p, normalizedVec=%p) returns", v, normalizedVec);
#endif
	}

#endif//#ifdef HQ_NEON_ASM
	
	
#ifdef __cplusplus
}
#endif


#endif
