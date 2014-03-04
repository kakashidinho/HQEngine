/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_NEON_QUATERNION_H
#define HQ_NEON_QUATERNION_H

#include "../../HQUtilMathCommon.h"
#include "HQNeonVector.h"


#ifdef __cplusplus
extern "C" {
#endif
#ifdef NEON_ASM
	float HQNeonQuatMagnitude(float * quat);
	
	float HQNeonQuatSumSquares(float * quat);
	
	float HQNeonQuatDot(const float * quat1 ,const float *quat2);
	
	void HQNeonQuatNormalize(const float* q , float *normalizedQuat);
	
	void HQNeonQuatInverse(const float* q , float *result);
	
	void HQNeonQuatMultiply(const float * quat1 ,const  float *quat2 , float* result);
	
	void HQNeonQuatUnitToRotAxis(const float* q , float *axisVector);
	
	void HQNeonQuatUnitToMatrix3x4c(const float* q , float *matrix);
	
	
	void HQNeonQuatUnitToMatrix4r(const float* q , float *matrix);

	
	void HQNeonQuatFromMatrix3x4c(const float *matrix , float * quaternion);

	void HQNeonQuatFromMatrix4r(const float *matrix , float * quaternion);
#else//#ifdef NEON_ASM
	HQ_FORCE_INLINE float HQNeonQuatMagnitude(float * quat)
	{
		register float re;
		
		asm volatile(
					 "vld1.32		{d0 , d1}  , [%1 , :128]				\n\t"
					 "vmul.f32		q0 , q0 , q0					\n\t"//x^2 y^2 z^2 w^2
					 "vpadd.f32		d0 , d0 , d1					\n\t"//x^2 + y^2			z^2 + w^2
					 "vadd.f32		s0 , s0 , s1					\n\t"//x^2 + y^2 + z^2 + w^2
					 "vsqrt.f32		s0 , s0							\n\t"
					 "vmov			%0 , s0							\n\t"
					 :"=r" (re)
					 :"r" (quat)
					 :"q0"
		);
		
		return re;
	}
	
	HQ_FORCE_INLINE float HQNeonQuatSumSquares(float * quat)
	{
		register float re;
		
		asm volatile(
					 "vld1.32		{d0 , d1}  , [%1 , :128]				\n\t"
					 "vmul.f32		q0 , q0 , q0					\n\t"//x^2 y^2 z^2 w^2
					 "vpadd.f32		d0 , d0 , d1					\n\t"//x^2 + y^2			z^2 + w^2
					 "vadd.f32		s0 , s0 , s1					\n\t"//x^2 + y^2 + z^2 + w^2
					 "vmov			%0 , s0							\n\t"
					 :"=r" (re)
					 :"r" (quat)
					 :"q0"
					 );
		
		return re;
	}
	
	HQ_FORCE_INLINE float HQNeonQuatDot(const float * quat1 ,const float *quat2)
	{
		register float re;
		
		asm volatile(
					 "vld1.32		{d0 , d1}  , [%1 , :128]				\n\t"
					 "vld1.32		{d2 , d3}  , [%2 , :128]				\n\t"
					 "vmul.f32		q0 , q0 , q1					\n\t"//x1.x2 y1.y2 z1.z2 w1.w2
					 "vpadd.f32		d0 , d0 , d1					\n\t"//x1x2 + y1y2			z1z2 + w1w2
					 "vadd.f32		s0 , s0 , s1					\n\t"//x1x2 + y1y2 + z1z2 + w1w2
					 "vmov			%0 , s0							\n\t"
					 :"=r" (re)
					 :"r" (quat1) , "r" (quat2)
					 :"q0" , "q1"
					 );
		
		return re;
	}
	
	HQ_FORCE_INLINE void HQNeonQuatNormalize(const float* q , float *normalizedQuat)
	{
		asm volatile(
					 "vld1.32		{d0 , d1} , [%0 , :128]			\n\t" //x y z w
					 
					 "vmul.f32		q1 , q0 , q0				\n\t" //x^2  y^2 z^2 w^2
					 "vpadd.f32		d2 , d2 , d3				\n\t" //x^2 + y^2		z^2 + w^2
					 "vpadd.f32		d2 , d2 ,d2					\n\t" //d2 = x^2 + y^2 + z^2 + w^2     x^2 + y^2 + z^2 + w^2
					 
					 "vrsqrte.f32	d3 , d2						\n\t"//d3 ~= 1 / sqrt(x^2 + y^2 + z^2 + w^2)
					 //Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .
					 //here d2 = x ; d3 = y(n)
					 "vmul.f32		d4 , d3 , d3				\n\t"//y(n) ^ 2
					 "vrsqrts.f32		d4 , d2 , d4			\n\t"//1/2  * (3 - x*y(n)^2))
					 "vmul.f32		d3 , d3 , d4				\n\t"//1/2 *(y(n)*(3-x*y(n)^2))
					 
					 "vmul.f32		q0 , q0 , d3[0]				\n\t"//(x  , y  ,  z , w)/ length
					 
					 "vst1.32		{d0 , d1} , [%1 , :128]			\n\t"
					 
					 :
					 :"r"  (q) ,"r" (normalizedQuat) 
					 :"q0"  , "q1" , "d4" , "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonQuatInverse(const float* q , float *result)
	{
		asm volatile(
					 "vld1.32		{d0 , d1} , [%0 , :128]		\n\t" //x y z w
					 "vneg.f32		d0 , d0						\n\t"//-x -y
					 "vneg.f32		s2 , s2						\n\t"//-z
					 
					 "vmul.f32		q1 , q0 , q0				\n\t" //x^2  y^2 z^2 w^2
					 "vpadd.f32		d2 , d2 , d3				\n\t" //x^2 + y^2		z^2 + w^2
					 "vpadd.f32		d2 , d2 ,d2					\n\t" //d2 = x^2 + y^2 + z^2 + w^2     x^2 + y^2 + z^2 + w^2
					 
					 "vrecpe.f32	d3 , d2						\n\t"//d3 ~= 1 / (x^2 + y^2 + z^2 + w^2)
					 //Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 =  Y(n) * (2 - x * Y (n)) ; reduce estimate error .
					 //here d2 = x ; d3 = y(n)
					 "vrecps.f32	d4 , d2 , d3				\n\t"//2 - x * Y (n)
					 "vmul.f32		d3 , d3 , d4				\n\t"//Y(n) * (2 - x * Y (n))
					 
					 "vmul.f32		q0 , q0 , d3[0]				\n\t"//(-x  , -y  ,  -z , w) * d3[0]
					 
					 "vst1.32		{d0 , d1} , [%1 , :128]			\n\t"
					 
					 :
					 :"r"  (q) ,"r" (result) 
					 :"q0"  , "q1" , "d4" , "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonQuatMultiply(const float * quat1 ,const  float *quat2 , float* result)
	{
		asm volatile(
					 "vld1.32		{d2 , d3} , [%1  , :128]			\n\t" //q2 = x2 y1 z2 w2
					 "vld1.32		{d0 , d1} , [%0  , :128]			\n\t" //q1 = x1 y1 z1 w1
					 
					 "vmov			d5 , d2						\n\t"
					 "vmov			d4 , d3						\n\t"//q2 = z2 w2 x2 y2
					 "vrev64.32		q3 , q2						\n\t"//q3 = w2 z2 y2 x2
					 
					 
					 "vmul.f32		q5 , q3 , d0[0]				\n\t"//q5 = x1w2    x1z2    x1y2    x1x2
					 "vmul.f32		q4 , q1 , d1[1]				\n\t"//q4 = w1x2	w1y2	w2z2	w1w2
					 "vswp			d6 , d7						\n\t"//q3 = y2 x2 w2 z2
					 "vmul.f32		q3 , q3 , d1[0]				\n\t"//q3 = z1y2	z1x2	z1w2	z1z2
					 "vmla.f32		d8 , d4 , d0[1]				\n\t"//d8 = w1x1 + y1z2     w1x2 + y1w2
					 "vmls.f32		d9 , d5 , d0[1]				\n\t"//d9 = w1z2 - y1x2		w1w2 - y1y2
					 "vneg.f32		s21 , s21					\n\t"//q5 = x1w2    -x1z2    x1y2    x1x2
					 "vneg.f32		s23 , s23					\n\t"//q5 = x1w2    -x1z2    x1y2    -x1x2
					 "vadd.f32		q4 , q4 , q5				\n\t"
					 "vneg.f32		s12 , s12					\n\t"//q3 = -z1y2	z1x2	z1w2	z1z2
					 "vneg.f32		s15 , s15					\n\t"//q3 = -z1y2	z1x2	z1w2	-z1z2
					 "vadd.f32		q4 , q4 , q3				\n\t"
					 
					 "vst1.32		{d8 , d9} , [%2 , :128]			\n\t"
					 :
					 :"r"(quat1) , "r" (quat2) , "r"(result)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory"
					 
					 
		);
	}
	
	HQ_FORCE_INLINE void HQNeonQuatUnitToRotAxis(const float* q , float *axisVector)
	{
		HQNeonVector4Normalize(q, axisVector);
	}
	
	HQ_FORCE_INLINE void HQNeonQuatUnitToMatrix3x4c(const float* q , float *matrix)
	{
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1  , :128]				\n\t" //x y z w
					 "vmov.i64		d7 , #0x0000000000000000		\n\t"//d7 = 0 0
					 "vmov.f32		s13 , #1.0						\n\t"//s13 = 1
					 
					 "vadd.f32		q1 , q0 , q0					\n\t"//q1 = d2{2x 2y} d3{2z 2w}
					 "vrev64.32		d8 , d0							\n\t"//d8 = y x
					 
					 "vmul.f32		d4 , d8 , d2[1]					\n\t"//d4 = 2yy  2xy 
					 "vmul.f32		d9 , d8 , d3[1]					\n\t"//d9 = 2yw  2xw
					 "vmul.f32		d10 , d0 , d2					\n\t"//d10 = 2xx 2yy
					 "vmul.f32		d5 , d0 , d3[0]					\n\t"//d5 = 2xz  2yz ; q2 = d4{2yy  2xy}	d5{2xz  2yz}
					 "vsub.f32		s14 , s13 , s20					\n\t"//s14 = 1 - 2xx
					 "vmul.f32		d8 , d1 , d3[0]					\n\t"//d8 = 2zz  2zw ; q4 = d8{2zz	2zw}	d9{2yw	2xw}
					 
					 "vsub.f32		s14 , s14 , s21					\n\t"//s14 = 1 - 2xx - 2yy ; d7 = 1-2xx-2yy	0.0
					 
					 //first row
					 "vneg.f32		d5 , d5						\n\t"//q2 = 2yy 2xy -2xz -2yz
					 "vneg.f32		d8 , d8						\n\t"//q4 = -2zz -2zw 2yw 2xw
					 "vneg.f32		s9 , s9						\n\t"//q2 = 2yy -2xy -2xz -2yz
					 "vadd.f32		s16 , s16 , s13				\n\t"//q4 = 1-2zz	-2zw	2yw		2xw
					 "vsub.f32		q0 , q4 , q2				\n\t"//q0 = -2zz-2yy+1 -2zw+2xy 2yw+2xz 2xw+2yz //first row (except last element)
					 
					 "vst1.32		{d0} , [%0 , :64]!				\n\t"//11 12
					 "vst1.32		{d1[0]}  , [%0, :32]!					\n\t"//13
					 "vst1.32		{d7[1]} , [%0, :32]!				\n\t"//store 0.0 to 14
					 
					 
					 //second row
					 "vmov.f32			s8 , s20					\n\t"//q2 = 2xx -2xy -2xz -2yz
					 "vneg.f32		d9 , d9						\n\t"//q4 = -2zz+1, -2zw, -2yw, -2xw
					 "vneg.f32		s17 , s17					\n\t"//q4 = -2zz+1, 2zw, -2yw, -2xw
					 "vsub.f32		q4 , q4 , q2				\n\t"//q4 = -2zz-2xx+1 2xy+2zw -2yw+2xz -2xw+2yz
					 "vrev64.32		q1 , q4						\n\t"//q1 = 2xy+2zw -2zz-2xx+1 -2xw+2yz -2yw+2xz//second row (except last element)
					 
					 "vst1.32		{d2} , [%0 , :64]!				\n\t"//21	22
					 "vst1.32		{d3[0]}  , [%0, :32]!					\n\t"//23
					 "vst1.32			{d7[1]} , [%0 , :32]!				\n\t"//store 0.0 to 24
					 
					 //third row
					 "vtrn.32		d3 , d1						\n\t"//d1 =	-2yw+2xz	2xw+2yz ; d7 = 1-2xx-2yy	0.0
					 "vst1.32		{d1 }	,[%0 , :64]!				\n\t"//31 32
					 "vst1.32		{d7}	,[%0 , :64]				\n\t"//33 34
					 
					 :"+r" (matrix)
					 :"r" (q)
					 :"q0" , "q1" , "q2" , "s13" , "d7" , "q4" , "d10" , "memory"
		);
	}
	
	
	HQ_FORCE_INLINE void HQNeonQuatUnitToMatrix4r(const float* q , float *matrix)
	{
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1  , :128]				\n\t" //x y z w
					 "vmov.i64		d7 , #0x0000000000000000		\n\t"//d7 = 0 0
					 "vmov.i64		d9 , #0x0000000000000000		\n\t"//d9 = 0 0
					 "vmov.f32		s19 , #1.0						\n\t"//s19 = 1 ; d9 = 0 1
					 
					 "vadd.f32		q1 , q0 , q0					\n\t"//q1 = d2{2x 2y} d3{2z 2w}
					 "vrev64.32		d10 , d0						\n\t"//d10 = y x
					 
					 "vmul.f32		d4 , d10 , d2[1]				\n\t"//d4 = 2yy  2xy 
					 "vmul.f32		d11 , d10, d3[1]				\n\t"//d11 = 2yw  2xw
					 "vmul.f32		d8 , d0 , d2					\n\t"//d8 = 2xx 2yy
					 "vmul.f32		d5 , d0 , d3[0]					\n\t"//d5 = 2xz  2yz ; q2 = d4{2yy  2xy}	d5{2xz  2yz}
					 "vmul.f32		d10 , d1 , d3[0]				\n\t"//d10 = 2zz  2zw ; q5 = d10{2zz	2zw}	d11{2yw	2xw}
					 
					 "vsub.f32		s14 , s19 , s16					\n\t"//s14 = 1 - 2xx
					 "vsub.f32		s14 , s14 , s17					\n\t"//s14 = 1 - 2xx - 2yy ; d7 = 1-2xx-2yy	0.0
					 
					 //first column
					 "vneg.f32		d5 , d5						\n\t"//q2 = 2yy 2xy -2xz -2yz
					 "vneg.f32		d10 , d10					\n\t"//q5 = -2zz -2zw 2yw 2xw
					 "vneg.f32		s9 , s9						\n\t"//q2 = 2yy -2xy -2xz -2yz
					 "vadd.f32		s20 , s20 , s19				\n\t"//q5 = 1-2zz	-2zw	2yw		2xw
					 "vsub.f32		q1 , q5 , q2				\n\t"//q1 = -2zz-2yy+1 -2zw+2xy 2yw+2xz 2xw+2yz //first column (except last element)
					 
					 //second column
					 "vmov.f32			s8 , s16					\n\t"//q2 = 2xx -2xy -2xz -2yz
					 "vmov.i64		d8 , #0x0000000000000000	\n\t"//q4 = 0 0 0 1
					 "vneg.f32		d11 , d11					\n\t"//q5 = -2zz+1, -2zw, -2yw, -2xw
					 "vneg.f32		s21 , s21					\n\t"//q5 = -2zz+1, 2zw, -2yw, -2xw
					 "vsub.f32		q5 , q5 , q2				\n\t"//q5 = -2zz-2xx+1 2xy+2zw -2yw+2xz -2xw+2yz
					 "vrev64.32		q2 , q5						\n\t"//q2 = 2xy+2zw -2zz-2xx+1 -2xw+2yz -2yw+2xz//second column (except last element)
					 
					 //third column
					 "vmov			d6 , d3						\n\t"//d6 = 2yw+2xz 2xw+2yz
					 "vtrn.32		d5 , d6						\n\t"//q3 : d6 =	-2yw+2xz	2xw+2yz ; d7 = 1-2xx-2yy	0.0
					 
					 "vmov.f32			s7 , s15					\n\t"//q1 =  -2zz-2yy+1 -2zw+2xy 2yw+2xz 0
					 "vmov.f32			s11 , s15					\n\t"//q2 =  2xy+2zw -2zz-2xx+1 -2xw+2yz 0
					 
					 //store transpose of {q1 , q2 , q3 , q4}
					 "vst4.32		{d2 , d4 , d6 , d8}	,[%0 , :128]!			\n\t"
					 "vst4.32		{d3 , d5 , d7 , d9}	,[%0 , :128]			\n\t"
					 
					 :"+r" (matrix)
					 :"r" (q)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory"
					 );
	}

	
	static const hq_uint32 SIMD_DW_mat2quatShuffle0  = (12<<0)|(8<<8)|(4<<16)|(0<<24) ;
	static const hq_uint32 SIMD_DW_mat2quatShuffle1  = (0<<0)|(4<<8)|(8<<16)|(12<<24) ;
	static const hq_uint32 SIMD_DW_mat2quatShuffle2  = (4<<0)|(0<<8)|(12<<16)|(8<<24) ; 
	static const hq_uint32 SIMD_DW_mat2quatShuffle3  = (8<<0)|(12<<8)|(0<<16)|(4<<24) ;
	static const hq_uint32 NegSignMask = 0x80000000;
	
#ifdef WIN32
	struct A16BYTE A16_Bytes
#else
	struct A16_Bytes
#endif
	{
		unsigned char bytes[16];
		unsigned char& operator[] (hq_int32 i)
		{
			return bytes[i];
		}
#if defined LINUX || defined APPLE
	}  A16BYTE ;
#else
};
#endif

	
	HQ_FORCE_INLINE void HQNeonQuatFromMatrix3x4c(const float *matrix , float * quaternion)
	{
		asm volatile(
					 "vmov.i64		d6 , #0x0000000000000000							\n\t"//d6 = 0 0
					 "vld1.f32		{d0 , d1} , [%0  , :128] !									\n\t"//11 12 13 14
					 "vld1.f32		{d2 , d3} , [%0  , :128] !									\n\t"//21 22 23 24
					 "vld1.f32		{d4 , d5} , [%0  , :128]									\n\t"//31 32 33 34
					 "vrev64.32		d2 , d2												\n\t"//d2 = 22 21
					 
					 "vcge.f32		d7 , d0 , d2										\n\t"//d7[0] = 11 >= 22 ?
					 "vcge.f32		d8 , d0 , d5										\n\t"//d8[0] = 11 >= 33 ?
					 "vadd.f32		s12 , s0 , s4										\n\t"//s12 = 11 + 22
					 "vadd.f32		s12 , s12 , s10										\n\t"//s12 = 11 + 22 + 33
					 "vcge.f32		d6 , d6 , #0										\n\t"//d6[0] = (11 + 22 + 33) >= 0?
					 "vand			d8 , d7 , d8										\n\t"//d8[0] = 11 is max?
					 
					 "vcge.f32		d9 , d2 , d5										\n\t"//d9[0] = 22 >= 33 ?
					 "vbic			d7 , d8 , d6										\n\t"//d7[0] = (max == 11 && !((11 + 22 + 33) >= 0))
					 "vorr			d10 , d8 , d6										\n\t"//d10[0] = ((11 + 22 + 33) >= 0 || max = 11)?
					 
					 "vbic			d8 , d9 , d10										\n\t"//d8[0] =!((11 + 22 + 33) >= 0 || max = 11) && 22 >= 33 ?
					 "vorr			d10 , d9 , d10										\n\t"//d10[0] =((11 + 22 + 33) >= 0 || max = 11) || 22 >= 33
					 "vmvn			d9 , d10											\n\t"//d9[0] = !d10[0]
					 
					 "vmov			s22 , %[SIMD_DW_mat2quatShuffle0]					\n\t"
					 "vmov			s24	, %[SIMD_DW_mat2quatShuffle1]					\n\t"
					 "vmov			s26	, %[SIMD_DW_mat2quatShuffle2]					\n\t"
					 "vmov			s28	, %[SIMD_DW_mat2quatShuffle3]					\n\t"
					 
					 "vand			d6 , d6 , d11										\n\t"
					 "vand			d10 , d7 , d12										\n\t"
					 "vorr			d6 , d6 , d10										\n\t"
					 "vand			d10 , d8 , d13										\n\t"
					 "vorr			d6 , d6 , d10										\n\t"
					 "vand			d10 , d9 , d14										\n\t"
					 "vorr			d10 , d6 , d10										\n\t"
					 
					 "vmov			r4 , s20											\n\t"
					 
					 "vmov			s22 , %[NegSignMask]								\n\t"
					 "vorr			d6 , d8 , d9										\n\t"//(max = 22 || max = 33) && sum < 0
					 "vorr			d8 , d7 , d8										\n\t"//(max = 11 || max = 22) && sum < 0
					 "vorr			d7 , d7 , d9										\n\t"//(max = 11 || max = 33) && sum < 0
					 
					 "vand			d6 , d6 , d11										\n\t"//d6[0] = s0
					 "vand			d7 , d7 , d11										\n\t"//d7[0] = s1
					 "vand			d8 , d8 , d11										\n\t"//d8[0] = s2
					 "vdup.32		d6 , d6[0]											\n\t"
					 "vdup.32		d7 , d7[0]											\n\t"
					 "vdup.32		d8 , d8[0]											\n\t"
					 
					 "veor			d9 , d6 , d0										\n\t"//s0 ^ 11 (^ is xor)
					 "veor			d10 , d7 , d2										\n\t"//s1 ^ 22 (^ is xor)
					 "veor			d11 , d8 , d5										\n\t"//s2 ^ 33 (^ is xor)
					 "vmov.f32			s19 , s20											\n\t"//d9[1] = d10[0]
					 
					 "vmov.f32		s23 , #1.0											\n\t"//d11[1] = 1
					 "vpadd.f32		d9 , d9 , d11										\n\t"//d9 = s0 ^ 11 + s1 ^ 22			s2 ^ 33 + 1.0f
					 "vpadd.f32		d9 , d9 , d9										\n\t"//d9 = s0 ^ 11 + s1 ^ 22 + s2 ^ 33 + 1.0f = t
					 
					 "vmov.f32		s24 , #0.5											\n\t"
					 
					 "vrsqrte.f32	d10 , d9											\n\t"/*d10 ~= 1 / sqrt(t)*/
					 /*Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .*/
					 /*here d9 = x ; d10 = y(n)*/
					 "vmul.f32		d11 , d10 , d10										\n\t"/*y(n) ^ 2*/
					 "vrsqrts.f32	d11 , d9 , d11										\n\t"/*1/2  * (3 - x*y(n)^2))*/
					 "vmul.f32		d10 , d10 , d11										\n\t"/*1/2 *(y(n)*(3-x*y(n)^2))*/
					 
					 "vmul.f32		d10 , d10 , d12[0]									\n\t"//d10 = 0.5 / sqrt(t) = s
					 "vmul.f32		d11 , d9 , d10										\n\t"//d11 = s * t
					 
					 //q[k0]
					 "and			r5 , r4 , #0xff										\n\t"//r5 = k0 * 4 = r4 & 0xff
					 "add			r5 , r5 , %[quatPointer]							\n\t"//r5 = &q[k0]
					 "vst1.32		{d11[0]}	, [r5, :32]											\n\t"//q[k0] = s * t
					 
					 
					 //q[k1]
					 "veor			d9 , d0 , d8										\n\t"//d9[1] = s2 ^ 12
					 "vsub.f32		d9 , d2 , d9										\n\t"//d9[1] = 21 - s2 ^ 12
					 "vmul.f32		d9 , d9 , d10										\n\t"//d9[1] = (21 - s2 ^ 12) * s
					 
					 "and			r5 , r4 , #0xff00									\n\t"//r5 = r4 & 0xff00
					 "add			r5 , %[quatPointer]	, r5, LSR #8					\n\t"//r5 = &q[k1] = q + (r5 >> 8)
					 "vst1.32		{d9[1]}	, [r5,  :32]											\n\t"//q[k1] = (21 - s2 ^ 12) * s
					 
					 //q[k2]
					 "veor			d9 , d4 , d7										\n\t"//d9[0] = s1 ^ 31
					 "vsub.f32		d9 , d1 , d9										\n\t"//d9[0] = 13 - s1 ^ 31
					 "vmul.f32		d9 , d9 , d10										\n\t"//d9[0] = (13 - s1 ^ 31) * s
					 
					 "and			r5 , r4 , #0xff0000									\n\t"//r5 = (r4) & 0xff0000
					 "add			r5 , %[quatPointer]	, r5 ,LSR #16					\n\t"//r5 = &q[k2] = q + (r5 >> 16)
					 "vst1.32		{d9[0]}	, [r5 , :32]											\n\t"//q[k2] = (13 - s1 ^ 31) * s
					 
					 //q[k3]
					 "veor			d9 , d3 , d6										\n\t"//d9[0] = s0 ^ 23
					 "vsub.f32		s18 , s9 , s18										\n\t"//d9[0] = 32 - s0 ^ 23
					 "vmul.f32		d9 , d9 , d10										\n\t"//d9[0] = (32 - s0 ^ 23) * s
					 
					 "add			r5 , %[quatPointer], r4 , LSR #24					\n\t"//r5 = &q[k3] = q + (r4 >> 24)
					 "vst1.32		{d9[0]}	, [r5, :32]											\n\t"//q[k3] = (32 - s0 ^ 23) * s
					 
					 :	"+r"(matrix)
					 :	[SIMD_DW_mat2quatShuffle0]"r" (SIMD_DW_mat2quatShuffle0),
						[SIMD_DW_mat2quatShuffle1]"r" (SIMD_DW_mat2quatShuffle1),
						[SIMD_DW_mat2quatShuffle2]"r" (SIMD_DW_mat2quatShuffle2),
						[SIMD_DW_mat2quatShuffle3]"r" (SIMD_DW_mat2quatShuffle3),
						[NegSignMask]"r" (NegSignMask),
						[quatPointer]"r" (quaternion)
					 :"r4" , "r5" ,"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7", "memory"
		);
	}


	HQ_FORCE_INLINE void HQNeonQuatFromMatrix4r(const float *matrix , float * quaternion)
	{
		asm volatile(
					 "vmov.i64		d6 , #0x0000000000000000							\n\t"//d6 = 0 0
					 "vmov.i64		d11 , #0xffffffffffffffff							\n\t"//
					 "vld1.f32		{d0 , d1} , [%0 , :128] !							\n\t"//11 12 13 14
					 "vld1.f32		{d2 , d3} , [%0 , :128] !							\n\t"//21 22 23 24
					 "vld1.f32		{d4 , d5} , [%0 , :128]								\n\t"//31 32 33 34
					 "vrev64.32		d2 , d2												\n\t"//d2 = 22 21
					 
					 "vcge.f32		d7 , d0 , d2										\n\t"//d7[0] = 11 >= 22 ?
					 "vcge.f32		d8 , d0 , d5										\n\t"//d8[0] = 11 >= 33 ?
					 "vadd.f32		s12 , s0 , s4										\n\t"//s12 = 11 + 22
					 "vadd.f32		s12 , s12 , s10										\n\t"//s12 = 11 + 22 + 33
					 "vcge.f32		d6 , d6 , #0										\n\t"//d6[0] = (11 + 22 + 33) >= 0?
					 "vand			d8 , d7 , d8										\n\t"//d8[0] = 11 is max?
					 
					 "vcge.f32		d9 , d2 , d5										\n\t"//d9[0] = 22 >= 33 ?
					 "vbic			d7 , d8 , d6										\n\t"//d7[0] = (max == 11 && !((11 + 22 + 33) >= 0))
					 "vorr			d10 , d8 , d6										\n\t"//d10[0] = ((11 + 22 + 33) >= 0 || max = 11)?
					 
					 "vbic			d8 , d9 , d10										\n\t"//d8[0] =!((11 + 22 + 33) >= 0 || max = 11) && 22 >= 33 ?
					 "vorr			d9 , d9 , d10										\n\t"//d9[0] =((11 + 22 + 33) >= 0 || max = 11) || 22 >= 33
					 "veor			d9 , d9 , d11										\n\t"//d10[0] = !d9[0]
					 
					 "vmov			s22 , %[SIMD_DW_mat2quatShuffle0]					\n\t"
					 "vmov			s24	, %[SIMD_DW_mat2quatShuffle1]					\n\t"
					 "vmov			s26	, %[SIMD_DW_mat2quatShuffle2]					\n\t"
					 "vmov			s28	, %[SIMD_DW_mat2quatShuffle3]					\n\t"
					 
					 "vand			d6 , d6 , d11										\n\t"
					 "vand			d10 , d7 , d12										\n\t"
					 "vorr			d6 , d6 , d10										\n\t"
					 "vand			d10 , d8 , d13										\n\t"
					 "vorr			d6 , d6 , d10										\n\t"
					 "vand			d10 , d9 , d14										\n\t"
					 "vorr			d10 , d6 , d10										\n\t"
					 
					 "vmov			r4 , s20											\n\t"
					 
					 "vmov			s22 , %[NegSignMask]								\n\t"
					 "vorr			d6 , d8 , d9										\n\t"//(max = 22 || max = 33) && sum < 0
					 "vorr			d8 , d7 , d8										\n\t"//(max = 11 || max = 22) && sum < 0
					 "vorr			d7 , d7 , d9										\n\t"//(max = 11 || max = 33) && sum < 0
					 
					 "vand			d6 , d6 , d11										\n\t"//d6[0] = s0
					 "vand			d7 , d7 , d11										\n\t"//d7[0] = s1
					 "vand			d8 , d8 , d11										\n\t"//d8[0] = s2
					 "vdup.32		d6 , d6[0]											\n\t"
					 "vdup.32		d7 , d7[0]											\n\t"
					 "vdup.32		d8 , d8[0]											\n\t"
					 
					 "veor			d9 , d6 , d0										\n\t"//s0 ^ 11 (^ is xor)
					 "veor			d10 , d7 , d2										\n\t"//s1 ^ 22 (^ is xor)
					 "veor			d11 , d8 , d5										\n\t"//s2 ^ 33 (^ is xor)
					 "vmov.f32			s19 , s20											\n\t"//d9[1] = d10[0]
					 
					 "vmov.f32		s23 , #1.0											\n\t"//d11[1] = 1
					 "vpadd.f32		d9 , d9 , d11										\n\t"//d9 = s0 ^ 11 + s1 ^ 22			s2 ^ 33 + 1.0f
					 "vpadd.f32		d9 , d9 , d9										\n\t"//d9 = s0 ^ 11 + s1 ^ 22 + s2 ^ 33 + 1.0f = t
					 
					 "vmov.f32		s24 , #0.5											\n\t"
					 
					 "vrsqrte.f32	d10 , d9											\n\t"/*d10 ~= 1 / sqrt(t)*/
					 /*Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .*/
					 /*here d9 = x ; d10 = y(n)*/
					 "vmul.f32		d11 , d10 , d10										\n\t"/*y(n) ^ 2*/
					 "vrsqrts.f32	d11 , d9 , d11										\n\t"/*1/2  * (3 - x*y(n)^2))*/
					 "vmul.f32		d10 , d10 , d11										\n\t"/*1/2 *(y(n)*(3-x*y(n)^2))*/
					 
					 "vmul.f32		d10 , d10 , d12[0]									\n\t"//d10 = 0.5 / sqrt(t) = s
					 "vmul.f32		d11 , d9 , d10										\n\t"//d11 = s * t
					 
					 //q[k0]
					 "and			r5 , r4 , #0xff										\n\t"//r5 = k0 * 4
					 "add			r5 , r5 , %[quatPointer]							\n\t"//r5 = &q[k0]
					 "vst1.32		{d11[0]}	, [r5, :32]											\n\t"//q[k0] = s * t
					 
					 
					 //q[k1]
					 "veor			d9 , d2 , d8										\n\t"//d9[1] = s2 ^ 21
					 "vsub.f32		d9 , d0 , d9										\n\t"//d9[1] = 12 - s2 ^ 21
					 "vmul.f32		d9 , d9 , d10										\n\t"//d9[1] = (12 - s2 ^ 21) * s
					 
					 "and			r5 , r4 , #0xff00									\n\t"//r5 = r4 & 0xff00
					 "add			r5 , %[quatPointer]	, r5, LSR #8					\n\t"//r5 = &q[k1] = q + (r5 >> 8)
					 "vst1.32		{d9[1]}	, [r5, :32]											\n\t"//q[k1] = (12 - s2 ^ 21) * s
					 
					 //q[k2]
					 "veor			d9 , d1 , d7										\n\t"//d9[0] = s1 ^ 13
					 "vsub.f32		d9 , d4 , d9										\n\t"//d9[0] = 31 - s1 ^ 13
					 "vmul.f32		d9 , d9 , d10										\n\t"//d9[0] = (31 - s1 ^ 13) * s
					 
					 "and			r5 , r4 , #0xff0000									\n\t"//r5 = (r4) & 0xff0000
					 "add			r5 , %[quatPointer]	, r5 ,LSR #16					\n\t"//r5 = &q[k2] = q + (r5 >> 16)
					 "vst1.32		{d9[0]}	, [r5, :32]											\n\t"//q[k2] = (31 - s1 ^ 13) * s
					 
					 //q[k3]
					 "veor			d9 , d4 , d6										\n\t"//d9[1] = s0 ^ 32
					 "vsub.f32		s19 , s6 , s19										\n\t"//d9[1] = 23 - s0 ^ 32
					 "vmul.f32		d9 , d9 , d10										\n\t"//d9[1] = (23 - s0 ^ 32) * s
					 
					 "add			r5 , %[quatPointer], r4 , LSR #24					\n\t"//r5 = &q[k3] = q + (r4 >> 24)
					 "vst1.32		{d9[1]}	, [r5, :32]											\n\t"//q[k3] = (23 - s0 ^ 32) * s
					 
					 :	"+r"(matrix)
					 :	[SIMD_DW_mat2quatShuffle0]"r" (SIMD_DW_mat2quatShuffle0),
					 [SIMD_DW_mat2quatShuffle1]"r" (SIMD_DW_mat2quatShuffle1),
					 [SIMD_DW_mat2quatShuffle2]"r" (SIMD_DW_mat2quatShuffle2),
					 [SIMD_DW_mat2quatShuffle3]"r" (SIMD_DW_mat2quatShuffle3),
					 [NegSignMask]"r" (NegSignMask),
					 [quatPointer]"r" (quaternion)
					 :"r4" , "r5" ,"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7", "memory"
		);
	}
#endif//#ifdef NEON_ASM
	
#ifdef __cplusplus
}
#endif

#endif

