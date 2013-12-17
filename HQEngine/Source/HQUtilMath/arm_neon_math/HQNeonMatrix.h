/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_NEON_MATRIX_H
#define HQ_NEON_MATRIX_H

#include "../../HQUtilMathCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NEON_ASM
	void HQNeonMatrix4Inverse(const float *matrix1 , float *result , float * pDeterminant);
	
	void HQNeonMatrix3x4InverseToMatrix4(const float *matrix1 , float *result , float * pDeterminant);
	
	void HQNeonMatrix4Multiply(const float * matrix1 , const float *matrix2 , float * result );
	
	void HQNeonMultiMatrix4Multiply(const float * matrix , hq_uint32 numMat , float * result );
	
	void HQNeonVector4MultiplyMatrix4(const float *vector ,const float * matrix ,  float * resultVector );
	
	void HQNeonMultiVector4MultiplyMatrix4(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector );
	
	void HQNeonVector4TransformCoord(const float *vector ,const float * matrix ,  float * resultVector );
	
	void HQNeonMultiVector4TransformCoord(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector );
	
	void HQNeonVector4TransformNormal(const float *vector ,const float * matrix ,  float * resultVector );
	
	void HQNeonMultiVector4TransformNormal(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector );
	
	void HQNeonMatrix4MultiplyVector4(const float * matrix , const float *vector , float * resultVector );
	
	
	void HQNeonMatrix3x4Multiply(const float * matrix1 , const float *matrix2 , float * result );
	
	void HQNeonMatrix4MultiplyMatrix3x4(const float * matrix1 , const float *matrix2 , float * result );
	
	void HQNeonMultiMatrix3x4Multiply(const float * matrix , hq_uint32 numMat , float * result );
	
	void HQNeonMatrix3x4MultiplyVector4(const float * matrix , const float *vector , float * resultVector );
	
	void HQNeonMatrix3x4MultiplyMultiVector4(const float * matrix , const float *vector ,hq_uint32 numVec , float * resultVector );
	
	void HQNeonVector4TransformCoordMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector );

	void HQNeonMultiVector4TransformCoordMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix , float * resultVector );
	
	void HQNeonVector4TransformNormalMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector );
	
	void HQNeonMultiVector4TransformNormalMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix ,  float * resultVector );
#else//#ifdef NEON_ASM
//matrix1 (q4 - q7)  matrix2 (q8 - q11)
#define NEON_MATRIX_MUL_ASM_BLOCK1 \
"vmul.f32		q0  , q8 , d8[0]			\n\t"\
"vmul.f32		q1  , q8 , d10[0]			\n\t"\
"vmul.f32		q2  , q8 , d12[0]			\n\t"\
"vmul.f32		q3  , q8 , d14[0]			\n\t"\
\
"vmla.f32		q0 ,  q9 , d8[1]			\n\t"\
"vmla.f32		q1 ,  q9 , d10[1]			\n\t"\
"vmla.f32		q2 ,  q9 , d12[1]			\n\t"\
"vmla.f32		q3 ,  q9 , d14[1]			\n\t"\
\
"vmla.f32		q0 ,  q10 , d9[0]			\n\t"\
"vmla.f32		q1 ,  q10 , d11[0]			\n\t"\
"vmla.f32		q2 ,  q10 , d13[0]			\n\t"\
"vmla.f32		q3 ,  q10 , d15[0]			\n\t"\
\
"vmla.f32		q0 ,  q11 , d9[1]			\n\t"\
"vmla.f32		q1 ,  q11 , d11[1]			\n\t"\
"vmla.f32		q2 ,  q11 , d13[1]			\n\t"\
"vmla.f32		q3 ,  q11 , d15[1]			\n\t"
	
	
//matrix1 (q0 - q3)  matrix2 (q8 - q11)
#define NEON_MATRIX_MUL_ASM_BLOCK1_REV \
"vmul.f32		q4  , q8 , d0[0]			\n\t"\
"vmul.f32		q5  , q8 , d2[0]			\n\t"\
"vmul.f32		q6  , q8 , d4[0]			\n\t"\
"vmul.f32		q7  , q8 , d6[0]			\n\t"\
\
"vmla.f32		q4 ,  q9 , d0[1]			\n\t"\
"vmla.f32		q5 ,  q9 , d2[1]			\n\t"\
"vmla.f32		q6 ,  q9 , d4[1]			\n\t"\
"vmla.f32		q7 ,  q9 , d6[1]			\n\t"\
\
"vmla.f32		q4 ,  q10 , d1[0]			\n\t"\
"vmla.f32		q5 ,  q10 , d3[0]			\n\t"\
"vmla.f32		q6 ,  q10 , d5[0]			\n\t"\
"vmla.f32		q7 ,  q10 , d7[0]			\n\t"\
\
"vmla.f32		q4 ,  q11 , d1[1]			\n\t"\
"vmla.f32		q5 ,  q11 , d3[1]			\n\t"\
"vmla.f32		q6 ,  q11 , d5[1]			\n\t"\
"vmla.f32		q7 ,  q11 , d7[1]			\n\t"
	
	
//matrix1 (q3 - q5)  matrix2 (q6 - q8) .matrices is 3x4
#define NEON_MATRIX_MUL_ASM_BLOCK2 \
"vmul.f32		q0  , q6 , d6[0]			\n\t"\
"vmul.f32		q1  , q6 , d8[0]			\n\t"\
"vmul.f32		q2  , q6 , d10[0]			\n\t"\
\
"vadd.f32		s3 , s3, s15				\n\t"\
"vadd.f32		s7 , s7, s19				\n\t"\
"vadd.f32		s11 , s11, s23				\n\t"\
\
"vmla.f32		q0 ,  q7 , d6[1]			\n\t"\
"vmla.f32		q1 ,  q7 , d8[1]			\n\t"\
"vmla.f32		q2 ,  q7 , d10[1]			\n\t"\
\
"vmla.f32		q0 ,  q8 , d7[0]			\n\t"\
"vmla.f32		q1 ,  q8 , d9[0]			\n\t"\
"vmla.f32		q2 ,  q8 , d11[0]			\n\t"
	
//matrix1 (q0 - q2)  matrix2 (q6 - q8) .matrices is 3x4
#define NEON_MATRIX_MUL_ASM_BLOCK2_REV \
"vmul.f32		q3  , q6 , d0[0]			\n\t"\
"vmul.f32		q4  , q6 , d2[0]			\n\t"\
"vmul.f32		q5  , q6 , d4[0]			\n\t"\
\
"vadd.f32		s15 , s3, s15				\n\t"\
"vadd.f32		s19 , s7, s19				\n\t"\
"vadd.f32		s23 , s11, s23				\n\t"\
\
"vmla.f32		q3 ,  q7 , d0[1]			\n\t"\
"vmla.f32		q4 ,  q7 , d2[1]			\n\t"\
"vmla.f32		q5 ,  q7 , d4[1]			\n\t"\
\
"vmla.f32		q3 ,  q8 , d1[0]			\n\t"\
"vmla.f32		q4 ,  q8 , d3[0]			\n\t"\
"vmla.f32		q5 ,  q8 , d5[0]			\n\t"

//matrix1 (q4 - q7)  matrix2 (q8 - q10) .matrices1 is 4x4 , matrix2 is 3x4
#define NEON_MATRIX_MUL_ASM_BLOCK3 \
"vmul.f32		q0  , q8 , d8[0]			\n\t"\
"vmul.f32		q1  , q8 , d10[0]			\n\t"\
"vmul.f32		q2  , q8 , d12[0]			\n\t"\
"vmul.f32		q3  , q8 , d14[0]			\n\t"\
\
"vadd.f32		s3 , s3, s19				\n\t"\
"vadd.f32		s7 , s7, s23				\n\t"\
"vadd.f32		s11 , s11, s27				\n\t"\
"vadd.f32		s15 , s15, s31				\n\t"\
\
"vmla.f32		q0 ,  q9 , d8[1]			\n\t"\
"vmla.f32		q1 ,  q9 , d10[1]			\n\t"\
"vmla.f32		q2 ,  q9 , d12[1]			\n\t"\
"vmla.f32		q3 ,  q9 , d14[1]			\n\t"\
\
"vmla.f32		q0 ,  q10 , d9[0]			\n\t"\
"vmla.f32		q1 ,  q10 , d11[0]			\n\t"\
"vmla.f32		q2 ,  q10 , d13[0]			\n\t"\
"vmla.f32		q3 ,  q10 , d15[0]			\n\t"
						
//matrix (q1 - q4)  vector (q0)
#define NEON_MATRIX_MUL_VEC_ASM_BLOCK1 \
"vmul.f32		q5  , q1 , d0[0]			\n\t"\
\
"vmla.f32		q5 ,  q2 , d0[1]			\n\t"\
"vmla.f32		q5 ,  q3 , d1[0]			\n\t"\
"vmla.f32		q5 ,  q4 , d1[1]			\n\t"
	
//matrix (q1 - q4)  vector (q0) . transformNormal
#define NEON_MATRIX_MUL_VEC_ASM_BLOCK2 \
"vmul.f32		q5  , q1 , d0[0]			\n\t"\
\
"vmla.f32		q5 ,  q2 , d0[1]			\n\t"\
"vmla.f32		q5 ,  q3 , d1[0]			\n\t"

//matrix (q1 - q4)  vector (q0) . transformCoord
#define NEON_MATRIX_MUL_VEC_ASM_BLOCK3 \
"vmul.f32		q5  , q1 , d0[0]			\n\t"\
\
"vadd.f32		q5 ,  q5 , q4				\n\t"\
"vmla.f32		q5 ,  q2 , d0[1]			\n\t"\
"vmla.f32		q5 ,  q3 , d1[0]			\n\t"
	
//matrix3x4 (d2 - d5  , q3)  vector (q0)
#define NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK1 \
"vmul.f32		q3  , q3 , q0				\n\t"/*q3 = x * 31      y * 32		z * 33		w * 34*/\
"vpadd.f32		d9  , d6 , d7				\n\t"/*d9 = x * 31 + y * 32		z * 33 + w * 34*/\
"vadd.f32		s18 , s18 ,s19				\n\t"/*s18 = x * 31 + y * 32 + z * 33 + w * 34*/\
\
"vmul.f32		d8  , d2 , d0[0]			\n\t"/*x * 11   x * 21*/\
"vmla.f32		d8 ,  d3 , d0[1]			\n\t"/*x * 11 + y * 12			x * 21 + y * 22*/\
"vmla.f32		d8 ,  d4 , d1[0]			\n\t"/*x * 11 + y * 12 + z * 13			x * 21 + y * 22 + z * 23*/\
"vmla.f32		d8 ,  d5 , d1[1]			\n\t"/*d8 = x * 11 + y * 12 + z * 13 + w * 14			x * 21 + y * 22 + z * 23 + w * 24*/

//matrix3x4 (d2 - d5  , q3)  vector (q0) transformNormal
#define NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK2 \
"vmul.f32		q3  , q3 , q0				\n\t"/*q3 = x * 31      y * 32		z * 33		w * 34*/\
"vadd.f32		s18  , s12 , s13			\n\t"/*s18 = x * 31 + y * 32*/\
"vadd.f32		s18 , s18 ,s14				\n\t"/*s18 = x * 31 + y * 32 + z * 33*/\
\
"vmul.f32		d8  , d2 , d0[0]			\n\t"/*x * 11   x * 21*/\
"vmla.f32		d8 ,  d3 , d0[1]			\n\t"/*x * 11 + y * 12			x * 21 + y * 22*/\
"vmla.f32		d8 ,  d4 , d1[0]			\n\t"/*x * 11 + y * 12 + z * 13			x * 21 + y * 22 + z * 23*/
	
//matrix3x4 (d2 - d5  , q3)  vector (q0) transformCoord
#define NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK3 \
"vmul.f32		q3  , q3 , q0				\n\t"/*q3 = x * 31      y * 32		z * 33		(w = 1) * 34*/\
"vpadd.f32		d9  , d6 , d7				\n\t"/*d9 = x * 31 + y * 32		z * 33 + (w = 1) * 34*/\
"vadd.f32		s18 , s18 ,s19				\n\t"/*s18 = x * 31 + y * 32 + z * 33 + (w = 1) * 34*/\
\
"vmul.f32		d8  , d2 , d0[0]			\n\t"/*x * 11   x * 21*/\
"vadd.f32		d8 ,  d8 , d5				\n\t"/*d8 = x * 11 + 14			x * 21 + 24*/\
"vmla.f32		d8 ,  d3 , d0[1]			\n\t"/*x * 11 + y * 12 + 14			x * 21 + y * 22 + 24*/\
"vmla.f32		d8 ,  d4 , d1[0]			\n\t"/*x * 11 + y * 12 + z * 13 + 14			x * 21 + y * 22 + z * 23 + 24*/
	
	
	
	HQ_FORCE_INLINE void HQNeonMatrix4Inverse(const float *matrix1 , float *result , float * pDeterminant)
	{
		
		asm volatile(
					 "vld4.32	{d0 , d2 , d4 , d6} , [%1 , :128] !				\n\t"//first 2 transposed rows
					 "vld4.32	{d1 , d3 , d5 , d7} , [%1 , :128]					\n\t"//last 2 transposed rows
					//****************************
					// m11 m21 m31 m41		row0
					// m12 m22 m32 m42		row1
					// m13 m23 m33 m43		row2
					// m14 m24 m34 m44		row3
					//****************************
					 "vswp		d2 , d3										\n\t"//m12 m22 m32 m42			=>			m32 m42 m12 m22
					 "vswp		d6 , d7										\n\t"//m14 m24 m34 m44			=>			m34 m44 m14 m24
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 
					
					
					 "vmul.f32	q4 , q2 , q3								\n\t"//q4 = 31.43, 32.44 ,33.41 ,34.42
					 "vrev64.32	q4 , q4										\n\t"//q4 = 32.44, 31.43, 34.42, 33.41
					 "vmul.f32	q6 , q1 , q4								\n\t"//23.32.44, 24.31.43, 21.33.42, 22.33.41
					 "vmul.f32	q7 , q0 , q4								\n\t"//11.32.44, 12.31.43, 13.33.42, 14.33.41
					 "vswp		d8 , d9										\n\t"//q4 = 34.42, 33.41, 32.44, 31.43
					 "vmul.f32	q8 , q1 , q4								\n\t"//23.34.42, 24.33.41, 21.32.43, 22.31.43
					 "vmul.f32	q9 , q0 , q4								\n\t"//11.34.42, 12.33.41, 13.32.43, 14.31.43
					//q6 = 23.34.42 - 23.32.44, 24.33.41 - 24.31.43, 21.32.44 - 21.34.42, 22.31.43 - 22.33.41 = minor0
					 "vsub.f32	q6 , q8 , q6								\n\t"
					//q7 = 11.34.42 - 11.32.44, 12.33.41 - 12.31.43, 13.32.44 - 13.34.42, 14.31.43 - 14.33.41 = minor1
					 "vsub.f32	q7 , q9 , q7								\n\t"
					//q7 = 13.32.44 - 13.34.42, 14.31.43 - 14.33.41, 11.34.42 - 11.32.44, 12.33.41 - 12.31.43 = minor1
					 "vswp		d14 , d15									\n\t"
					//	-----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 , q1 , q2								\n\t"//q4 = 23.31, 24.32, 21.33, 22.34
					 "vrev64.32	q4 , q4										\n\t"//q4 = 24.32, 23.31, 22.34, 21.33
					 "vmul.f32	q8 , q3 , q4								\n\t"//q8 = 43.24.32, 44.23.31 , 41.22.34 , 42.21.33
					 "vmul.f32	q9 , q0 , q4								\n\t"//q9 = 11.24.32, 12.23.31 , 13.22.34 , 14.21.33 = minor3
					 "vswp		d8 , d9										\n\t"//q4 = 22.34 , 21.33 , 24.32 , 23.31
					 "vadd.f32	q6 , q8 , q6								\n\t"//q6 = minor0
					 "vmul.f32  q10 , q3 , q4								\n\t"//q10 = 43.22.34, 44.21.33 , 41.24.32 , 42.23.31
					 "vmul.f32	q11 , q0 , q4								\n\t"//q11 = 11.22.34, 12.21.33 , 13.24.32 , 14.23.31
					 "vsub.f32	q6 , q6 , q10								\n\t"//q6 = minor0
					 "vsub.f32	q9 , q11 , q9								\n\t"//q9 = minor3
					 "vswp		d18 , d19									\n\t"
					//	-----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmov		d20 , d3									\n\t"
					 "vmov		d21 , d2									\n\t"//q10 = swap (q1)
					 "vmul.f32	q4 , q10 , q3								\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vswp		d4 , d5										\n\t"//swap(q2)
					 "vmul.f32	q11 , q2 , q4								\n\t"
					 "vmul.f32	q8 , q0 , q4								\n\t"//q8 = minor2
					 "vadd.f32	q6 , q6 , q11								\n\t"//q6 = minor0
					 "vswp		d8 , d9										\n\t"//swap(q4)
					 "vmul.f32	q10 , q2 , q4								\n\t"
					 "vmul.f32	q11 , q0 , q4								\n\t"
					 "vsub.f32	q6 , q6 , q10								\n\t"
					 "vsub.f32	q8 , q11 , q8								\n\t"
					 "vswp		d16 , d17									\n\t"
					//	-----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 , q0 , q1								\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q2 , q4								\n\t"
					 "vadd.f32	q8 , q10 , q8								\n\t"
					 "vsub.f32	q9 , q11 , q9								\n\t"
					 "vswp		d8 , d9										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q2 , q4								\n\t"
					 "vsub.f32	q8 , q10 , q8								\n\t"
					 "vsub.f32	q9 , q9 , q11								\n\t"
					// -----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 ,q0 , q3									\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vmul.f32	q10 , q2 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vsub.f32	q7 , q7 , q10								\n\t"
					 "vadd.f32	q8 , q8 , q11								\n\t"
					 "vswp		d8 , d9										\n\t"
					 "vmul.f32	q10 , q2 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vadd.f32	q7 , q7 , q10								\n\t"
					 "vsub.f32	q8 , q8 , q11								\n\t"
					// -----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 , q0 , q2								\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vadd.f32	q7 , q7 , q10								\n\t"
					 "vsub.f32	q9 , q9 , q11								\n\t"
					 "vswp		d8 , d9										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vsub.f32	q7 , q7 , q10								\n\t"
					 "vadd.f32	q9 , q9 , q11								\n\t"
					
					// determinant
					 "vmul.f32	q5 , q0 , q6								\n\t"
					 "vpadd.f32	d10 , d10 , d11								\n\t"
					 "vpadd.f32	d10 , d10 , d10								\n\t"//d10 = det	,	det
					
					 //store determinant
					 "cmp		%2 , #0										\n\t"//pDeterminant == NULL?
					 "beq		L_CONTINUE%=								\n\t"//skip if NULL
					 //"fstsne	s20 , [%2]									\n\t"//store if not NULL pre UAL
					 "vst1.32	{d10[0]} , [%2]									\n\t"//store if not NULL 
					 
					 "L_CONTINUE%=:\n\t"  
					 "vrecpe.f32	d8 , d10				\n\t"//d8 ~= 1/det
					 
					 //Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 =  Y(n) * (2 - x * Y (n)) - reduce estimate error
					 //Y(n) = d8
					 //x = d10
					 "vrecps.f32	d9 , d8 , d10							\n\t"//d9 = (2 - d8 * d10)
					 "vmul.f32		d8 , d9 , d8							\n\t"//d8 ~= 1/det
					
					// multiply with 1/det, store result
					 "vmul.f32		q6 , q6 , d8[0]							\n\t"
					 "vmul.f32		q7 , q7 , d8[0]							\n\t"
					 "vmul.f32		q8 , q8 , d8[0]							\n\t"
					 "vmul.f32		q9 , q9 , d8[0]							\n\t"
					 
					 "vst1.32		{d12 , d13} , [%0 , :128] !					\n\t"
					 "vst1.32		{d14 , d15} , [%0 , :128] !					\n\t"
					 "vst1.32		{d16 , d17} , [%0 , :128] !					\n\t"
					 "vst1.32		{d18 , d19} , [%0 , :128]					\n\t"
					 
					 : "+r" (result) , "+r" (matrix1)
					 : "r" (pDeterminant)
					 : "q0" , "q1" , "q2" , "q3" , "q4"  , "q5" , "q6" ,
					   "q7" , "q8" , "q9" , "q10" , "q11" , "memory"
		);
	}
	
	HQ_FORCE_INLINE void HQNeonMatrix3x4InverseToMatrix4(const float *matrix1 , float *result , float * pDeterminant)
	{
		asm volatile(
					 "vld4.32	{d0 , d2 , d4 , d6} , [%1 , :128] !				\n\t"//first 2 transposed rows
					 "vld1.32	{d1 } , [%1 , :64] !							\n\t"//d1 = m31 m32
					 "vld1.32	{d5 } , [%1 , :64] 								\n\t"//d5 = m33 m34
					 "vmov.i64	d7  , #0x0000000000000000						\n\t"//d7 = 0	0
					 "vmov.i64	d3  , #0x0000000000000000						\n\t"//d3 = 0	0
					 "vmov.f32	s15	, #1.0										\n\t"//d7 = 0 	1
					 "vtrn.32	d1 , d3											\n\t"//d1 = m31 0  ; d3 = m32 0
					 "vtrn.32	d5 , d7											\n\t"//d5 = m33 0  ; d7 = m34 1
					//****************************
					// m11 m21 m31 0		row0	d0  d1
					// m12 m22 m32 0		row1	d2	d3
					// m13 m23 m33 0		row2	d4	d5
					// m14 m24 m34 1		row3	d6	d7
					//****************************
					 "vswp		d2 , d3										\n\t"//m12 m22 m32 m42			=>			m32 m42 m12 m22
					 "vswp		d6 , d7										\n\t"//m14 m24 m34 m44			=>			m34 m44 m14 m24
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 
					
					
					 "vmul.f32	q4 , q2 , q3								\n\t"//q4 = 31.43, 32.44 ,33.41 ,34.42
					 "vrev64.32	q4 , q4										\n\t"//q4 = 32.44, 31.43, 34.42, 33.41
					 "vmul.f32	q6 , q1 , q4								\n\t"//23.32.44, 24.31.43, 21.33.42, 22.33.41
					 "vmul.f32	q7 , q0 , q4								\n\t"//11.32.44, 12.31.43, 13.33.42, 14.33.41
					 "vswp		d8 , d9										\n\t"//q4 = 34.42, 33.41, 32.44, 31.43
					 "vmul.f32	q8 , q1 , q4								\n\t"//23.34.42, 24.33.41, 21.32.43, 22.31.43
					 "vmul.f32	q9 , q0 , q4								\n\t"//11.34.42, 12.33.41, 13.32.43, 14.31.43
					//q6 = 23.34.42 - 23.32.44, 24.33.41 - 24.31.43, 21.32.44 - 21.34.42, 22.31.43 - 22.33.41 = minor0
					 "vsub.f32	q6 , q8 , q6								\n\t"
					//q7 = 11.34.42 - 11.32.44, 12.33.41 - 12.31.43, 13.32.44 - 13.34.42, 14.31.43 - 14.33.41 = minor1
					 "vsub.f32	q7 , q9 , q7								\n\t"
					//q7 = 13.32.44 - 13.34.42, 14.31.43 - 14.33.41, 11.34.42 - 11.32.44, 12.33.41 - 12.31.43 = minor1
					 "vswp		d14 , d15									\n\t"
					//	-----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 , q1 , q2								\n\t"//q4 = 23.31, 24.32, 21.33, 22.34
					 "vrev64.32	q4 , q4										\n\t"//q4 = 24.32, 23.31, 22.34, 21.33
					 "vmul.f32	q8 , q3 , q4								\n\t"//q8 = 43.24.32, 44.23.31 , 41.22.34 , 42.21.33
					 "vmul.f32	q9 , q0 , q4								\n\t"//q9 = 11.24.32, 12.23.31 , 13.22.34 , 14.21.33 = minor3
					 "vswp		d8 , d9										\n\t"//q4 = 22.34 , 21.33 , 24.32 , 23.31
					 "vadd.f32	q6 , q8 , q6								\n\t"//q6 = minor0
					 "vmul.f32  q10 , q3 , q4								\n\t"//q10 = 43.22.34, 44.21.33 , 41.24.32 , 42.23.31
					 "vmul.f32	q11 , q0 , q4								\n\t"//q11 = 11.22.34, 12.21.33 , 13.24.32 , 14.23.31
					 "vsub.f32	q6 , q6 , q10								\n\t"//q6 = minor0
					 "vsub.f32	q9 , q11 , q9								\n\t"//q9 = minor3
					 "vswp		d18 , d19									\n\t"
					//	-----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmov		d20 , d3									\n\t"
					 "vmov		d21 , d2									\n\t"//q10 = swap (q1)
					 "vmul.f32	q4 , q10 , q3								\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vswp		d4 , d5										\n\t"//swap(q2)
					 "vmul.f32	q11 , q2 , q4								\n\t"
					 "vmul.f32	q8 , q0 , q4								\n\t"//q8 = minor2
					 "vadd.f32	q6 , q6 , q11								\n\t"//q6 = minor0
					 "vswp		d8 , d9										\n\t"//swap(q4)
					 "vmul.f32	q10 , q2 , q4								\n\t"
					 "vmul.f32	q11 , q0 , q4								\n\t"
					 "vsub.f32	q6 , q6 , q10								\n\t"
					 "vsub.f32	q8 , q11 , q8								\n\t"
					 "vswp		d16 , d17									\n\t"
					//	-----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 , q0 , q1								\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q2 , q4								\n\t"
					 "vadd.f32	q8 , q10 , q8								\n\t"
					 "vsub.f32	q9 , q11 , q9								\n\t"
					 "vswp		d8 , d9										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q2 , q4								\n\t"
					 "vsub.f32	q8 , q10 , q8								\n\t"
					 "vsub.f32	q9 , q9 , q11								\n\t"
					// -----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 ,q0 , q3									\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vmul.f32	q10 , q2 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vsub.f32	q7 , q7 , q10								\n\t"
					 "vadd.f32	q8 , q8 , q11								\n\t"
					 "vswp		d8 , d9										\n\t"
					 "vmul.f32	q10 , q2 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vadd.f32	q7 , q7 , q10								\n\t"
					 "vsub.f32	q8 , q8 , q11								\n\t"
					// -----------------------------------------------
					
					//**************************************
					//transposed matrix 
					//11 12 13 14<====row0	
					//23 24 21 22<====row1
					//31 32 33 34<====row2
					//43 44 41 42<====row3
					//**************************************
					 "vmul.f32	q4 , q0 , q2								\n\t"
					 "vrev64.32	q4 , q4										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vadd.f32	q7 , q7 , q10								\n\t"
					 "vsub.f32	q9 , q9 , q11								\n\t"
					 "vswp		d8 , d9										\n\t"
					 "vmul.f32	q10 , q3 , q4								\n\t"
					 "vmul.f32	q11 , q1 , q4								\n\t"
					 "vsub.f32	q7 , q7 , q10								\n\t"
					 "vadd.f32	q9 , q9 , q11								\n\t"
					
					// determinant
					 "vmul.f32	q5 , q0 , q6								\n\t"
					 "vpadd.f32	d10 , d10 , d11								\n\t"
					 "vpadd.f32	d10 , d10 , d10								\n\t"//d10 = det	,	det
					
					 //store determinant
					 "cmp		%2 , #0										\n\t"//pDeterminant == NULL?
					 "beq		L_CONTINUE%=								\n\t"//skip if NULL
					 "vst1.32	{d10[0]} , [%2]									\n\t"//store if not NULL
					 
					 
					 "L_CONTINUE%=:\n\t"  
					 "vrecpe.f32	d8 , d10				\n\t"//d8 ~= 1/det
					 
					 //Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 =  Y(n) * (2 - x * Y (n)) - reduce estimate error
					 //Y(n) = d8
					 //x = d10
					 "vrecps.f32	d9 , d8 , d10							\n\t"//d9 = (2 - d8 * d10)
					 "vmul.f32		d8 , d9 , d8							\n\t"//d8 ~= 1/det
					
					// multiply with 1/det, store result
					 "vmul.f32		q6 , q6 , d8[0]							\n\t"
					 "vmul.f32		q7 , q7 , d8[0]							\n\t"
					 "vmul.f32		q8 , q8 , d8[0]							\n\t"
					 "vmul.f32		q9 , q9 , d8[0]							\n\t"
					 
					 "vst1.32		{d12 , d13} , [%0 , :128] !					\n\t"
					 "vst1.32		{d14 , d15} , [%0 , :128] !					\n\t"
					 "vst1.32		{d16 , d17} , [%0 , :128] !					\n\t"
					 "vst1.32		{d18 , d19} , [%0 , :128]					\n\t"
					 
					 : "+r" (result) , "+r" (matrix1)
					 : "r" (pDeterminant)
					 : "q0" , "q1" , "q2" , "q3" , "q4"  , "q5" , "q6" ,
					   "q7" , "q8" , "q9" , "q10" , "q11" , "memory"
		);
	}
	
	/*
	void HQNeonMatrix4InverseNonInline(const float *matrix1 , float *result , float * pDeterminant)  __attribute__ ((noinline));
	
	void HQNeonMatrix4InverseNonInline(const float *matrix1 , float *result , float * pDeterminant)
	{
		HQNeonMatrix4Inverse(matrix1 , result, pDeterminant);
	}
	*/
	HQ_FORCE_INLINE void HQNeonMatrix4Multiply(const float * matrix1 , const float *matrix2 , float * result )
	{
		
		asm volatile(
					 "vld1.32		{d8 , d9} , [%0 , :128]	!			\n\t"//1st row of matrix1
					 "vld1.32		{d10 , d11} , [%0 , :128]	!		\n\t"//2nd row of matrix1
					 "vld1.32		{d12 , d13} , [%0 , :128]	!		\n\t"//3rd row of matrix1
					 "vld1.32		{d14 , d15} , [%0 , :128]			\n\t"//4th row of matrix1
					 
					 "vld1.32		{d16 , d17} , [%1 , :128]	!		\n\t"//1st row of matrix2
					 "vld1.32		{d18 , d19} , [%1 , :128]	!		\n\t"//2nd row of matrix2
					 "vld1.32		{d20 , d21} , [%1 , :128]	!		\n\t"//3rd row of matrix2
					 "vld1.32		{d22 , d23} , [%1 , :128]			\n\t"//4th row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK1
					 
					 "vst1.32		{d0 , d1}  , [%2 , :128]	!		\n\t"//store first row	
					 "vst1.32		{d2 , d3}  , [%2 , :128]	!		\n\t"//store second row	
					 "vst1.32		{d4 , d5}  , [%2 , :128]	!		\n\t"//store third row
					 "vst1.32		{d6 , d7}  , [%2 , :128]			\n\t"//store fourth row
					 
					 :"+r" (matrix1) , "+r" (matrix2) , "+r" (result)
					 ::"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7" , "q8" , "q9",
					   "q10" , "q11" , "memory"
		);
		 
	}
	
	HQ_FORCE_INLINE void HQNeonMultiMatrix4Multiply(const float * matrix , hq_uint32 numMat , float * result )
	{
		
		asm volatile(
					 "and			r5  ,  %[numMat] , #0x1				\n\t"
					 
					 "cmp			r5 , #0								\n\t"//if r5 = 0 , num matrices is divisible by 2.If it is , number of matrix pairs is odd
					 "bne			LEVEN_PAIRS%=						\n\t"//jump if r5 != 0 (we have even number of pairs)
					 
					 //multiply first pair of matrices
					 "vld1.32		{d8 , d9} , [%0 , :128]	!			\n\t"//1st row of matrix1
					 "vld1.32		{d10 , d11} , [%0 , :128]	!		\n\t"//2nd row of matrix1
					 "vld1.32		{d12 , d13} , [%0 , :128]	!		\n\t"//3rd row of matrix1
					 "vld1.32		{d14 , d15} , [%0 , :128]	!		\n\t"//4th row of matrix1
					 
					 "vld1.32		{d16 , d17} , [%0 , :128]	!		\n\t"//1st row of matrix2
					 "vld1.32		{d18 , d19} , [%0 , :128]	!		\n\t"//2nd row of matrix2
					 "vld1.32		{d20 , d21} , [%0 , :128]	!		\n\t"//3rd row of matrix2
					 "vld1.32		{d22 , d23} , [%0 , :128]	!		\n\t"//4th row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK1							
					 
					 "mov			r5 , #2								\n\t"//start loop at matrix index 2
					 "b				LSTART_LOOP%=						\n\t"
					 
					 "LEVEN_PAIRS%=:\n\t"
					 //we have even pairs , so just load the first matrix
					 "vld1.32		{d0 , d1} , [%0 , :128]	!		\n\t"//1st row of matrix1
					 "vld1.32		{d2 , d3} , [%0 , :128]	!		\n\t"//2nd row of matrix1
					 "vld1.32		{d4 , d5} , [%0 , :128]	!		\n\t"//3rd row of matrix1
					 "vld1.32		{d6 , d7} , [%0 , :128]	!		\n\t"//4th row of matrix1
					 
					 "LSTART_LOOP%=:\n\t"//multiply 2 pairs of matrices each iteration
					 
					 "cmp			r5 , %[numMat]				\n\t"
					 "bhs			LEND_LOOP%=					\n\t"//end loop if r5 >= numMat
					 
					 "vld1.32		{d16 , d17} , [%0 , :128]	!		\n\t"//1st row of matrix2
					 "vld1.32		{d18 , d19} , [%0 , :128]	!		\n\t"//2nd row of matrix2
					 "vld1.32		{d20 , d21} , [%0 , :128]	!		\n\t"//3rd row of matrix2
					 "vld1.32		{d22 , d23} , [%0 , :128]	!		\n\t"//4th row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK1_REV
					 
					 "vld1.32		{d16 , d17} , [%0 , :128]	!		\n\t"//1st row of matrix2
					 "vld1.32		{d18 , d19} , [%0 , :128]	!		\n\t"//2nd row of matrix2
					 "vld1.32		{d20 , d21} , [%0 , :128]	!		\n\t"//3rd row of matrix2
					 "vld1.32		{d22 , d23} , [%0 , :128]	!		\n\t"//4th row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK1
					 
					 "add			r5 , r5 , #2						\n\t"//r5 += 2
					 "b				LSTART_LOOP%=						\n\t"
					 
					 "LEND_LOOP%=:\n\t"
					 
					 "vst1.32		{d0 , d1}  , [%1 , :128]	!		\n\t"//store first row	
					 "vst1.32		{d2 , d3}  , [%1 , :128]	!		\n\t"//store second row	
					 "vst1.32		{d4 , d5}  , [%1 , :128]	!		\n\t"//store third row
					 "vst1.32		{d6 , d7}  , [%1 , :128]			\n\t"//store fourth row
					 
					 :"+r" (matrix) , "+r" (result)
					 :[numMat]"r"(numMat)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7" , "q8" , "q9",
					 "q10" , "q11" , "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonVector4MultiplyMatrix4(const float *vector ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1 , :128]				\n\t"//vector
					 
					 "vld1.32		{d2 , d3} , [%0 , :128]	!			\n\t"//1st row of matrix
					 "vld1.32		{d4 , d5} , [%0 , :128]	!			\n\t"//2nd row of matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]	!			\n\t"//3rd row of matrix
					 "vld1.32		{d8 , d9} , [%0 , :128]				\n\t"//4th row of matrix
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK1
					 
					 "vst1.32		{d10 , d11}  , [%2 , :128]			\n\t"
					 
					 :"+r" (matrix)
					 :"r" (vector) , "r"(resultVector)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMultiVector4MultiplyMatrix4(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d2 , d3} , [%0 , :128]	!			\n\t"//1st row of matrix
					 "vld1.32		{d4 , d5} , [%0 , :128]	!			\n\t"//2nd row of matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]	!			\n\t"//3rd row of matrix
					 "vld1.32		{d8 , d9} , [%0 , :128]				\n\t"//4th row of matrix
					 
					 
					 "mov			r5	, #0						\n\t"//r5 = 0
					 
					 "L_LOOP%=:\n\t"
					 "cmp			r5 , %[numVec]					\n\t"
					 "bhs			LEND_LOOP%=						\n\t"//end loop if r5 >= numVec
					 
					 "vld1.32		{d0 , d1} , [%1 , :128]	!			\n\t"//vector
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK1
					 
					 "vst1.32		{d10 , d11}  , [%2 , :128]!			\n\t"
					 
					 "add			r5 , r5 , #1					\n\t"//increase by 1
					 "b				L_LOOP%=						\n\t"//loop 
					 
					 "LEND_LOOP%=:\n\t"
					 
					 :"+r" (matrix), "+r" (vector) , "+r"(resultVector)
					 :[numVec] "r" (numVec)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonVector4TransformCoord(const float *vector ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1 , :128]				\n\t"//vector
					 
					 "vld1.32		{d2 , d3} , [%0 , :128]	!			\n\t"//1st row of matrix
					 "vld1.32		{d4 , d5} , [%0 , :128]	!			\n\t"//2nd row of matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]	!			\n\t"//3rd row of matrix
					 "vld1.32		{d8 , d9} , [%0 , :128]				\n\t"//4th row of matrix
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK3
					 
					 "vst1.32		{d10 , d11}  , [%2 , :128]			\n\t"
					 
					 :"+r" (matrix)
					 :"r" (vector) , "r"(resultVector)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMultiVector4TransformCoord(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d2 , d3} , [%0 , :128]	!			\n\t"//1st row of matrix
					 "vld1.32		{d4 , d5} , [%0 , :128]	!			\n\t"//2nd row of matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]	!			\n\t"//3rd row of matrix
					 "vld1.32		{d8 , d9} , [%0 , :128]				\n\t"//4th row of matrix
					 
					 
					 "mov			r5	, #0							\n\t"//r5 = count
					 
					 "L_LOOP%=:\n\t"
					 
					 "cmp			r5 , %[numVec]					\n\t"
					 "bhs			LEND_LOOP%=						\n\t"//end loop if r5 >= numVec
					 
					 "vld1.32		{d0 , d1} , [%1 , :128]	!			\n\t"//vector
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK3
					 
					 "vst1.32		{d10 , d11}  , [%2 , :128]!			\n\t"
					 
					 "add			r5 , r5 , #1						\n\t"//increase by 1
					 "b			L_LOOP%=								\n\t"//loop 
					 
					 "LEND_LOOP%=:\n\t"
					 
					 :"+r" (matrix), "+r" (vector) , "+r"(resultVector)
					 :[numVec] "r" (numVec)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonVector4TransformNormal(const float *vector ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1 , :128]				\n\t"//vector
					 
					 "vld1.32		{d2 , d3} , [%0 , :128]	!			\n\t"//1st row of matrix
					 "vld1.32		{d4 , d5} , [%0 , :128]	!			\n\t"//2nd row of matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]	!			\n\t"//3rd row of matrix
					 "vld1.32		{d8 , d9} , [%0 , :128]				\n\t"//4th row of matrix
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK2
					 
					 "vst1.32		{d10 , d11}  , [%2 , :128]			\n\t"
					 
					 :"+r" (matrix)
					 :"r" (vector) , "r"(resultVector)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMultiVector4TransformNormal(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d2 , d3} , [%0 , :128]	!			\n\t"//1st row of matrix
					 "vld1.32		{d4 , d5} , [%0 , :128]	!			\n\t"//2nd row of matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]	!			\n\t"//3rd row of matrix
					 "vld1.32		{d8 , d9} , [%0 , :128]				\n\t"//4th row of matrix
					 
					 
					 "mov			r5	, #0							\n\t"//r5 = count
					 
					 "L_LOOP%=:\n\t"
					 
					 "cmp			r5 , %[numVec]					\n\t"
					 "bhs			LEND_LOOP%=						\n\t"//end loop if r5 >= numVec
					 
					 "vld1.32		{d0 , d1} , [%1 , :128]	!			\n\t"//vector
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK2
					 
					 "vst1.32		{d10 , d11}  , [%2 , :128]!			\n\t"
					 
					 "add			r5 , r5 , #1					\n\t"//increase by 1
					 "b			L_LOOP%=							\n\t"
					 
					 "LEND_LOOP%=:									\n\t"
					 
					 :"+r" (matrix), "+r" (vector) , "+r"(resultVector)
					 :[numVec] "r" (numVec)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMatrix4MultiplyVector4(const float * matrix , const float *vector , float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1 , :128]				\n\t"//vector
					 
					 "vld4.32		{d2 , d4  ,d6 , d8} , [%0 , :128]!	\n\t"//2 rows of transposed matrix
					 "vld4.32		{d3 , d5  ,d7 , d9} , [%0 , :128]	\n\t"//last 2 rows of transposed matrix
					 
					 NEON_MATRIX_MUL_VEC_ASM_BLOCK1
					 
					 "vst1.32		{d10 , d11}  , [%2  , :128]			\n\t"
					 
					 :"+r" (matrix)
					 :"r" (vector) , "r" (resultVector) 
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "memory"
					 );
		
	}
	
	
	HQ_FORCE_INLINE void HQNeonMatrix3x4Multiply(const float * matrix1 , const float *matrix2 , float * result )
	{
		
		asm volatile(
					 "vld1.32		{d6 , d7} , [%0 , :128]	!				\n\t"//1st row of matrix1
					 "vld1.32		{d8 , d9} , [%0 , :128]	!				\n\t"//2nd row of matrix1
					 "vld1.32		{d10 , d11} , [%0 , :128]				\n\t"//3rd row of matrix1
					 
					 "vld1.32		{d12 , d13} , [%1 , :128]	!			\n\t"//1st row of matrix2
					 "vld1.32		{d14 , d15} , [%1 , :128]	!			\n\t"//2nd row of matrix2
					 "vld1.32		{d16 , d17} , [%1 , :128]				\n\t"//3rd row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK2
					 
					 "vst1.32		{d0 , d1}  , [%2 , :128]	!		\n\t"//store first row	
					 "vst1.32		{d2 , d3}  , [%2 , :128]	!		\n\t"//store second row	
					 "vst1.32		{d4 , d5}  , [%2 , :128]			\n\t"//store third row
					 
					 :"+r" (matrix1) , "+r" (matrix2) , "+r" (result)
					 ::"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7" , "q8", "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMatrix4MultiplyMatrix3x4(const float * matrix1 , const float *matrix2 , float * result )
	{
		
		asm volatile(
					 "vld1.32		{d8 , d9} , [%0 , :128]	!				\n\t"//1st row of matrix1
					 "vld1.32		{d10 , d11} , [%0 , :128]	!			\n\t"//2nd row of matrix1
					 "vld1.32		{d12 , d13} , [%0 , :128]	!			\n\t"//3rd row of matrix1
					 "vld1.32		{d14 , d15} , [%0 , :128]				\n\t"//4th row of matrix1
					 
					 "vld1.32		{d16 , d17} , [%1 , :128]	!			\n\t"//1st row of matrix2
					 "vld1.32		{d18 , d19} , [%1 , :128]	!			\n\t"//2nd row of matrix2
					 "vld1.32		{d20 , d21} , [%1 , :128]				\n\t"//3rd row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK3
					 
					 "vst1.32		{d0 , d1}  , [%2 , :128]	!		\n\t"//store first row	
					 "vst1.32		{d2 , d3}  , [%2 , :128]	!		\n\t"//store second row	
					 "vst1.32		{d4 , d5}  , [%2 , :128]	!		\n\t"//store third row
					 "vst1.32		{d6 , d7}  , [%2 , :128]			\n\t"//store fourth row
					 
					 :"+r" (matrix1) , "+r" (matrix2) , "+r" (result)
					 ::"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7" , "q8", "q9" , "q10" , "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMultiMatrix3x4Multiply(const float * matrix , hq_uint32 numMat , float * result )
	{
		
		asm volatile(
					 "and			r5  ,  %[numMat] , #0x1				\n\t"
					 
					 "cmp			r5 , #0								\n\t"//if r5 = 0 , num matrices is divisible by 2.If it is , number of matrix pairs is odd
					 "bne			LEVEN_PAIRS%=						\n\t"//jump if r5 != 0 (we have even number of pairs)
					 
					 //multiply first pair of matrices
					 "vld1.32		{d6 , d7} , [%0 , :128]	!				\n\t"//1st row of matrix1
					 "vld1.32		{d8 , d9} , [%0 , :128]	!				\n\t"//2nd row of matrix1
					 "vld1.32		{d10 , d11} , [%0 , :128]	!			\n\t"//3rd row of matrix1
					 
					 "vld1.32		{d12 , d13} , [%0 , :128]	!			\n\t"//1st row of matrix2
					 "vld1.32		{d14 , d15} , [%0 , :128]	!			\n\t"//2nd row of matrix2
					 "vld1.32		{d16 , d17} , [%0 , :128]	!			\n\t"//3rd row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK2					
					 
					 "mov			r5 , #2									\n\t"//start loop at matrix index 2
					 "b				LSTART_LOOP%=							\n\t"
					 
					 "LEVEN_PAIRS%=:\n\t"
					 //we have even pairs , so just load the first matrix
					 "vld1.32		{d0 , d1} , [%0 , :128]	!				\n\t"//1st row of matrix1
					 "vld1.32		{d2 , d3} , [%0 , :128]	!				\n\t"//2nd row of matrix1
					 "vld1.32		{d4 , d5} , [%0 , :128]	!				\n\t"//3rd row of matrix1
					 
					 
					 "LSTART_LOOP%=:\n\t"//multiply 2 pairs of matrices each iteration
					 
					 "cmp			r5 , %[numMat]					\n\t"
					 "bhs			LEND_LOOP%=						\n\t"//end loop if r5 >= numMat
					 
					 "vld1.32		{d12 , d13} , [%0 , :128]	!			\n\t"//1st row of matrix2
					 "vld1.32		{d14 , d15} , [%0 , :128]	!			\n\t"//2nd row of matrix2
					 "vld1.32		{d16 , d17} , [%0 , :128]	!			\n\t"//3rd row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK2_REV
					 
					 "vld1.32		{d12 , d13} , [%0 , :128]	!			\n\t"//1st row of matrix2
					 "vld1.32		{d14 , d15} , [%0 , :128]	!			\n\t"//2nd row of matrix2
					 "vld1.32		{d16 , d17} , [%0 , :128]	!			\n\t"//3rd row of matrix2
					 
					 NEON_MATRIX_MUL_ASM_BLOCK2
					 
					 "add			r5 , r5 , #2					\n\t"//r5 += 2
					 "b				LSTART_LOOP%=					\n\t"
					 
					 "LEND_LOOP%=:									\n\t"
					 
					 "vst1.32		{d0 , d1}  , [%1 , :128]	!			\n\t"//store first row	
					 "vst1.32		{d2 , d3}  , [%1 , :128]	!			\n\t"//store second row	
					 "vst1.32		{d4 , d5}  , [%1 , :128]				\n\t"//store third row
					 
					 :"+r" (matrix), "+r" (result)
					 :[numMat] "r" (numMat)
					 :"q0" , "q1" , "q2" , "q3" , "q4" , "q5" , "q6" , "q7" , "q8", "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMatrix3x4MultiplyVector4(const float * matrix , const float *vector , float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0 , d1} , [%1 , :128]						\n\t"//vector
					 
					 "vld4.32		{d2 , d3  ,d4 , d5} , [%0 , :128]!				\n\t"//2 rows of transposed matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]						\n\t"//last row of matrix
					 
					 NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK1
					 
					 "vmov.f32			s19   ,  s3								\n\t"//copy w
					 "vst1.32		{d8 , d9}  , [%2  , :128]						\n\t"//store
					 
					 :"+r" (matrix)
					 :"r" (vector) , "r" (resultVector)
					 :"q0" , "q1" , "q2" , "q3" , "q4", "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMatrix3x4MultiplyMultiVector4(const float * matrix , const float *vector ,hq_uint32 numVec , float * resultVector )
	{
		asm volatile(
					 "vld4.32		{d2 , d3  ,d4 , d5} , [%0 , :128]!			\n\t"//2 rows of transposed matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]					\n\t"//last row of matrix
					 
					 "mov			r5	, #0							\n\t"//r5 = 0
					 
					 "L_LOOP%=:\n\t"
					 
					 "cmp			r5 , %[numVec]						\n\t"
					 "bhs			LEND_LOOP%=							\n\t"//end loop if r5 >= numVec
					 
					 "vld1.32		{d0 , d1} , [%1 , :128]	!				\n\t"//vector
					 
					 NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK1
					 
					 "vmov.f32			s19   ,  s3							\n\t"//copy w
					 "vst1.32		{d8 , d9}  , [%2  , :128]!					\n\t"
					 
					 "add			r5 , r5 , #1						\n\t"//increase by 1
					 "b				L_LOOP%=							\n\t"//loop 
					 
					 "LEND_LOOP%=:										\n\t"
					 
					 :"+r" (matrix) , "+r" (vector) , "+r" (resultVector) 
					 :[numVec]"r" (numVec)
					 :"q0" , "q1" , "q2" , "q3" , "q4", "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonVector4TransformCoordMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0} , [%2  , :64]!						\n\t"//vector d0 =  x y
					 "vld1.32		{d1[0]}  , [%2 , :32]							\n\t"//q0 = x y z _
					 "vmov.f32		s3	, #1.0							\n\t"//q0 = x y z 1
					 
					 "vld4.32		{d2 , d3  ,d4 , d5} , [%0 , :128]!			\n\t"//2 rows of transposed matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]					\n\t"//last row of matrix
					 
					 NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK3
					 
					 "vst1.32		{d8}  , [%1  , :64 ]!						\n\t"//store x y
					 "vst1.32		{d9[0]}   , [%1, :32] !						\n\t"//store z
					 "vst1.32		{d1[1]}	  , [%1 ]					\n\t"//store 1
					 
					 :"+r" (matrix) , "+r" (resultVector) ,"+r" (vector)
					 :
					 :"q0" , "q1" , "q2" , "q3" , "q4", "memory"
					 );
		
	}

	HQ_FORCE_INLINE void HQNeonMultiVector4TransformCoordMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix , float * resultVector )
	{
		asm volatile(
					 "vld4.32		{d2 , d3  ,d4 , d5} , [%0 , :128]!			\n\t"//2 rows of transposed matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]					\n\t"//last row of matrix
					 
					 "vmov.f32		s3	, #1.0							\n\t"//q0 = _ _ _ 1
					 
					 "mov			r5	, #0							\n\t"//r5 = 0
					 
					 "L_LOOP%=:\n\t"
					 
					 "cmp			r5 , %[numVec]						\n\t"
					 "bhs			LEND_LOOP%=							\n\t"//end loop if r5 >= numVec
					 
					 "vld1.32		{d0} , [%1 , :64]	!						\n\t"//x y
					 "vld1.32		{d1[0]}   , [%1 , :32]							\n\t"//z
					 
					 NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK3
					 
					 "vst1.32		{d8}  , [%2 , :64]!						\n\t"//store x y
					 "vst1.32		{d9[0]}   , [%2, :32]! 						\n\t"//store z
					 "vst1.32		{d1[1]}	  , [%2 ]					\n\t"//store 1
					 
					 "add			%1 , %1 , #8						\n\t"//next source vector
					 "add			%2 , %2 , #8						\n\t"//next dest vector
					 
					 "add			r5 , r5 , #1						\n\t"//increase by 1
					 "b				L_LOOP%=							\n\t"//loop
					 
					 "LEND_LOOP%=:										\n\t"
					 
					 :"+r" (matrix) , "+r" (vector) , "+r" (resultVector) 
					 :[numVec]"r" (numVec)
					 :"q0" , "q1" , "q2" , "q3" , "q4", "memory" , "r5"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonVector4TransformNormalMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld1.32		{d0 , d1} , [%2 , :128]				\n\t"//vector
					 "vmov.i64		d9 , 0x0000000000000000				\n\t"//d9 = 0 0
					 
					 "vld4.32		{d2 , d3  ,d4 , d5} , [%0 , :128]!	\n\t"//2 rows of transposed matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]				\n\t"//last row of matrix
					 
					 NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK2
					 
					 "vst1.32		{d8 , d9}  , [%1 , :128 ]				\n\t"//store x y z 0
					 
					 :"+r" (matrix) 
					 :"r" (resultVector)  , "r" (vector)
					 :"q0" , "q1" , "q2" , "q3" , "q4", "memory"
					 );
		
	}
	
	HQ_FORCE_INLINE void HQNeonMultiVector4TransformNormalMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix ,  float * resultVector )
	{
		
		asm volatile(
					 "vld4.32		{d2 , d3  ,d4 , d5} , [%0 , :128]!	\n\t"//2 rows of transposed matrix
					 "vld1.32		{d6 , d7} , [%0 , :128]				\n\t"//last row of matrix
					 "vmov.i64		d9 , 0x0000000000000000				\n\t"//d9 = 0 0
					 
					 "mov			r5	, #0							\n\t"//r5 = 0
					 
					 "L_LOOP%=:\n\t"
					 
					 "cmp			r5 , %[numVec]						\n\t"
					 "bhs			LEND_LOOP%=							\n\t"//end loop if r5 >= numVec
					 
					 "vld1.32		{d0 , d1} , [%1 , :128]!					\n\t"//vector
					 
					 NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK2
					 
					 "vst1.32		{d8 , d9}  , [%2 , :128]!					\n\t"//store x y z 0
					 
					 "add			r5 , r5 , #1						\n\t"//increase by 1
					 "b				L_LOOP%=							\n\t"//loop
					 
					 "LEND_LOOP%=:										\n\t"
					 
					 :"+r" (matrix)   , "+r" (vector),"+r" (resultVector)
					 :[numVec] "r" (numVec)
					 :"q0" , "q1" , "q2" , "q3" , "q4", "memory" , "r5"
					 );
		
	}

#endif//#ifdef NEON_ASM
	
#ifdef __cplusplus
}
#endif


#endif

