/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
#ifdef NEON_MATH
#include "arm_neon_math/HQNeonMatrix.h"
#elif defined HQ_DIRECTX_MATH
#include "directx_math/HQDXMatrix.h"
#endif
#include <stdio.h>
#include <assert.h>

//***********************************
//nhân matran với vector
//***********************************
#ifndef HQ_EXPLICIT_ALIGN
HQVector4 HQMatrix4::operator *(const HQVector4 &v) const{
	HQ_DECL_STACK_VECTOR4(result);
#ifdef CMATH
	

	result.x=_11*v.x+_12*v.y+_13*v.z+_14*v.w;
	result.y=_21*v.x+_22*v.y+_23*v.z+_24*v.w;
	result.z=_31*v.x+_32*v.y+_33*v.z+_34*v.w;
	result.w=_41*v.x+_42*v.y+_43*v.z+_44*v.w;
	
#elif defined NEON_MATH

	HQNeonMatrix4MultiplyVector4(this->m, v, result);
	
#elif defined HQ_DIRECTX_MATH

	HQDXMatrix4MultiplyVector4(this->m, v, result);

#else
	/* SSE version*/

	float4 m0,m1,m2,m3,m4,m5,m6,m7;

	m0=_mm_load_ps(m); 
	m1=_mm_load_ps(m+4);
	m2=_mm_load_ps(m+8);
	m3=_mm_load_ps(m+12);

	m5=_mm_shuffle_ps(m0,m1,SSE_SHUFFLEL(0,1,0,1));
	m6=_mm_shuffle_ps(m2,m3,SSE_SHUFFLEL(0,1,0,1));
	m4=_mm_shuffle_ps(m5,m6,SSE_SHUFFLEL(0,2,0,2));
	m5=_mm_shuffle_ps(m5,m6,SSE_SHUFFLEL(1,3,1,3));

	m7=_mm_shuffle_ps(m0,m1,SSE_SHUFFLEL(2,3,2,3));
	m0=_mm_shuffle_ps(m2,m3,SSE_SHUFFLEL(2,3,2,3));
	m6=_mm_shuffle_ps(m7,m0,SSE_SHUFFLEL(0,2,0,2));
	m7=_mm_shuffle_ps(m7,m0,SSE_SHUFFLEL(1,3,1,3));

	m0=_mm_load_ps(v.v);
	m1=m0;
	m2=m0;
	m3=m0;

	m0=_mm_shuffle_ps(m0,m0,0x00);
	m1=_mm_shuffle_ps(m1,m1,0x55);
	m2=_mm_shuffle_ps(m2,m2,0xaa);
	m3=_mm_shuffle_ps(m3,m3,0xff);

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	m3=_mm_mul_ps(m3,m7);

	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(result.v,m0);
#endif
	return result;
}

#endif//#ifndef HQ_EXPLICIT_ALIGN

HQVector4* HQMatrix4rMulVec(const HQMatrix4* mat,const HQVector4* v1,HQVector4* out)
{
#ifdef CMATH
	if(out == v1)
	{
		hq_float32 X=mat->_11*v1->x+mat->_12*v1->y+mat->_13*v1->z+mat->_14*v1->w;
		hq_float32 Y=mat->_21*v1->x+mat->_22*v1->y+mat->_23*v1->z+mat->_24*v1->w;
		hq_float32 Z=mat->_31*v1->x+mat->_32*v1->y+mat->_33*v1->z+mat->_34*v1->w;
		out->w=mat->_41*v1->x+mat->_42*v1->y+mat->_43*v1->z+mat->_44*v1->w;
		out->x = X;
		out->y = Y;
		out->z = Z;
	}
	else{
		out->x=mat->_11*v1->x+mat->_12*v1->y+mat->_13*v1->z+mat->_14*v1->w;
		out->y=mat->_21*v1->x+mat->_22*v1->y+mat->_23*v1->z+mat->_24*v1->w;
		out->z=mat->_31*v1->x+mat->_32*v1->y+mat->_33*v1->z+mat->_34*v1->w;
		out->w=mat->_41*v1->x+mat->_42*v1->y+mat->_43*v1->z+mat->_44*v1->w;
	}
#elif defined NEON_MATH
	
	HQNeonMatrix4MultiplyVector4(mat->m, v1->v , out->v);
	
#elif defined HQ_DIRECTX_MATH

	HQDXMatrix4MultiplyVector4(mat->m, v1->v , out->v);

#else
	float4 m0,m1,m2,m3,m4,m5,m6,m7;

	m0=_mm_load_ps(mat->m); 
	m1=_mm_load_ps(mat->m+4);
	m2=_mm_load_ps(mat->m+8);
	m3=_mm_load_ps(mat->m+12);

	m5=_mm_shuffle_ps(m0,m1,SSE_SHUFFLEL(0,1,0,1));
	m6=_mm_shuffle_ps(m2,m3,SSE_SHUFFLEL(0,1,0,1));
	m4=_mm_shuffle_ps(m5,m6,SSE_SHUFFLEL(0,2,0,2));
	m5=_mm_shuffle_ps(m5,m6,SSE_SHUFFLEL(1,3,1,3));

	m7=_mm_shuffle_ps(m0,m1,SSE_SHUFFLEL(2,3,2,3));
	m0=_mm_shuffle_ps(m2,m3,SSE_SHUFFLEL(2,3,2,3));
	m6=_mm_shuffle_ps(m7,m0,SSE_SHUFFLEL(0,2,0,2));
	m7=_mm_shuffle_ps(m7,m0,SSE_SHUFFLEL(1,3,1,3));

	m0=_mm_load_ps(v1->v);
	m1=m0;
	m2=m0;
	m3=m0;

	m0=_mm_shuffle_ps(m0,m0,0x00);
	m1=_mm_shuffle_ps(m1,m1,0x55);
	m2=_mm_shuffle_ps(m2,m2,0xaa);
	m3=_mm_shuffle_ps(m3,m3,0xff);

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	m3=_mm_mul_ps(m3,m7);

	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(out->v,m0);
#endif
	return out;
}
//***********************************************
//ma trận quay quanh Ox
//***********************************************
HQMatrix4& HQMatrix4::RotateX(hq_float32 angle){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	_11=_44=1.0f;
	_12=_13=_14=_21=_24=_31=_34=_41=_42=_43=0.0f;
	_22=fCos;
	_23=fSin;
	_32=-fSin;
	_33=fCos;
	return *this;
}
HQMatrix4* HQMatrix4rRotateX(hq_float32 angle, HQMatrix4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_11=out->_44=1.0f;
	out->_12=out->_13=out->_14=out->_21=out->_24
		=out->_31=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_22=fCos;
	out->_23=fSin;
	out->_32=-fSin;
	out->_33=fCos;
	return out;
}
HQMatrix4* HQMatrix4cRotateX(hq_float32 angle, HQMatrix4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_11=out->_44=1.0f;
	out->_12=out->_13=out->_14=out->_21=out->_24
		=out->_31=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_22=fCos;
	out->_32=fSin;
	out->_23=-fSin;
	out->_33=fCos;
	return out;
}
//***********************************************
//ma trận quay quanh Oy
//***********************************************
HQMatrix4& HQMatrix4::RotateY(hq_float32 angle){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	_22=_44=1.0f;
	_12=_14=_21=_23=_24=_32=_34=_41=_42=_43=0.0f;
	_11=fCos;
	_13=-fSin;
	_31=fSin;
	_33=fCos;
	return *this;
}
HQMatrix4* HQMatrix4rRotateY(hq_float32 angle, HQMatrix4 *out ){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_22=out->_44=1.0f;
	out->_12=out->_14=out->_21=out->_23
		=out->_24=out->_32=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_11=fCos;
	out->_13=-fSin;
	out->_31=fSin;
	out->_33=fCos;
	return out;
}
HQMatrix4* HQMatrix4cRotateY(hq_float32 angle, HQMatrix4 *out ){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_22=out->_44=1.0f;
	out->_12=out->_14=out->_21=out->_23
		=out->_24=out->_32=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_11=fCos;
	out->_31=-fSin;
	out->_13=fSin;
	out->_33=fCos;
	return out;
}
//***********************************************
//ma trận quay quanh Oz
//***********************************************
HQMatrix4& HQMatrix4::RotateZ(hq_float32 angle){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	_33=_44=1.0f;
	_13=_14=_23=_24=_31=_32=_34=_41=_42=_43=0.0f;
	_11=fCos;
	_12=fSin;
	_21=-fSin;
	_22=fCos;
	return *this;
}
HQMatrix4* HQMatrix4rRotateZ(hq_float32 angle, HQMatrix4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_33=out->_44=1.0f;
	out->_13=out->_14=out->_23=out->_24=out->_31
		=out->_32=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_11=fCos;
	out->_12=fSin;
	out->_21=-fSin;
	out->_22=fCos;
	return out;
}
HQMatrix4* HQMatrix4cRotateZ(hq_float32 angle, HQMatrix4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_33=out->_44=1.0f;
	out->_13=out->_14=out->_23=out->_24=out->_31
		=out->_32=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_11=fCos;
	out->_21=fSin;
	out->_12=-fSin;
	out->_22=fCos;
	return out;
}
//************************************************
//ma trận tịnh tiến
//************************************************
HQMatrix4& HQMatrix4::Translate(hq_float32 x, hq_float32 y, hq_float32 z){
	_44=_11=_22=_33=1.0f;
	_12=_13=_21=_23=_31=_32=0.0f;
	_41=x;_42=y;_43=z;
	_14=_24=_34=0.0f;
	return *this;
}
HQMatrix4* HQMatrix4rTranslate(hq_float32 x, hq_float32 y, hq_float32 z, HQMatrix4 *out){
	out->_44=out->_11=out->_22=out->_33=1.0f;
	out->_12=out->_13=out->_21=out->_23=out->_31=out->_32=0.0f;
	out->_41=x;out->_42=y;out->_43=z;
	out->_14=out->_24=out->_34=0.0f;
	return out;
}
HQMatrix4* HQMatrix4cTranslate(hq_float32 x, hq_float32 y, hq_float32 z, HQMatrix4 *out){
	out->_44=out->_11=out->_22=out->_33=1.0f;
	out->_12=out->_13=out->_21=out->_23=out->_31=out->_32=0.0f;
	out->_14=x;out->_24=y;out->_34=z;
	out->_41=out->_42=out->_43=0.0f;
	return out;
}
//******************************************************
//ma trận tỷ lệ
//******************************************************
HQMatrix4& HQMatrix4::Scale(hq_float32 sx, hq_float32 sy, hq_float32 sz){
	_11=sx;_22=sy;_33=sz;
	_12=_13=_14=_21=_23=_24=_31=_32=_34=_41=_42=_43=0.0f;
	_44=1.0f;
	return *this;
}
HQMatrix4& HQMatrix4::Scale(hq_float32 s[3]){
	_11=s[0];_22=s[1];_33=s[2];
	_12=_13=_14=_21=_23=_24=_31=_32=_34=_41=_42=_43=0.0f;
	_44=1.0f;
	return *this;
}

HQMatrix4* HQMatrix4Scale(hq_float32 sx, hq_float32 sy, hq_float32 sz, HQMatrix4 *out){
	out->_11=sx;out->_22=sy;out->_33=sz;
	out->_12=out->_13=out->_14=out->_21=out->_23=out->_24=out->_31=
		out->_32=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_44=1.0f;
	return out;
}

HQMatrix4* HQMatrix4Scale(hq_float32 s[3], HQMatrix4 *out){
	out->_11=s[0];out->_22=s[1];out->_33=s[2];
	out->_12=out->_13=out->_14=out->_21=out->_23=out->_24=out->_31=
		out->_32=out->_34=out->_41=out->_42=out->_43=0.0f;
	out->_44=1.0f;
	return out;
}

//******************************************************
//nhân 2 ma trận
//******************************************************
HQMatrix4& HQMatrix4::operator *=(const HQMatrix4 &m){
	
#ifdef CMATH
	HQMatrix4 result( 0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f );
	for (hq_int32 i = 0; i < 4 ; ++i)
		for(hq_int32 j = 0; j < 4 ; ++j)
			for( hq_int32 k = 0 ; k < 4 ; ++k)
				result.mt[i][j] += mt[i][k] * m.mt[k][j];
	memcpy(this, &result, sizeof(HQMatrix4));
#elif defined NEON_MATH

	HQNeonMatrix4Multiply(this->m, m, this->m);

#elif defined HQ_DIRECTX_MATH

	HQDXMatrix4Multiply(this->m, m, this->m);

#else
	float4 xmm[4],re,row,e;

	//load 4 hàng của matix m
	xmm[0]=_mm_load_ps(&m.m[0]);
	xmm[1]=_mm_load_ps(&m.m[4]);
	xmm[2]=_mm_load_ps(&m.m[8]);
	xmm[3]=_mm_load_ps(&m.m[12]);
	
	//first row
	row=_mm_load_ps(&this->m[0]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&this->m[0],re);

	//second row
	row=_mm_load_ps(&this->m[4]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&this->m[4],re);

	//third row
	row=_mm_load_ps(&this->m[8]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&this->m[8],re);

	//fourth row
	row=_mm_load_ps(&this->m[12]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&this->m[12],re);
	
#endif

	return *this;
}

#ifndef HQ_EXPLICIT_ALIGN

HQMatrix4 HQMatrix4::operator *(const HQMatrix4 &m) const{

	
#ifdef CMATH
	HQMatrix4  result( 0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f );

	for (hq_int32 i = 0; i < 4 ; ++i)
		for(hq_int32 j = 0; j < 4 ; ++j)
			for( hq_int32 k = 0 ; k < 4 ; ++k)
				result.mt[i][j] += mt[i][k] * m.mt[k][j];

#elif defined NEON_MATH
	HQA16ByteMatrix4Ptr pM(NULL);
	HQMatrix4  &result = *pM;

	HQNeonMatrix4Multiply(this->m, m, result);

#elif defined HQ_DIRECTX_MATH

	HQMatrix4  result;

	HQDXMatrix4Multiply(this->m, m, result);

#else

	HQMatrix4  result(NULL);

	float4 xmm[4],re,row,e;

	//load 4 hàng của matix m
	xmm[0]=_mm_load_ps(&m.m[0]);
	xmm[1]=_mm_load_ps(&m.m[4]);
	xmm[2]=_mm_load_ps(&m.m[8]);
	xmm[3]=_mm_load_ps(&m.m[12]);
	
	//first row
	row=_mm_load_ps(&this->m[0]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&result.m[0],re);

	//second row
	row=_mm_load_ps(&this->m[4]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&result.m[4],re);

	//third row
	row=_mm_load_ps(&this->m[8]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&result.m[8],re);

	//fourth row
	row=_mm_load_ps(&this->m[12]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&result.m[12],re);
#endif
	return result;
}

#endif//#ifndef HQ_EXPLICIT_ALIGN

HQMatrix4* HQMatrix4Multiply(const HQMatrix4* pM1,const HQMatrix4* pM2,HQMatrix4* pOut){
#ifdef CMATH
	HQMatrix4 result( 0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f ,
					0.0f , 0.0f , 0.0f , 0.0f );
	for (hq_int32 i = 0; i < 4 ; ++i)
		for(hq_int32 j = 0; j < 4 ; ++j)
			for( hq_int32 k = 0 ; k < 4 ; ++k)
				result.mt[i][j] += pM1->mt[i][k] * pM2->mt[k][j];
	memcpy(pOut, &result, sizeof(HQMatrix4));
	
#elif defined NEON_MATH
	
	HQNeonMatrix4Multiply(pM1->m, pM2->m, pOut->m);

#elif defined HQ_DIRECTX_MATH

	HQDXMatrix4Multiply(pM1->m, pM2->m, pOut->m);

#else
	float4 xmm[4],re,row,e;

	//load 4 hàng của matix 2
	xmm[0]=_mm_load_ps(&pM2->m[0]);
	xmm[1]=_mm_load_ps(&pM2->m[4]);
	xmm[2]=_mm_load_ps(&pM2->m[8]);
	xmm[3]=_mm_load_ps(&pM2->m[12]);

	//first row
	row=_mm_load_ps(&pM1->m[0]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&pOut->m[0],re);

	//second row
	row=_mm_load_ps(&pM1->m[4]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&pOut->m[4],re);

	//third row
	row=_mm_load_ps(&pM1->m[8]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&pOut->m[8],re);

	//fourth row
	row=_mm_load_ps(&pM1->m[12]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[3]));

	_mm_store_ps(&pOut->m[12],re);
#endif
	return pOut;
}

HQMatrix4* HQMatrix4Multiply(const HQMatrix4* pM1,const HQMatrix3x4* pM2,HQMatrix4* pOut){
#ifdef CMATH
	HQMatrix4 result( 0.0f , 0.0f , 0.0f , pM1->_14 ,
					0.0f , 0.0f , 0.0f , pM1->_24 ,
					0.0f , 0.0f , 0.0f , pM1->_34 ,
					0.0f , 0.0f , 0.0f , pM1->_44 );
	for (hq_int32 i = 0; i < 4 ; ++i)
		for(hq_int32 j = 0; j < 4 ; ++j)
			for( hq_int32 k = 0 ; k < 3 ; ++k)
				result.mt[i][j] += pM1->mt[i][k] * pM2->mt[k][j];

	memcpy(pOut, &result, sizeof(HQMatrix4));
	
#elif defined NEON_MATH
	
	HQNeonMatrix4MultiplyMatrix3x4(pM1->m, pM2->m, pOut->m);

#elif defined HQ_DIRECTX_MATH

	HQDXMatrix4MultiplyMatrix3x4(pM1->m, pM2->m, pOut->m);

#else
	static const HQ_ALIGN16 hq_uint32 _3Zeros_1One_Masks[4]={0x00000000,0x00000000,0x00000000,0xffffffff};

	float4 xmm[3],re,row , e,masks;
	
	masks = _mm_load_ps((hq_float32*)_3Zeros_1One_Masks);
	//load 3 hàng của matix 2
	xmm[0]=_mm_load_ps(&pM2->m[0]);
	xmm[1]=_mm_load_ps(&pM2->m[4]);
	xmm[2]=_mm_load_ps(&pM2->m[8]);
	
	//first row
	row=_mm_load_ps(pM1->m);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	row = _mm_and_ps(row , masks);// 0 0 0 pM1->m[3]
	re = _mm_add_ps(re , row);

	_mm_store_ps(pOut->m,re);

	//second row
	row=_mm_load_ps(&pM1->m[4]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	row = _mm_and_ps(row , masks);// 0 0 0 pM1->m[7]
	re = _mm_add_ps(re , row);

	_mm_store_ps(&pOut->m[4],re);

	//third row
	row=_mm_load_ps(&pM1->m[8]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	row = _mm_and_ps(row , masks);// 0 0 0 pM1->m[11]
	re = _mm_add_ps(re , row);

	_mm_store_ps(&pOut->m[8],re);

	//fourth row
	row=_mm_load_ps(&pM1->m[12]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);
	
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));

	row = _mm_and_ps(row , masks);// 0 0 0 pM1->m[15]
	re = _mm_add_ps(re , row);

	_mm_store_ps(&pOut->m[12],re);
#endif
	return pOut;
}

HQ_UTIL_MATH_API HQMatrix4* HQMatrix4MultiMultiply(const HQMatrix4* pM, hq_uint32 numMatrices ,HQMatrix4* pOut)
{
	if (numMatrices >= 2)
	{
#ifdef CMATH
		if (pOut != &pM[0])
			*pOut = pM[0];
		for (hq_uint32 i = 1 ; i < numMatrices ; ++i)
		{
			(*pOut) *= pM[i];
		}
#elif defined NEON_MATH
		
		HQNeonMultiMatrix4Multiply(pM->m, numMatrices, pOut->m);

#elif defined HQ_DIRECTX_MATH

		HQDXMultiMatrix4Multiply(pM->m, numMatrices, pOut->m);

#else/*SSE*/
		float4 xmm[4] , reRow[4] , row , e;

		//load 4 hàng của matix 0
		reRow[0]=_mm_load_ps(&pM[0].m[0]);
		reRow[1]=_mm_load_ps(&pM[0].m[4]);
		reRow[2]=_mm_load_ps(&pM[0].m[8]);
		reRow[3]=_mm_load_ps(&pM[0].m[12]);

		for (hq_uint32 i = 1 ; i < numMatrices ; ++i)
		{
			//load 4 hàng của matix i
			xmm[0]=_mm_load_ps(&pM[i].m[0]);
			xmm[1]=_mm_load_ps(&pM[i].m[4]);
			xmm[2]=_mm_load_ps(&pM[i].m[8]);
			xmm[3]=_mm_load_ps(&pM[i].m[12]);

			//first row
			row = reRow[0];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[0]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[0]=_mm_add_ps(reRow[0],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[0]=_mm_add_ps(reRow[0],_mm_mul_ps(e,xmm[2]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
			reRow[0]=_mm_add_ps(reRow[0],_mm_mul_ps(e,xmm[3]));


			//second row
			row = reRow[1];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[1]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[1]=_mm_add_ps(reRow[1],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[1]=_mm_add_ps(reRow[1],_mm_mul_ps(e,xmm[2]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
			reRow[1]=_mm_add_ps(reRow[1],_mm_mul_ps(e,xmm[3]));

			//third row
			row = reRow[2];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[2]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[2]=_mm_add_ps(reRow[2],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[2]=_mm_add_ps(reRow[2],_mm_mul_ps(e,xmm[2]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
			reRow[2]=_mm_add_ps(reRow[2],_mm_mul_ps(e,xmm[3]));

			//fourth row
			row = reRow[3];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[3]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[3]=_mm_add_ps(reRow[3],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[3]=_mm_add_ps(reRow[3],_mm_mul_ps(e,xmm[2]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(3,3,3,3));
			reRow[3]=_mm_add_ps(reRow[3],_mm_mul_ps(e,xmm[3]));
		}

		_mm_store_ps(pOut->m , reRow[0]);
		_mm_store_ps(pOut->m + 4, reRow[1]);
		_mm_store_ps(pOut->m + 8, reRow[2]);
		_mm_store_ps(pOut->m + 12, reRow[3]);
#endif
	}
	return pOut;
}

//**********************************************************
//ma trận quay quanh trục bất kỳ
//**********************************************************
HQMatrix4& HQMatrix4::RotateAxis(HQVector4 &axis, hq_float32 angle){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	
	hq_float32 fSum = 1.0f - fCos;
	axis.Normalize();

	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	_14 = _24 = _34 = _41 = _42 = _43 = 0.0f;
	_44 = 1.0f;
	_11 = sqr(axis.x) * fSum + fCos;
	_22 = (sqr(axis.y)) * fSum + fCos ;
	_33 = (sqr(axis.z)) * fSum + fCos;
	_21 = (fXY) * fSum - (axis.z*fSin);
	_31 = (fXZ) * fSum + (axis.y*fSin);
	_12 = (fXY) * fSum + (axis.z*fSin);
	_32 = (fYZ) * fSum - (axis.x*fSin);
	_13 = (fXZ) * fSum - (axis.y*fSin);
	_23 = (fYZ) * fSum + (axis.x*fSin);
	return *this;
}
HQMatrix4& HQMatrix4::RotateAxisUnit(const HQVector4 &axis, hq_float32 angle){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	_14 = _24 = _34 = _41 = _42 = _43 = 0.0f;
	_44 = 1.0f;
	_11 = sqr(axis.x) * fSum + fCos;
	_22 = (sqr(axis.y)) * fSum + fCos ;
	_33 = (sqr(axis.z)) * fSum + fCos;
	_21 = (fXY) * fSum - (axis.z*fSin);
	_31 = (fXZ) * fSum + (axis.y*fSin);
	_12 = (fXY) * fSum + (axis.z*fSin);
	_32 = (fYZ) * fSum - (axis.x*fSin);
	_13 = (fXZ) * fSum - (axis.y*fSin);
	_23 = (fYZ) * fSum + (axis.x*fSin);
	return *this;
}
HQMatrix4* HQMatrix4rRotateAxis(HQVector4 &axis, hq_float32 angle, HQMatrix4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	axis.Normalize();

	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	out->_14 = out->_24 = out->_34 = out->_41 = out->_42 = out->_43 = 0.0f;
	out->_44 = 1.0f;
	out->_11 = sqr(axis.x) * fSum + fCos;
	out->_22 = (sqr(axis.y)) * fSum + fCos ;
	out->_33 = (sqr(axis.z)) * fSum + fCos;
	out->_21 = (fXY) * fSum - (axis.z*fSin);
	out->_31 = (fXZ) * fSum + (axis.y*fSin);
	out->_12 = (fXY) * fSum + (axis.z*fSin);
	out->_32 = (fYZ) * fSum - (axis.x*fSin);
	out->_13 = (fXZ) * fSum - (axis.y*fSin);
	out->_23 = (fYZ) * fSum + (axis.x*fSin);
	return out;
}

HQMatrix4* HQMatrix4cRotateAxis(HQVector4 &axis, hq_float32 angle, HQMatrix4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	axis.Normalize();

	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	out->_14 = out->_24 = out->_34 = out->_41 = out->_42 = out->_43 = 0.0f;
	out->_44 = 1.0f;
	out->_11 = sqr(axis.x) * fSum + fCos;
	out->_22 = (sqr(axis.y)) * fSum + fCos ;
	out->_33 = (sqr(axis.z)) * fSum + fCos;
	out->_12 = (fXY) * fSum - (axis.z*fSin);
	out->_13 = (fXZ) * fSum + (axis.y*fSin);
	out->_21 = (fXY) * fSum + (axis.z*fSin);
	out->_23 = (fYZ) * fSum - (axis.x*fSin);
	out->_31 = (fXZ) * fSum - (axis.y*fSin);
	out->_32 = (fYZ) * fSum + (axis.x*fSin);
	return out;
}

HQMatrix4* HQMatrix4rRotateAxisUnit(const HQVector4 &axis, hq_float32 angle, HQMatrix4 *out ){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	out->_14 = out->_24 = out->_34 = out->_41 = out->_42 = out->_43 = 0.0f;
	out->_44 = 1.0f;
	out->_11 = sqr(axis.x) * fSum + fCos;
	out->_22 = (sqr(axis.y)) * fSum + fCos ;
	out->_33 = (sqr(axis.z)) * fSum + fCos;
	out->_21 = (fXY) * fSum - (axis.z*fSin);
	out->_31 = (fXZ) * fSum + (axis.y*fSin);
	out->_12 = (fXY) * fSum + (axis.z*fSin);
	out->_32 = (fYZ) * fSum - (axis.x*fSin);
	out->_13 = (fXZ) * fSum - (axis.y*fSin);
	out->_23 = (fYZ) * fSum + (axis.x*fSin);
	return out;
}

HQMatrix4* HQMatrix4cRotateAxisUnit(const HQVector4 &axis, hq_float32 angle, HQMatrix4 *out ){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	out->_14 = out->_24 = out->_34 = out->_41 = out->_42 = out->_43 = 0.0f;
	out->_44 = 1.0f;
	out->_11 = sqr(axis.x) * fSum + fCos;
	out->_22 = (sqr(axis.y)) * fSum + fCos ;
	out->_33 = (sqr(axis.z)) * fSum + fCos;
	out->_12 = (fXY) * fSum - (axis.z*fSin);
	out->_13 = (fXZ) * fSum + (axis.y*fSin);
	out->_21 = (fXY) * fSum + (axis.z*fSin);
	out->_23 = (fYZ) * fSum - (axis.x*fSin);
	out->_31 = (fXZ) * fSum - (axis.y*fSin);
	out->_32 = (fYZ) * fSum + (axis.x*fSin);
	return out;
}


//************************************************************
//ma trận nghịch đảo
//************************************************************
HQMatrix4& HQMatrix4::Inverse(){
	HQMatrix4Inverse(this,0,this);

	return *this;
}

HQMatrix4* HQMatrix4Inverse(const HQMatrix4* in,HQMatrix4* out){
	return HQMatrix4Inverse(in,0,out);
}

HQMatrix4* HQMatrix4Inverse(const HQMatrix4* pM,hq_float32* Determinant,HQMatrix4* pOut){
	
#ifdef CMATH
	HQMatrix4 trans;
	hq_float32     f[12],det;
	
	// transpose matrix
	HQMatrix4Transpose(pM, &trans);
	
	
	f[0]  = trans._33 * trans._44;
	f[1]  = trans._34 * trans._43;
	f[2]  = trans._32 * trans._44;
	f[3]  = trans._34 * trans._42;
	f[4]  = trans._32 * trans._43;
	f[5]  = trans._33 * trans._42;
	f[6]  = trans._31 * trans._44;
	f[7]  = trans._34 * trans._41;
	f[8]  = trans._31 * trans._43;
	f[9]  = trans._33 * trans._41;
	f[10]  = trans._31 * trans._42;
	f[11]  = trans._32 * trans._41;
	
	
	pOut->_11  = f[0]*trans._22 + f[3]*trans._23 + f[4] *trans._24;
	pOut->_11 -= f[1]*trans._22 + f[2]*trans._23 + f[5] *trans._24;
	pOut->_12  = f[1]*trans._21 + f[6]*trans._23 + f[9] *trans._24;
	pOut->_12 -= f[0]*trans._21 + f[7]*trans._23 + f[8] *trans._24;
	pOut->_13  = f[2]*trans._21 + f[7]*trans._22 + f[10]*trans._24;
	pOut->_13 -= f[3]*trans._21 + f[6]*trans._22 + f[11]*trans._24;
	pOut->_14  = f[5]*trans._21 + f[8]*trans._22 + f[11]*trans._23;
	pOut->_14 -= f[4]*trans._21 + f[9]*trans._22 + f[10]*trans._23;
	pOut->_21  = f[1]*trans._12 + f[2]*trans._13 + f[5] *trans._14;
	pOut->_21 -= f[0]*trans._12 + f[3]*trans._13 + f[4] *trans._14;
	pOut->_22  = f[0]*trans._11 + f[7]*trans._13 + f[8] *trans._14;
	pOut->_22 -= f[1]*trans._11 + f[6]*trans._13 + f[9] *trans._14;
	pOut->_23  = f[3]*trans._11 + f[6]*trans._12 + f[11]*trans._14;
	pOut->_23 -= f[2]*trans._11 + f[7]*trans._12 + f[10]*trans._14;
	pOut->_24  = f[4]*trans._11 + f[9]*trans._12 + f[10]*trans._13;
	pOut->_24 -= f[5]*trans._11 + f[8]*trans._12 + f[11]*trans._13;
	
	f[0]  = trans._13 * trans._24;
	f[1]  = trans._14 * trans._23;
	f[2]  = trans._12 * trans._24;
	f[3]  = trans._14 * trans._22;
	f[4]  = trans._12 * trans._23;
	f[5]  = trans._13 * trans._22;
	f[6]  = trans._11 * trans._24;
	f[7]  = trans._14 * trans._21;
	f[8]  = trans._11 * trans._23;
	f[9]  = trans._13 * trans._21;
	f[10]  = trans._11 * trans._22;
	f[11]  = trans._12 * trans._21;
	
	
	
	pOut->_31  = f[0] *trans._42 + f[3] *trans._43 + f[4] *trans._44;
	pOut->_31 -= f[1] *trans._42 + f[2] *trans._43 + f[5] *trans._44;
	pOut->_32  = f[1] *trans._41 + f[6] *trans._43 + f[9] *trans._44;
	pOut->_32 -= f[0] *trans._41 + f[7] *trans._43 + f[8] *trans._44;
	pOut->_33  = f[2] *trans._41 + f[7] *trans._42 + f[10]*trans._44;
	pOut->_33 -= f[3] *trans._41 + f[6] *trans._42 + f[11]*trans._44;
	pOut->_34  = f[5] *trans._41 + f[8] *trans._42 + f[11]*trans._43;
	pOut->_34 -= f[4] *trans._41 + f[9] *trans._42 + f[10]*trans._43;
	pOut->_41  = f[2] *trans._33 + f[5] *trans._34 + f[1] *trans._32;
	pOut->_41 -= f[4] *trans._34 + f[0] *trans._32 + f[3] *trans._33;
	pOut->_42  = f[8] *trans._34 + f[0] *trans._31 + f[7] *trans._33;
	pOut->_42 -= f[6] *trans._33 + f[9] *trans._34 + f[1] *trans._31;
	pOut->_43  = f[6] *trans._32 + f[11]*trans._34 + f[3] *trans._31;
	pOut->_43 -= f[10]*trans._34 + f[2] *trans._31 + f[7] *trans._32;
	pOut->_44  = f[10]*trans._33 + f[4] *trans._31 + f[9] *trans._32;
	pOut->_44 -= f[8] *trans._32 + f[11]*trans._33 + f[5] *trans._31;
	

	//định thức
	det = trans._11*pOut->_11 + 
	trans._12*pOut->_12 + 
	trans._13*pOut->_13 +
	trans._14*pOut->_14;
	
	if(Determinant)
		*Determinant = det;
	
	det = 1.0f/det;
	
	pOut->_11 *= det;  
	pOut->_12 *= det;  
	pOut->_13 *= det;  
	pOut->_14 *= det;
	
	pOut->_21 *= det;  
	pOut->_22 *= det;  
	pOut->_23 *= det;  
	pOut->_24 *= det;
	
	pOut->_31 *= det;  
	pOut->_32 *= det;  
	pOut->_33 *= det;  
	pOut->_34 *= det;
	
	pOut->_41 *= det;  
	pOut->_42 *= det;  
	pOut->_43 *= det;  
	pOut->_44 *= det;

#elif defined NEON_MATH
	
	HQNeonMatrix4Inverse(pM->m, pOut->m, Determinant);

#elif defined HQ_DIRECTX_MATH

	HQDXMatrix4Inverse(pM->m, pOut->m, Determinant);

#else /*SSE version*/
	float4 temp1;
	float4 det;
	float4 f0 , f1 , f2;
	float4 m0 , m1 , m2 , m3;

	m0 = _mm_load_ps(pM->m);//row0		m11 m12 m13 m14
	m1 = _mm_load_ps(pM->m + 4);//row1  m21 m22 m23 m24
	m2 = _mm_load_ps(pM->m + 8);//row2  m31 m32 m33 m34
	m3 = _mm_load_ps(pM->m + 12);//row3 m41 m42 m43 m44

	float4 s0 = _mm_unpackhi_ps(m3 , m2);//m43 m33 m44 m34
	float4 s1 = _mm_unpackhi_ps(m1 , m0);//m23 m13 m24 m14
	float4 s2 = _mm_unpacklo_ps(m2 , m3);//m31 m41 m32 m42
	float4 s3 = _mm_unpacklo_ps(m0 , m1);//m11 m21 m12 m22
	
	m0 = _mm_movelh_ps(s2 , s3);//m31 m41 m11 m21==>13  14 11 12  	
	m1 = _mm_movehl_ps(s3 , s2);//m32 m42 m12 m22==>23  24 21 22
	m2 = _mm_movelh_ps(s0 , s1);//m43 m33 m23 m13==>34  33 32 31
	m3 = _mm_movehl_ps(s1 , s0);//m44 m34 m24 m14==>44  43 42 41 
	f1 = hq_mm_copy_ps(m2 , SSE_SHUFFLEL(3 , 2 , 1 , 0));//31 32 33 34
	f2 = hq_mm_copy_ps(m2 , SSE_SHUFFLEL(1 , 0 , 3 , 2));//33 34 31 32
	f0 = hq_mm_copy_ps(m3 , SSE_SHUFFLEL(2 , 3 , 0 , 1));//42 41 44 43
	
	f0 = _mm_mul_ps(f0 , m2);//34.42 33.41 32.44 31.43
	f1 = _mm_mul_ps(f1 , m3);//31.44 32.43 33.42 34.41
	f2 = _mm_mul_ps(f2 , m3);//33.44 34.43 31.42 32.42

	/*------------matrix of cofactors--------------*/
	//first 2 rows
	s0 = _mm_mul_ps(f0 , m1);
	s3 = _mm_mul_ps(f0 , m0);
	f0 = _mm_shuffle_ps(f0 , f0 , SSE_SHUFFLEL(2,3,0,1));//32.44 31.43 34.42 33.41
	s2 = _mm_mul_ps(f0 , m1);
	s1 = _mm_mul_ps(f0 , m0);
	s0 = _mm_sub_ps(s0 , s2);
	s1 = _mm_sub_ps(s1 , s3);

	s2 = _mm_mul_ps(f1 , m1);
	s3 = _mm_mul_ps(f1 , m0);
	f1 = _mm_shuffle_ps(f1 , f1 , SSE_SHUFFLEL(3 , 2 , 1 , 0));//34.41 33.42 32.43 31.44  
	f0 = _mm_mul_ps(f1 , m1);
	f1 = _mm_mul_ps(f1 , m0);
	s2 = _mm_sub_ps(s2 , f0);
	f1 = _mm_sub_ps(f1 , s3);
	s2 = _mm_shuffle_ps(s2 , s2 , SSE_SHUFFLEL(1 , 0 , 3 , 2));
	f1 = _mm_shuffle_ps(f1 , f1 , SSE_SHUFFLEL(1 , 0 , 3 , 2));
	s0 = _mm_add_ps(s0 , s2);
	s1 = _mm_add_ps(s1 , f1);
	
	temp1 = hq_mm_copy_ps(m1 , SSE_SHUFFLEL(3 , 2 , 1 , 0));//22 21 24 23
	f0 = hq_mm_copy_ps(m0 , SSE_SHUFFLEL(3 , 2 , 1 , 0));//12 11 14 13
	s2 = _mm_mul_ps(temp1 , f2);
	s3 = _mm_mul_ps(f0 , f2);
	f2 = _mm_shuffle_ps(f2 , f2 , SSE_SHUFFLEL(1 , 0 , 3 , 2));//34.43 33.44 32.42 31.42
	f1 = _mm_mul_ps(temp1 , f2);
	f2 = _mm_mul_ps(f2 , f0);
	s2 = _mm_sub_ps(s2 , f1);
	f2 = _mm_sub_ps(f2 , s3);
	s0 = _mm_add_ps(s0 , s2);
	s1 = _mm_add_ps(s1 , f2);
	
	//last 2 rows

	det = hq_mm_copy_ps(f0 , SSE_SHUFFLEL(1 , 0 , 3 , 2));//det = 11 12 13 14
	f1 = hq_mm_copy_ps(temp1 , SSE_SHUFFLEL(1, 0 , 3 , 2));//21 22 23 24
	f0 = _mm_mul_ps(f0 , m1);//12.23 11.24 14.21  13.22
	f1 = _mm_mul_ps(f1 , m0);//13.21 14.22  11.23 12.24
	f2 = _mm_mul_ps(det , temp1);//11.22 12.21 13.24 14.23 
	
	s2 = _mm_mul_ps(m3 , f0);
	m1 = _mm_mul_ps(m2 , f0);
	f0 = _mm_shuffle_ps(f0 , f0 , SSE_SHUFFLEL(3 , 2 , 1 , 0));//13.22 14.21 11.24 12.23
	m0 = _mm_mul_ps(m3 , f0);
	s3 = _mm_mul_ps(m2 , f0);
	s2 = _mm_sub_ps(s2 , m0);
	s3 = _mm_sub_ps(s3 , m1);

	m0 = _mm_mul_ps(m3 , f1);
	m1 = _mm_mul_ps(m2 , f1);
	f1 = _mm_shuffle_ps(f1 , f1 , SSE_SHUFFLEL(2 , 3 , 0 , 1));//11.23 12.24 13.21 14.22
	f0 = _mm_mul_ps(m3 , f1);
	f1 = _mm_mul_ps(f1 , m2);
	m0 = _mm_sub_ps(m0 , f0);
	f1 = _mm_sub_ps(f1 , m1);
	m0 = _mm_shuffle_ps(m0 , m0 , SSE_SHUFFLEL(1 , 0 , 3 , 2));
	f1 = _mm_shuffle_ps(f1 , f1 , SSE_SHUFFLEL(1 , 0 , 3 , 2));
	s2 = _mm_add_ps(s2 , m0);
	s3 = _mm_add_ps(s3 , f1);

	m0 = _mm_mul_ps(m3 , f2);
	f1 = _mm_mul_ps(m2 , f2);
	f2 = _mm_shuffle_ps(f2 , f2 , SSE_SHUFFLEL(1 , 0 , 3 , 2));//12.21 11.22 14.23  13.24
	f0 = _mm_mul_ps(m3 , f2);
	m1 = _mm_mul_ps(m2 , f2);
	m0 = _mm_sub_ps(m0 , f0);
	m1 = _mm_sub_ps(m1 , f1);
	m0 = _mm_shuffle_ps(m0 , m0 , SSE_SHUFFLEL(2 , 3 , 0 , 1));
	m1 = _mm_shuffle_ps(m1 , m1 , SSE_SHUFFLEL(2 , 3 , 0 , 1));
	s2 = _mm_add_ps(s2 , m0);
	s3 = _mm_add_ps(s3 , m1);
	
	//determinant
	det = _mm_mul_ps(det , s0);
	det = _mm_add_ps(det , hq_mm_copy_ps(det , SSE_SHUFFLEL(1 , 0 , 3 , 2)));
	det = _mm_add_ps(det , hq_mm_copy_ps(det , SSE_SHUFFLEL(2 , 2 , 0 , 0)));
	
	if (Determinant != NULL)
		_mm_store_ss(Determinant , det); 

	//temp1=_mm_set_ps1(1.0f);
	//det=_mm_div_ps(temp1,det);
	
	temp1 = _mm_rcp_ps(det);//tính gần đúng 1/det
	//Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 - giảm sai số
	//Y = 1/x  
	//x = det ; Y(n) = temp1
	det  = _mm_sub_ps(_mm_add_ps(temp1, temp1), _mm_mul_ps(det, _mm_mul_ps(temp1, temp1)));
	
	//multiply with 1/det, store result
	s0 = _mm_mul_ps(s0 , det);
	s1 = _mm_mul_ps(s1 , det);
	s2 = _mm_mul_ps(s2 , det);
	s3 = _mm_mul_ps(s3 , det);

	_mm_store_ps(pOut->m , s0);
	_mm_store_ps(pOut->m + 4, s1);
	_mm_store_ps(pOut->m + 8, s2);
	_mm_store_ps(pOut->m + 12, s3);
	
#endif
	return pOut;

}
//*************************************
//tạo ma trận view
//*************************************
HQMatrix4* HQMatrix4rView(const HQVector4* pX,const HQVector4* pY,const HQVector4* pZ,const HQVector4* pPos,HQMatrix4 *out)
{
	out->_44=1.0f;
	out->_11=pX->x;
	out->_22=pY->y;
	out->_33=pZ->z;
	out->_12=pY->x;
	out->_13=pZ->x;
	out->_14=0.0f;

	out->_21=pX->y;
	out->_23=pZ->y;
	out->_24=0.0f;

	out->_31=pX->z;
	out->_32=pY->z;
	out->_34=0.0f;

	out->_41=-((*pX)*(*pPos));
	out->_42=-((*pY)*(*pPos));
	out->_43=-((*pZ)*(*pPos));
	return out;
}

HQMatrix4* HQMatrix4cView(const HQVector4* pX,const HQVector4* pY,const HQVector4* pZ,const HQVector4* pPos,HQMatrix4 *out)
{
	out->_44=1.0f;
	out->_11=pX->x;
	out->_22=pY->y;
	out->_33=pZ->z;
	out->_21=pY->x;
	out->_31=pZ->x;
	out->_41=0.0f;

	out->_12=pX->y;
	out->_32=pZ->y;
	out->_42=0.0f;

	out->_13=pX->z;
	out->_23=pY->z;
	out->_43=0.0f;

	out->_14=-((*pX)*(*pPos));
	out->_24=-((*pY)*(*pPos));
	out->_34=-((*pZ)*(*pPos));
	return out;
}


HQMatrix4* HQMatrix4rLookAtLH(const HQVector4 *pEye,const HQVector4* pAt,const HQVector4* pUp,HQMatrix4*out)
{
	HQ_DECL_STACK_3VECTOR4( Xaxis,Yaxis,Zaxis);
	HQVector4Sub(pAt, pEye, &Zaxis);

	Zaxis.Normalize();

	Xaxis.Cross(*pUp,Zaxis);
	Xaxis.Normalize();

	Yaxis.Cross(Zaxis,Xaxis);

	return HQMatrix4rView(&Xaxis,&Yaxis,&Zaxis,pEye,out);
}

HQMatrix4* HQMatrix4cLookAtLH(const HQVector4 *pEye,const HQVector4* pAt,const HQVector4* pUp,HQMatrix4*out)
{
	HQ_DECL_STACK_3VECTOR4( Xaxis,Yaxis,Zaxis);
	HQVector4Sub(pAt, pEye, &Zaxis);

	Zaxis.Normalize();

	Xaxis.Cross(*pUp,Zaxis);
	Xaxis.Normalize();

	Yaxis.Cross(Zaxis,Xaxis);

	return HQMatrix4cView(&Xaxis,&Yaxis,&Zaxis,pEye,out);
}

HQMatrix4* HQMatrix4rLookAtRH(const HQVector4 *pEye,const HQVector4* pAt,const HQVector4* pUp,HQMatrix4*out)
{
	HQ_DECL_STACK_3VECTOR4( Xaxis,Yaxis,Zaxis);
	HQVector4Sub(pEye, pAt, &Zaxis);

	Zaxis.Normalize();

	Xaxis.Cross(*pUp,Zaxis);
	Xaxis.Normalize();

	Yaxis.Cross(Zaxis,Xaxis);

	return HQMatrix4rView(&Xaxis,&Yaxis,&Zaxis,pEye,out);
}

HQMatrix4* HQMatrix4cLookAtRH(const HQVector4 *pEye,const HQVector4* pAt,const HQVector4* pUp,HQMatrix4*out)
{
	HQ_DECL_STACK_3VECTOR4( Xaxis,Yaxis,Zaxis);
	HQVector4Sub(pEye, pAt, &Zaxis);

	Zaxis.Normalize();

	Xaxis.Cross(*pUp,Zaxis);
	Xaxis.Normalize();

	Yaxis.Cross(Zaxis,Xaxis);

	return HQMatrix4cView(&Xaxis,&Yaxis,&Zaxis,pEye,out);
}

//*************************************
//tạo ma trận chiếu trực giao
//*************************************
HQMatrix4* HQMatrix4rOrthoProjLH(const hq_float32 width,const hq_float32 height,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	out->_44=1.0f;
	out->_11=2.0f/width;
	out->_22=2.0f/height;
	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=1.0f/(zFar-zNear);
		out->_43=-zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zNear-zFar);
			out->_33=(-2.0f)*f;
			out->_43=(zNear+zFar)*f;
		}
		break;
	}
	return out;
}

HQMatrix4* HQMatrix4cOrthoProjLH(const hq_float32 width,const hq_float32 height,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	out->_44=1.0f;
	out->_11=2.0f/width;
	out->_22=2.0f/height;
	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=1.0f/(zFar-zNear);
		out->_34=-zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zNear-zFar);
			out->_33=(-2.0f)*f;
			out->_34=(zNear+zFar)*f;
		}
		break;
	}
	return out;
}

HQMatrix4* HQMatrix4rOrthoProjRH(const hq_float32 width,const hq_float32 height,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	out->_44=1.0f;
	out->_11=2.0f/width;
	out->_22=2.0f/height;
	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=1.0f/(zNear-zFar);
		out->_43=zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zNear-zFar);
			out->_33=2.0f*f;
			out->_43=(zNear+zFar)*f;
		}
		break;
	}
	return out;
}

HQMatrix4* HQMatrix4cOrthoProjRH(const hq_float32 width,const hq_float32 height,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	out->_44=1.0f;
	out->_11=2.0f/width;
	out->_22=2.0f/height;
	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=1.0f/(zNear-zFar);
		out->_34=zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zNear-zFar);
			out->_33=2.0f*f;
			out->_34=(zNear+zFar)*f;
		}
		break;
	}
	return out;
}

//*************************************
//tạo ma trận chiếu phối cảnh
//*************************************
HQMatrix4* HQMatrix4rPerspectiveProjLH(const hq_float32 vFOV,const hq_float32 aspect,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	hq_float32 sin,cos;
	HQSincosf(vFOV/2.0f,&sin,&cos);
	
	out->_22=cos/sin;
	out->_11=out->_22/aspect;
	out->_34=1.0f;

	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=zFar/(zFar-zNear);
		out->_43=-zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zFar-zNear);
			out->_33=(zNear+zFar)*f;
			out->_43=(-2)*zNear*zFar*f;
		}
		break;
	}
	return out;
}

HQMatrix4* HQMatrix4cPerspectiveProjLH(const hq_float32 vFOV,const hq_float32 aspect,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	hq_float32 sin,cos;
	HQSincosf(vFOV/2.0f,&sin,&cos);
	
	out->_22=cos/sin;
	out->_11=out->_22/aspect;
	out->_43=1.0f;

	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=zFar/(zFar-zNear);
		out->_34=-zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zFar-zNear);
			out->_33=(zNear+zFar)*f;
			out->_34=(-2)*zNear*zFar*f;
		}
		break;
	}
	return out;
}

HQMatrix4* HQMatrix4rPerspectiveProjRH(const hq_float32 vFOV,const hq_float32 aspect,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	hq_float32 sin,cos;
	HQSincosf(vFOV/2.0f,&sin,&cos);
	
	out->_22=cos/sin;
	out->_11=out->_22/aspect;
	out->_34=-1.0f;

	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=zFar/(zNear-zFar);
		out->_43=zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zNear-zFar);
			out->_33=(zNear+zFar)*f;
			out->_43=2*zNear*zFar*f;
		}
		break;
	}
	return out;
}

HQMatrix4* HQMatrix4cPerspectiveProjRH(const hq_float32 vFOV,const hq_float32 aspect,const hq_float32 zNear,const hq_float32 zFar,HQMatrix4*out,HQRenderAPI API)
{
	memset(out->m,0,sizeof(HQMatrix4));
	hq_float32 sin,cos;
	HQSincosf(vFOV/2.0f,&sin,&cos);
	
	out->_22=cos/sin;
	out->_11=out->_22/aspect;
	out->_43=-1.0f;

	switch(API)
	{
	case HQ_RA_D3D:
		out->_33=zFar/(zNear-zFar);
		out->_34=zNear*out->_33;
		break;
	case HQ_RA_OGL:
		{
			hq_float32 f=1.0f/(zNear-zFar);
			out->_33=(zNear+zFar)*f;
			out->_34=2*zNear*zFar*f;
		}
		break;
	}
	return out;
}

/*----------------------------------------
truy vấn mặt phẳng tạo thành thể tích nhìn
----------------------------------------*/

void HQMatrix4rGetFrustum(const HQMatrix4 * pViewProjMatrix , HQPlane planes[6] , HQRenderAPI API)
{
	
/*****************************************
(x1,y1,z1,1) là tọa độ một điểm  trong không gian thế giới
gọi (x/w , y/w , z/w, 1) là tọa độ điểm đó trong không gian chuẩn hóa của device
ma trận view * projection là M

(x,y,z,w)=(x1,y1,z1,1) * M 
=>x = M.11 * x1 + M.21 * y1 + M.31 * z1 + M.41
  w = M.14 * x1 + M.24 * y1 + M.34 * z1 + M.44

nếu điểm đó nằm trong thể tích nhìn thì trong không gian chuẩn hóa device:
    -1< (x/w) <1
=>-w < x < w
Xét giới hạn trái :-w < x
=>-(M.14 * x1 + M.24 * y1 + M.34 * z1 + M.44) < M.11 * x1 + M.21 * y1 + M.31 * z1 + M.41
=>-(M.14 + M.11) * x1 - (M.24 + M.21) * y1 -(M.34 + M.31) * z1 -(M.44 + M.41) < 0
=> vì trong không gian thế giới (x1,y1,z1,1) sẽ nằm trong thể tích nhìn tức là
sẽ nằm ngược phía với hướng của pháp vector mặt phẳng trái (giả sử pháp vector hướng ra ngoài
thể tích nhìn)
phương trình mặt trái là A*x + B*y + C*z + D=0
=>A*x1 + B*y1 + C*z1 + D < 0
=>A = -(M.14 + M.11) ; B = - (M.24 + M.21) ; C = -(M.34 + M.31) ; D = -(M.44 + M.41)

các mặt phẳng khác tương tự -w < y < w ; openGL : -w < z < w ; direct3D : 0 < z < w

*****************************************/
	// mặt phẳng trái
	planes[0].N.x = -(pViewProjMatrix->_14 + pViewProjMatrix->_11);
	planes[0].N.y = -(pViewProjMatrix->_24 + pViewProjMatrix->_21);
	planes[0].N.z = -(pViewProjMatrix->_34 + pViewProjMatrix->_31);
	planes[0].D    = -(pViewProjMatrix->_44 + pViewProjMatrix->_41);
	// mặt phẳng phải
	planes[1].N.x = -(pViewProjMatrix->_14 - pViewProjMatrix->_11);
	planes[1].N.y = -(pViewProjMatrix->_24 - pViewProjMatrix->_21);
	planes[1].N.z = -(pViewProjMatrix->_34 - pViewProjMatrix->_31);
	planes[1].D    = -(pViewProjMatrix->_44 - pViewProjMatrix->_41);
	// mặt phẳng trên
	planes[2].N.x = -(pViewProjMatrix->_14 - pViewProjMatrix->_12);
	planes[2].N.y = -(pViewProjMatrix->_24 - pViewProjMatrix->_22);
	planes[2].N.z = -(pViewProjMatrix->_34 - pViewProjMatrix->_32);
	planes[2].D    = -(pViewProjMatrix->_44 - pViewProjMatrix->_42);
	// mặt phẳng dưới
	planes[3].N.x = -(pViewProjMatrix->_14 + pViewProjMatrix->_12);
	planes[3].N.y = -(pViewProjMatrix->_24 + pViewProjMatrix->_22);
	planes[3].N.z = -(pViewProjMatrix->_34 + pViewProjMatrix->_32);
	planes[3].D    = -(pViewProjMatrix->_44 + pViewProjMatrix->_42);

	// mặt phẳng gần
	if (API == HQ_RA_OGL)//openGL
	{
		planes[4].N.x = -(pViewProjMatrix->_14 + pViewProjMatrix->_13);
		planes[4].N.y = -(pViewProjMatrix->_24 + pViewProjMatrix->_23);
		planes[4].N.z = -(pViewProjMatrix->_34 + pViewProjMatrix->_33);
		planes[4].D    = -(pViewProjMatrix->_44 + pViewProjMatrix->_43);
	}
	else //direct3D
	{
		planes[4].N.x = -pViewProjMatrix->_13;
		planes[4].N.y = -pViewProjMatrix->_23;
		planes[4].N.z = -pViewProjMatrix->_33;
		planes[4].D    = -pViewProjMatrix->_43;
	}
	// mặt phẳng xa
	planes[5].N.x = -(pViewProjMatrix->_14 - pViewProjMatrix->_13);
	planes[5].N.y = -(pViewProjMatrix->_24 - pViewProjMatrix->_23);
	planes[5].N.z = -(pViewProjMatrix->_34 - pViewProjMatrix->_33);
	planes[5].D    = -(pViewProjMatrix->_44 - pViewProjMatrix->_43);

	//chuẩn hóa pháp vector
	for (hq_int32 i=0; i<6 ; ++i)
	{
		planes[i].Normalize();
	}
}

/*----------------------------------------
truy vấn mặt phẳng tạo thành thể tích nhìn
----------------------------------------*/

void HQMatrix4cGetFrustum(const HQMatrix4 * pViewProjMatrix , HQPlane planes[6] , HQRenderAPI API)
{
	
/*****************************************
(x1,y1,z1,1) là tọa độ một điểm  trong không gian thế giới
gọi (x/w , y/w , z/w, 1) là tọa độ điểm đó trong không gian chuẩn hóa của device
ma trận view * projection là M

(x,y,z,w)= M * (x1,y1,z1,1)
=>x = M.11 * x1 + M.12 * y1 + M.13 * z1 + M.14
  w = M.41 * x1 + M.42 * y1 + M.43 * z1 + M.44

nếu điểm đó nằm trong thể tích nhìn thì trong không gian chuẩn hóa device:
    -1< (x/w) <1
=>-w < x < w
Xét giới hạn trái :-w < x
=>-(M.41 * x1 + M.42 * y1 + M.43 * z1 + M.44) < M.11 * x1 + M.12 * y1 + M.13 * z1 + M.14
=>-(M.41 + M.11) * x1 - (M.42 + M.12) * y1 -(M.43 + M.13) * z1 -(M.44 + M.14) < 0
=> vì trong không gian thế giới (x1,y1,z1,1) sẽ nằm trong thể tích nhìn tức là
sẽ nằm ngược phía với hướng của pháp vector mặt phẳng trái (giả sử pháp vector hướng ra ngoài
thể tích nhìn)
phương trình mặt trái là A*x + B*y + C*z + D=0
=>A*x1 + B*y1 + C*z1 + D < 0
=>A = -(M.41 + M.11) ; B = - (M.42 + M.12) ; C = -(M.43 + M.13) ; D = -(M.44 + M.14)

các mặt phẳng khác tương tự -w < y < w ; openGL : -w < z < w ; direct3D : 0 < z < w

*****************************************/
	// mặt phẳng trái
	planes[0].N.x = -(pViewProjMatrix->_41 + pViewProjMatrix->_11);
	planes[0].N.y = -(pViewProjMatrix->_42 + pViewProjMatrix->_12);
	planes[0].N.z = -(pViewProjMatrix->_43 + pViewProjMatrix->_13);
	planes[0].D    = -(pViewProjMatrix->_44 + pViewProjMatrix->_14);
	// mặt phẳng phải
	planes[1].N.x = -(pViewProjMatrix->_41 - pViewProjMatrix->_11);
	planes[1].N.y = -(pViewProjMatrix->_42 - pViewProjMatrix->_12);
	planes[1].N.z = -(pViewProjMatrix->_43 - pViewProjMatrix->_13);
	planes[1].D    = -(pViewProjMatrix->_44 - pViewProjMatrix->_14);
	// mặt phẳng trên
	planes[2].N.x = -(pViewProjMatrix->_41 - pViewProjMatrix->_21);
	planes[2].N.y = -(pViewProjMatrix->_42 - pViewProjMatrix->_22);
	planes[2].N.z = -(pViewProjMatrix->_43 - pViewProjMatrix->_23);
	planes[2].D    = -(pViewProjMatrix->_44 - pViewProjMatrix->_24);
	// mặt phẳng dưới
	planes[3].N.x = -(pViewProjMatrix->_41 + pViewProjMatrix->_21);
	planes[3].N.y = -(pViewProjMatrix->_42 + pViewProjMatrix->_22);
	planes[3].N.z = -(pViewProjMatrix->_43 + pViewProjMatrix->_23);
	planes[3].D    = -(pViewProjMatrix->_44 + pViewProjMatrix->_24);

	// mặt phẳng gần
	if (API == HQ_RA_OGL)//openGL
	{
		planes[4].N.x = -(pViewProjMatrix->_41 + pViewProjMatrix->_31);
		planes[4].N.y = -(pViewProjMatrix->_42 + pViewProjMatrix->_32);
		planes[4].N.z = -(pViewProjMatrix->_43 + pViewProjMatrix->_33);
		planes[4].D    = -(pViewProjMatrix->_44 + pViewProjMatrix->_34);
	}
	else //direct3D
	{
		planes[4].N.x = -pViewProjMatrix->_31;
		planes[4].N.y = -pViewProjMatrix->_32;
		planes[4].N.z = -pViewProjMatrix->_33;
		planes[4].D    = -pViewProjMatrix->_34;
	}
	// mặt phẳng xa
	planes[5].N.x = -(pViewProjMatrix->_41 - pViewProjMatrix->_31);
	planes[5].N.y = -(pViewProjMatrix->_42 - pViewProjMatrix->_32);
	planes[5].N.z = -(pViewProjMatrix->_43 - pViewProjMatrix->_33);
	planes[5].D    = -(pViewProjMatrix->_44 - pViewProjMatrix->_34);

	//chuẩn hóa pháp vector
	for (hq_int32 i=0; i<6 ; ++i)
	{
		planes[i].Normalize();
	}
}


//*************************************
//in matrix lên màn hình
//*************************************
void HQPrintMatrix4(const HQMatrix4* pMat)
{
	
	printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
		pMat->_11,pMat->_12,pMat->_13,pMat->_14,
		pMat->_21,pMat->_22,pMat->_23,pMat->_24,
		pMat->_31,pMat->_32,pMat->_33,pMat->_34,
		pMat->_41,pMat->_42,pMat->_43,pMat->_44
		);
}
