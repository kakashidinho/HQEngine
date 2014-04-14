/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
#ifdef HQ_NEON_MATH
#include "arm_neon_math/HQNeonVector.h"
#include "arm_neon_math/HQNeonMatrix.h"
#elif defined HQ_DIRECTX_MATH
#include "directx_math/HQDXVector.h"
#include "directx_math/HQDXMatrix.h"
#endif
#include <stdio.h>

#ifdef HQ_SSE_MATH

const hq_sse_float4 _3Halves_1Zero = {0.5f , 0.5f ,0.5f ,0.0f};
const hq_sse_float4 _4Threes = {3.0f , 3.0f ,3.0f ,3.0f};
const hq_sse_float4 _4Zeros = {0.0f , 0.0f , 0.0f , 0.0f};
const hq_sse_float4 _4Ones = {1.0f , 1.0f , 1.0f , 1.0f};
const hq_sse_float4 _3Zeros_1One = {0.0f , 0.0f ,0.0f , 1.0f};
const HQ_ALIGN16 hq_uint32 Mask[4]={0xffffffff,0xffffffff,0xffffffff,0x00000000};
const HQ_ALIGN16 hq_uint32 Mask2[4]={0x00000000,0x00000000,0x00000000,0xffffffff};

#endif


//**********************************************
//tích có hướng
//**********************************************
#ifndef HQ_EXPLICIT_ALIGN
HQVector4 HQVector4::Cross(const HQVector4& v2)const{
#ifdef HQ_CMATH
	HQVector4 result;
	result.x = y * v2.z - z * v2.y;
	result.y = z * v2.x - x * v2.z;
	result.z = x * v2.y - y * v2.x;
	result.w = 0.0f;
#elif defined HQ_NEON_MATH
	HQA16ByteVector4Ptr pResult;
	HQVector4 &result = *pResult;

	HQNeonVector4Cross(this->v, v2, result);

#elif defined HQ_DIRECTX_MATH

	HQVector4 result ;

	HQDXVector4Cross(this->v, v2, result);
#else
	HQVector4 result;
	hq_sse_float4 m0,m1,m2,m3;
	/*
	hq_float32 oldW1=v1->w;
	hq_float32 oldW2=v2->w;
	v1->w=0;
	v2->w=0;
	*/
	m0=_mm_load_ps(this->v);
	m1=_mm_load_ps(v2.v);
	m2=m0;
	m3=m1;
	m0=_mm_shuffle_ps(m0,m0,SSE_SHUFFLE(3,0,2,1));
	m1=_mm_shuffle_ps(m1,m1,SSE_SHUFFLE(3,1,0,2));
	m0=_mm_mul_ps(m0,m1);
	m2=_mm_shuffle_ps(m2,m2,SSE_SHUFFLE(3,1,0,2));
	m3=_mm_shuffle_ps(m3,m3,SSE_SHUFFLE(3,0,2,1));
	m2=_mm_mul_ps(m2,m3);
	m0=_mm_sub_ps(m0,m2);

	m0 = _mm_and_ps(m0 , _mm_load_ps((hq_float32*)Mask));//w = 0

	_mm_store_ps(result.v,m0);
	/*
	v1->w=oldW1;
	v2->w=oldW2;
	*/
#endif
	return result;
}
#endif//#ifndef HQ_EXPLICIT_ALIGN

HQVector4& HQVector4::Cross(const HQVector4 &v1, const HQVector4 &v2){
#ifdef HQ_CMATH
	HQVector4Cross(&v1, &v2 , this);
	
#elif defined HQ_NEON_MATH
	HQNeonVector4Cross(v1, v2, this->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4Cross(v1, v2, this->v);
#else
	hq_sse_float4 m0,m1,m2,m3;
	/*
	hq_float32 oldW1=v1->w;
	hq_float32 oldW2=v2->w;
	v1->w=0;
	v2->w=0;
	*/
	m0=_mm_load_ps(v1.v);
	m1=_mm_load_ps(v2.v);
	m2=m0;
	m3=m1;
	m0=_mm_shuffle_ps(m0,m0,SSE_SHUFFLE(3,0,2,1));
	m1=_mm_shuffle_ps(m1,m1,SSE_SHUFFLE(3,1,0,2));
	m0=_mm_mul_ps(m0,m1);
	m2=_mm_shuffle_ps(m2,m2,SSE_SHUFFLE(3,1,0,2));
	m3=_mm_shuffle_ps(m3,m3,SSE_SHUFFLE(3,0,2,1));
	m2=_mm_mul_ps(m2,m3);
	m0=_mm_sub_ps(m0,m2);

	m0=_mm_and_ps(m0 , _mm_load_ps((hq_float32*)Mask));//w = 0

	_mm_store_ps(this->v,m0);

	/*
	v1->w=oldW1;
	v2->w=oldW2;
	*/
#endif
	return *this;
}
HQVector4* HQVector4Cross(const HQVector4 *v1,const HQVector4 *v2,HQVector4* out){
#ifdef HQ_CMATH
	hq_float32 X = v1->y * v2->z - v1->z * v2->y;
	hq_float32 Y = v1->z * v2->x - v1->x * v2->z;
	out->z = v1->x * v2->y - v1->y * v2->x;
	out->x = X;
	out->y = Y;
	out->w = 0.0f;
#elif defined HQ_NEON_MATH
	HQNeonVector4Cross(v1->v, v2->v, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4Cross(v1->v, v2->v, out->v);
#else
	hq_sse_float4 m0,m1,m2,m3;
	/*
	hq_float32 oldW1=v1->w;
	hq_float32 oldW2=v2->w;
	v1->w=0;
	v2->w=0;
	*/
	m0=_mm_load_ps(v1->v);
	m1=_mm_load_ps(v2->v);
	m2=m0;
	m3=m1;
	m0=_mm_shuffle_ps(m0,m0,SSE_SHUFFLE(3,0,2,1));
	m1=_mm_shuffle_ps(m1,m1,SSE_SHUFFLE(3,1,0,2));
	m0=_mm_mul_ps(m0,m1);
	m2=_mm_shuffle_ps(m2,m2,SSE_SHUFFLE(3,1,0,2));
	m3=_mm_shuffle_ps(m3,m3,SSE_SHUFFLE(3,0,2,1));
	m2=_mm_mul_ps(m2,m3);
	m0=_mm_sub_ps(m0,m2);

	m0=_mm_and_ps(m0 , _mm_load_ps((hq_float32*)Mask));//w = 0

	_mm_store_ps(out->v,m0);

	/*
	v1->w=oldW1;
	v2->w=oldW2;
	 */
#endif
	return out;
}
//**********************************************
//chuẩn hóa
//**********************************************
HQVector4& HQVector4::Normalize(){
#ifdef HQ_CMATH
	hq_float32 f = 1.0f/sqrtf(x * x + y * y + z * z);
	x *= f;
	y *= f;
	z *= f;
	w = 0.0f;
#elif defined HQ_NEON_MATH
	HQNeonVector4Normalize(this->v, this->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4Normalize(this->v, this->v);
#else
	/* SSE intrinsics version*/
	hq_sse_float4 m0,m1,m2,m3;
	m0=_mm_load_ps(this->v);//copy vector data to xmm register
	m2=m0;//copy m0 vào m2
	m0=_mm_mul_ps(m0,m0);//nhân vector với chính nó x^2 y^2 z^2 w^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,0,3));//	y^2 x^2 x^2 w^2
	m3 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,1,3));//	z^2 z^2 y^2 w^2

	m0=_mm_add_ps(m1,m0);//x^2+y^2		y^2+x^2		z^2+x^2		w^2+w^2
	m0=_mm_add_ps(m3,m0);//x^2+y^2+z^2		x^2+y^2+z^2		x^2+y^2+z^2		w^2+w^2+w^2 


	hq_sse_float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	temp = _mm_and_ps(temp,_mm_load_ps((hq_float32*)Mask));// rsqrt(length) ,rsqrt(length) ,rsqrt(length)  ,0

	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_3Halves_1Zero,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));
	
	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài
	_mm_store_ps(this->v,m2);
#endif

	return *this;//trả về tham khảo đến this
}

HQVector4* HQVector4Normalize(const HQVector4* in,HQVector4* out){
#ifdef HQ_CMATH
	hq_float32 f = 1.0f/sqrtf(in->x * in->x + in->y * in->y + in->z * in->z);
	out->x = in->x * f;
	out->y = in->y * f;
	out->z = in->z * f;
	out->w = 0.0f;
#elif defined HQ_NEON_MATH
	HQNeonVector4Normalize(in->v , out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4Normalize(in->v , out->v);
#else
	/* SSE intrinsics version*/
	hq_sse_float4 m0,m1,m2,m3;
	m0=_mm_load_ps(in->v);//copy vector data to xmm register
	m2=m0;//copy m0 vào m2
	m0=_mm_mul_ps(m0,m0);//nhân vector với chính nó x^2 y^2 z^2 w^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,0,3));//	y^2 x^2 x^2 w^2
	m3 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,1,3));//	z^2 z^2 y^2 w^2 

	m0=_mm_add_ps(m1,m0);//x^2+y^2		y^2+x^2		z^2+x^2		w^2+w^2
	m0=_mm_add_ps(m3,m0);//x^2+y^2+z^2		x^2+y^2+z^2		x^2+y^2+z^2		w^2+w^2+w^2   
	
	hq_sse_float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	temp = _mm_and_ps(temp,_mm_load_ps((hq_float32*)Mask));// rsqrt(length) ,rsqrt(length) ,rsqrt(length)  ,0

	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_3Halves_1Zero,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));

	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài
	_mm_store_ps(out->v,m2);
#endif	

	return out;

}
//**********************************************************
//nhân vector với ma trận
//**********************************************************
HQVector4& HQVector4::operator *=(const HQMatrix4 &m){
#ifdef HQ_CMATH	
	/*normal version*/
	hq_float32 X=x*m._11+y*m._21+z*m._31+w*m._41;
	hq_float32 Y=x*m._12+y*m._22+z*m._32+w*m._42;
	hq_float32 Z=x*m._13+y*m._23+z*m._33+w*m._43;
	w = x*m._14+y*m._24+z*m._34+w*m._44;
	x = X;
	y = Y;
	z = Z;
#elif defined HQ_NEON_MATH
	HQNeonVector4MultiplyMatrix4(this->v, m, this->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4MultiplyMatrix4(this->v, m, this->v);
#else
	/*SSE version*/
	hq_sse_float4 m0,m1,m2,m3,m4,m5,m6,m7;
	//hq_float32 oldW=this->w;
	m0=_mm_load_ps(this->v);//x y z w
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m3 = hq_mm_copy_ps(m0,0xff);//w w w w
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

	m4=_mm_load_ps(m.m);
	m5=_mm_load_ps(m.m+4);
	m6=_mm_load_ps(m.m+8);
	m7=_mm_load_ps(m.m+12);

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	m3=_mm_mul_ps(m3,m7);

	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(v,m0);
	//w=oldW;
#endif
	return *this;
}

#ifndef HQ_EXPLICIT_ALIGN
HQVector4 HQVector4::operator *(const HQMatrix4 &m)const{
#ifdef HQ_CMATH
	HQVector4 result;
	/*normal version*/
	result.x=x*m._11+y*m._21+z*m._31+w*m._41;
	result.y=x*m._12+y*m._22+z*m._32+w*m._42;
	result.z=x*m._13+y*m._23+z*m._33+w*m._43;
	result.w =x*m._14+y*m._24+z*m._34+w*m._44;
#elif defined HQ_NEON_MATH
	HQA16ByteVector4Ptr pV;
	HQVector4 &result = *pV;
	HQNeonVector4MultiplyMatrix4(this->v, m, result);
#elif defined HQ_DIRECTX_MATH
	HQVector4 result;
	HQDXVector4MultiplyMatrix4(this->v, m, result);
#else
	HQVector4 result;
	/*SSE version*/
	hq_sse_float4 m0,m1,m2,m3,m4,m5,m6,m7;
	m0=_mm_load_ps(this->v);
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m3 = hq_mm_copy_ps(m0,0xff);//w w w w
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

	m4=_mm_load_ps(m.m);
	m5=_mm_load_ps(m.m+4);
	m6=_mm_load_ps(m.m+8);
	m7=_mm_load_ps(m.m+12);

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	m3=_mm_mul_ps(m3,m7);

	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(result.v,m0);
	//result.w=w;
#endif
	return result;
}
#endif //#ifndef HQ_EXPLICIT_ALIGN

HQVector4* HQVector4Transform(const HQVector4* v1,const HQMatrix4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v1)
		return &((*out) *= (*mat));
	else {
		out->x=v1->x*mat->_11+v1->y*mat->_21+v1->z*mat->_31+v1->w*mat->_41;
		out->y=v1->x*mat->_12+v1->y*mat->_22+v1->z*mat->_32+v1->w*mat->_42;
		out->z=v1->x*mat->_13+v1->y*mat->_23+v1->z*mat->_33+v1->w*mat->_43;
		out->w = v1->x*mat->_14+v1->y*mat->_24+v1->z*mat->_34+v1->w*mat->_44;
	}
	
#elif defined HQ_NEON_MATH
	HQNeonVector4MultiplyMatrix4(v1->v, mat->m, out->v);	
#elif defined HQ_DIRECTX_MATH
	HQDXVector4MultiplyMatrix4(v1->v, mat->m, out->v);	
#else
	//hq_float32 oldW=v1->w;
	hq_sse_float4 m0,m1,m2,m3,m4,m5,m6,m7;
	
	m4=_mm_load_ps(mat->m);
	m5=_mm_load_ps(mat->m+4);
	m6=_mm_load_ps(mat->m+8);
	m7=_mm_load_ps(mat->m+12);
	
	m0=_mm_load_ps(v1->v);
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m3 = hq_mm_copy_ps(m0,0xff);//w w w w
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x
	
	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	m3=_mm_mul_ps(m3,m7);
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);
	
	_mm_store_ps(out->v,m0);
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4MultiTransform(const HQVector4* v, hq_uint32 numVec, const HQMatrix4* mat,HQVector4* out)
{
#ifdef HQ_CMATH
	if (out == v)
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i] *= (*mat);
		}
	}
	else
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i].x = v[i].x*mat->_11+v[i].y*mat->_21+v[i].z*mat->_31+v[i].w*mat->_41;
			out[i].y = v[i].x*mat->_12+v[i].y*mat->_22+v[i].z*mat->_32+v[i].w*mat->_42;
			out[i].z = v[i].x*mat->_13+v[i].y*mat->_23+v[i].z*mat->_33+v[i].w*mat->_43;
			out[i].w = v[i].x*mat->_14+v[i].y*mat->_24+v[i].z*mat->_34+v[i].w*mat->_44;
		}
	}
#elif defined HQ_NEON_MATH
	HQNeonMultiVector4MultiplyMatrix4(v->v, numVec, mat->m, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXMultiVector4MultiplyMatrix4(v->v, numVec, mat->m, out->v);
#else
	//hq_float32 oldW=v1->w;
	hq_sse_float4 m0,m1,m2,m3,m4,m5,m6,m7;
	
	m4=_mm_load_ps(mat->m);
	m5=_mm_load_ps(mat->m+4);
	m6=_mm_load_ps(mat->m+8);
	m7=_mm_load_ps(mat->m+12);
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
		m0=_mm_load_ps(v[i].v);
		
		m1 = hq_mm_copy_ps(m0,0x55);//y y y y
		m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
		m3 = hq_mm_copy_ps(m0,0xff);//w w w w
		m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x
		
		m0=_mm_mul_ps(m0,m4);
		m1=_mm_mul_ps(m1,m5);
		m2=_mm_mul_ps(m2,m6);
		m3=_mm_mul_ps(m3,m7);
		
		m0=_mm_add_ps(m0,m1);
		m2=_mm_add_ps(m2,m3);
		m0=_mm_add_ps(m0,m2);
		
		_mm_store_ps(out[i].v,m0);
	}
	//out->w=oldW;
#endif
	return out;
}


HQVector4* HQVector4TransformCoord(const HQVector4* v1,const HQMatrix4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v1)
	{
		hq_float32 X=v1->x*mat->_11+v1->y*mat->_21+v1->z*mat->_31+ mat->_41;
		hq_float32 Y=v1->x*mat->_12+v1->y*mat->_22+v1->z*mat->_32+ mat->_42;
		hq_float32 Z=v1->x*mat->_13+v1->y*mat->_23+v1->z*mat->_33+ mat->_43;
		out->w = v1->x*mat->_14+v1->y*mat->_24+v1->z*mat->_34+ mat->_44;
		out->x = X;
		out->y = Y;
		out->z = Z;
	}
	else {
		out->x=v1->x*mat->_11+v1->y*mat->_21+v1->z*mat->_31+ mat->_41;
		out->y=v1->x*mat->_12+v1->y*mat->_22+v1->z*mat->_32+ mat->_42;
		out->z=v1->x*mat->_13+v1->y*mat->_23+v1->z*mat->_33+ mat->_43;
		out->w = v1->x*mat->_14+v1->y*mat->_24+v1->z*mat->_34+ mat->_44;
	}
#elif defined HQ_NEON_MATH
	HQNeonVector4TransformCoord(v1->v, mat->m, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4TransformCoord(v1->v, mat->m, out->v);
#else
	//hq_float32 oldW=v1->w;
	
	hq_sse_float4 m0,m1,m2,m3,m4,m5,m6;
	
	m4=_mm_load_ps(mat->m);
	m5=_mm_load_ps(mat->m+4);
	m6=_mm_load_ps(mat->m+8);
	m3=_mm_load_ps(mat->m+12);
	
	m0=_mm_load_ps(v1->v);
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x
	
	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);
	
	_mm_store_ps(out->v,m0);
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4MultiTransformCoord(const HQVector4* v, hq_uint32 numVec, const HQMatrix4* mat,HQVector4* out)
{
#ifdef HQ_CMATH
	if (out == v)
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			hq_float32 X=v[i].x*mat->_11+v[i].y*mat->_21+v[i].z*mat->_31+ mat->_41;
			hq_float32 Y=v[i].x*mat->_12+v[i].y*mat->_22+v[i].z*mat->_32+ mat->_42;
			hq_float32 Z=v[i].x*mat->_13+v[i].y*mat->_23+v[i].z*mat->_33+ mat->_43;
			out[i].w = v[i].x*mat->_14+v[i].y*mat->_24+v[i].z*mat->_34+ mat->_44;
			out[i].x = X;
			out[i].y = Y;
			out[i].z = Z;
		}
	}
	else
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i].x=v[i].x*mat->_11+v[i].y*mat->_21+v[i].z*mat->_31+ mat->_41;
			out[i].y=v[i].x*mat->_12+v[i].y*mat->_22+v[i].z*mat->_32+ mat->_42;
			out[i].z=v[i].x*mat->_13+v[i].y*mat->_23+v[i].z*mat->_33+ mat->_43;
			out[i].w = v[i].x*mat->_14+v[i].y*mat->_24+v[i].z*mat->_34+ mat->_44;
		}
	}
	
#elif defined HQ_NEON_MATH
	HQNeonMultiVector4TransformCoord(v->v, numVec, mat->m, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXMultiVector4TransformCoord(v->v, numVec, mat->m, out->v);
#else
	//hq_float32 oldW=v1->w;
	hq_sse_float4 m0,m1,m2,m3,m4,m5,m6;
	
	m4=_mm_load_ps(mat->m);
	m5=_mm_load_ps(mat->m+4);
	m6=_mm_load_ps(mat->m+8);
	m3=_mm_load_ps(mat->m+12);
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
		m0=_mm_load_ps(v[i].v);
		
		m1 = hq_mm_copy_ps(m0,0x55);//y y y y
		m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
		m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x
		
		m0=_mm_mul_ps(m0,m4);
		m1=_mm_mul_ps(m1,m5);
		m2=_mm_mul_ps(m2,m6);
		
		m0=_mm_add_ps(m0,m1);
		m2=_mm_add_ps(m2,m3);
		m0=_mm_add_ps(m0,m2);
		
		_mm_store_ps(out[i].v,m0);
	}
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4TransformNormal(const HQVector4* v1,const HQMatrix4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v1)
	{
		hq_float32 X=v1->x*mat->_11+v1->y*mat->_21+v1->z*mat->_31;
		hq_float32 Y=v1->x*mat->_12+v1->y*mat->_22+v1->z*mat->_32;
		hq_float32 Z=v1->x*mat->_13+v1->y*mat->_23+v1->z*mat->_33;
		out->w = v1->x*mat->_14+v1->y*mat->_24+v1->z*mat->_34;
		out->x = X;
		out->y = Y;
		out->z = Z;
	}
	else {
		out->x=v1->x*mat->_11+v1->y*mat->_21+v1->z*mat->_31;
		out->y=v1->x*mat->_12+v1->y*mat->_22+v1->z*mat->_32;
		out->z=v1->x*mat->_13+v1->y*mat->_23+v1->z*mat->_33;
		out->w = v1->x*mat->_14+v1->y*mat->_24+v1->z*mat->_34;
	}
	
#elif defined HQ_NEON_MATH
	HQNeonVector4TransformNormal(v1->v, mat->m, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXVector4TransformNormal(v1->v, mat->m, out->v);
#else
	//hq_float32 oldW=v1->w;

	hq_sse_float4 m0,m1,m2,m4,m5,m6;

	m4=_mm_load_ps(mat->m);
	m5=_mm_load_ps(mat->m+4);
	m6=_mm_load_ps(mat->m+8);
	
	m0=_mm_load_ps(v1->v);
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);

	m0=_mm_add_ps(m0,m1);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(out->v,m0);
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4MultiTransformNormal(const HQVector4* v, hq_uint32 numVec, const HQMatrix4* mat,HQVector4* out)
{
#ifdef HQ_CMATH
	if (out == v)
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			hq_float32 X=v[i].x*mat->_11+v[i].y*mat->_21+v[i].z*mat->_31;
			hq_float32 Y=v[i].x*mat->_12+v[i].y*mat->_22+v[i].z*mat->_32;
			hq_float32 Z=v[i].x*mat->_13+v[i].y*mat->_23+v[i].z*mat->_33;
			out[i].w = v[i].x*mat->_14+v[i].y*mat->_24+v[i].z*mat->_34;
			out[i].x = X;
			out[i].y = Y;
			out[i].z = Z;
		}
	}
	else
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i].x=v[i].x*mat->_11+v[i].y*mat->_21+v[i].z*mat->_31;
			out[i].y=v[i].x*mat->_12+v[i].y*mat->_22+v[i].z*mat->_32;
			out[i].z=v[i].x*mat->_13+v[i].y*mat->_23+v[i].z*mat->_33;
			out[i].w = v[i].x*mat->_14+v[i].y*mat->_24+v[i].z*mat->_34;
		}
	}
	
#elif defined HQ_NEON_MATH
	HQNeonMultiVector4TransformNormal(v->v, numVec, mat->m, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXMultiVector4TransformNormal(v->v, numVec, mat->m, out->v);
#else
	//hq_float32 oldW=v1->w;
	hq_sse_float4 m0,m1,m2,m4,m5,m6;
	
	m4=_mm_load_ps(mat->m);
	m5=_mm_load_ps(mat->m+4);
	m6=_mm_load_ps(mat->m+8);
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
		m0=_mm_load_ps(v[i].v);
		
		m1 = hq_mm_copy_ps(m0,0x55);//y y y y
		m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
		m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

		m0=_mm_mul_ps(m0,m4);
		m1=_mm_mul_ps(m1,m5);
		m2=_mm_mul_ps(m2,m6);

		m0=_mm_add_ps(m0,m1);
		m0=_mm_add_ps(m0,m2);
		
		_mm_store_ps(out[i].v,m0);
	}
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4Transform(const HQVector4* v1,const HQMatrix3x4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v1)
	{
		hq_float32 X=v1->x*mat->_11+v1->y*mat->_12+v1->z*mat->_13+ v1->w*mat->_14;
		hq_float32 Y=v1->x*mat->_21+v1->y*mat->_22+v1->z*mat->_23+ v1->w*mat->_24;
		hq_float32 Z=v1->x*mat->_31+v1->y*mat->_32+v1->z*mat->_33+ v1->w*mat->_34;
		out->w = v1->w;
		out->x = X;
		out->y = Y;
		out->z = Z;
	}
	else {
		out->x=v1->x*mat->_11+v1->y*mat->_12+v1->z*mat->_13+v1->w*mat->_14;
		out->y=v1->x*mat->_21+v1->y*mat->_22+v1->z*mat->_23+v1->w*mat->_24;
		out->z=v1->x*mat->_31+v1->y*mat->_32+v1->z*mat->_33+v1->w*mat->_34;
		out->w = v1->w;
	}
#elif defined HQ_NEON_MATH
	HQNeonMatrix3x4MultiplyVector4(mat->m, v1->v, out->v);
#elif defined HQ_DIRECTX_MATH
	HQDXMatrix3x4MultiplyVector4(mat->m, v1->v, out->v);
#elif defined HQ_SSE4_MATH
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m3 ;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44
	
	//load vector	
	m3 = _mm_load_ps(v1->v);
	
	m0 = _mm_dp_ps(m0 , m3 , SSE4_DP_MASK(1,1,1,1,0,0,0,1));
	m0 = _mm_or_ps(m0 , _mm_dp_ps(m1 , m3 , SSE4_DP_MASK(1,1,1,1,0,0,1,0)));
	m0 = _mm_or_ps(m0 , _mm_dp_ps(m2 , m3 , SSE4_DP_MASK(1,1,1,1,0,1,0,0)));
	m0 = _mm_or_ps(m0 , _mm_and_ps(m3 , _mm_load_ps((hq_float32*)Mask2)));
	
	_mm_store_ps(out->v,m0);
#else
	//hq_float32 oldW=v1->w;
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44

	//transpose matrix
	m5 = _mm_unpacklo_ps(m0 , m1);//11 21 12 22
	m6 = _mm_unpacklo_ps(m2 , _3Zeros_1One);//31 41 32 42
	m4 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//11 21 31 41
	m5 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//12 22 32 42

	m7 = _mm_unpackhi_ps(m0 , m1);//13 23 14 24
	m0 = _mm_unpackhi_ps(m2 , _3Zeros_1One);//33 43 34 44
	m6 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//13 23 33 43
	m7 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//14 24 34 44
	
	//load vector	
	m0=_mm_load_ps(v1->v);

	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m3 = hq_mm_copy_ps(m0,0xff);//w w w w
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x
	
	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	m3=_mm_mul_ps(m3,m7);
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);
	
	_mm_store_ps(out->v,m0);
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4MultiTransform(const HQVector4* v,hquint32 numVec , const HQMatrix3x4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v)
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			hq_float32 X=v[i].x*mat->_11+v[i].y*mat->_12+v[i].z*mat->_13+ v[i].w*mat->_14;
			hq_float32 Y=v[i].x*mat->_21+v[i].y*mat->_22+v[i].z*mat->_23+ v[i].w*mat->_24;
			hq_float32 Z=v[i].x*mat->_31+v[i].y*mat->_32+v[i].z*mat->_33+ v[i].w*mat->_34;
			out[i].w = v[i].w;
			out[i].x = X;
			out[i].y = Y;
			out[i].z = Z;
		}
	}
	else {
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i].x=v[i].x*mat->_11+v[i].y*mat->_12+v[i].z*mat->_13+v[i].w*mat->_14;
			out[i].y=v[i].x*mat->_21+v[i].y*mat->_22+v[i].z*mat->_23+v[i].w*mat->_24;
			out[i].z=v[i].x*mat->_31+v[i].y*mat->_32+v[i].z*mat->_33+v[i].w*mat->_34;
			out[i].w = v[i].w;
		}
	}
	
#elif defined HQ_NEON_MATH
	
	HQNeonMatrix3x4MultiplyMultiVector4(mat->m, v->v, numVec, out->v);

#elif defined HQ_DIRECTX_MATH

	HQDXMatrix3x4MultiplyMultiVector4(mat->m, v->v, numVec, out->v);
	
#elif defined HQ_SSE4_MATH
	hq_sse_float4 m0 , m1 , m2 , m3 , m4;

	m1 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m2 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m3 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
			
		//load vector	
		m4=_mm_load_ps(v[i].v);

		m0 = _mm_dp_ps(m1 , m4 , SSE4_DP_MASK(1,1,1,1,0,0,0,1));
		m0 = _mm_or_ps(m0 , _mm_dp_ps(m2 , m4 , SSE4_DP_MASK(1,1,1,1,0,0,1,0)));
		m0 = _mm_or_ps(m0 , _mm_dp_ps(m3 , m4 , SSE4_DP_MASK(1,1,1,1,0,1,0,0)));
		m0 = _mm_or_ps(m0 , _mm_and_ps(m4 , _mm_load_ps((hq_float32*)Mask2)));
		
		_mm_store_ps(out[i].v,m0);
	}
#else
	//hq_float32 oldW=v1->w;
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44

	//transpose matrix
	m5 = _mm_unpacklo_ps(m0 , m1);//11 21 12 22
	m6 = _mm_unpacklo_ps(m2 , _3Zeros_1One);//31 41 32 42
	m4 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//11 21 31 41
	m5 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//12 22 32 42

	m7 = _mm_unpackhi_ps(m0 , m1);//13 23 14 24
	m0 = _mm_unpackhi_ps(m2 , _3Zeros_1One);//33 43 34 44
	m6 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//13 23 33 43
	m7 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//14 24 34 44
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
			
		//load vector	
		m0=_mm_load_ps(v[i].v);

		m1 = hq_mm_copy_ps(m0,0x55);//y y y y
		m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
		m3 = hq_mm_copy_ps(m0,0xff);//w w w w
		m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x
		
		m0=_mm_mul_ps(m0,m4);
		m1=_mm_mul_ps(m1,m5);
		m2=_mm_mul_ps(m2,m6);
		m3=_mm_mul_ps(m3,m7);
		
		m0=_mm_add_ps(m0,m1);
		m2=_mm_add_ps(m2,m3);
		m0=_mm_add_ps(m0,m2);
		
		_mm_store_ps(out[i].v,m0);
	}
	//out->w=oldW;
#endif
	return out;
}


HQVector4* HQVector4TransformCoord(const HQVector4* v1,const HQMatrix3x4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v1)
	{
		hq_float32 X=v1->x*mat->_11+v1->y*mat->_12+v1->z*mat->_13+ mat->_14;
		hq_float32 Y=v1->x*mat->_21+v1->y*mat->_22+v1->z*mat->_23+ mat->_24;
		hq_float32 Z=v1->x*mat->_31+v1->y*mat->_32+v1->z*mat->_33+ mat->_34;
		out->w = 1.0f;
		out->x = X;
		out->y = Y;
		out->z = Z;
	}
	else {
		out->x=v1->x*mat->_11+v1->y*mat->_12+v1->z*mat->_13+ mat->_14;
		out->y=v1->x*mat->_21+v1->y*mat->_22+v1->z*mat->_23+ mat->_24;
		out->z=v1->x*mat->_31+v1->y*mat->_32+v1->z*mat->_33+ mat->_34;
		out->w = 1.0f;
	}
#elif defined HQ_NEON_MATH
	
	HQNeonVector4TransformCoordMatrix3x4(v1->v, mat->m, out->v);

#elif defined HQ_DIRECTX_MATH

	HQDXVector4TransformCoordMatrix3x4(v1->v, mat->m, out->v);
	
#elif defined HQ_SSE4_MATH
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m3;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44
	
	//load vector	
	m3 = _mm_load_ps(v1->v);//x y z w
	m3 = _mm_and_ps(m3 , _mm_load_ps((hq_float32*)Mask));//x y z 0
	m3 = _mm_or_ps(m3 , _3Zeros_1One);// x y z 1.0f
	
	m0 = _mm_dp_ps(m0 , m3 , SSE4_DP_MASK(1,1,1,1,0,0,0,1));
	m0 = _mm_or_ps(m0 , _mm_dp_ps(m1 , m3 , SSE4_DP_MASK(1,1,1,1,0,0,1,0)));
	m0 = _mm_or_ps(m0 , _mm_dp_ps(m2 , m3 , SSE4_DP_MASK(1,1,1,1,0,1,0,0)));
	m0 = _mm_or_ps(m0 , _3Zeros_1One);
	
	_mm_store_ps(out->v,m0);
#else
	//hq_float32 oldW=v1->w;
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m4 , m5 , m6 , m7;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44

	//transpose matrix
	m5 = _mm_unpacklo_ps(m0 , m1);//11 21 12 22
	m6 = _mm_unpacklo_ps(m2 , _3Zeros_1One);//31 41 32 42
	m4 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//11 21 31 41
	m5 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//12 22 32 42

	m7 = _mm_unpackhi_ps(m0 , m1);//13 23 14 24
	m0 = _mm_unpackhi_ps(m2 , _3Zeros_1One);//33 43 34 44
	m6 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//13 23 33 43
	m7 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//14 24 34 44

	//load vector
	m0=_mm_load_ps(v1->v);
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m7);
	m0=_mm_add_ps(m0,m2);
	
	_mm_store_ps(out->v,m0);
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4MultiTransformCoord(const HQVector4* v ,hquint32 numVec ,const HQMatrix3x4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v)
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			hq_float32 X=v[i].x*mat->_11+v[i].y*mat->_12+v[i].z*mat->_13+ mat->_14;
			hq_float32 Y=v[i].x*mat->_21+v[i].y*mat->_22+v[i].z*mat->_23+ mat->_24;
			hq_float32 Z=v[i].x*mat->_31+v[i].y*mat->_32+v[i].z*mat->_33+ mat->_34;
			out[i].w = 1.0f;
			out[i].x = X;
			out[i].y = Y;
			out[i].z = Z;
		}
	}
	else {
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i].x=v[i].x*mat->_11+v[i].y*mat->_12+v[i].z*mat->_13+ mat->_14;
			out[i].y=v[i].x*mat->_21+v[i].y*mat->_22+v[i].z*mat->_23+ mat->_24;
			out[i].z=v[i].x*mat->_31+v[i].y*mat->_32+v[i].z*mat->_33+ mat->_34;
			out[i].w = 1.0f;
		}
	}
	
#elif defined HQ_NEON_MATH
	
	HQNeonMultiVector4TransformCoordMatrix3x4(v->v, numVec, mat->m, out->v);

#elif defined HQ_DIRECTX_MATH

	HQDXMultiVector4TransformCoordMatrix3x4(v->v, numVec, mat->m, out->v);
	
#elif defined HQ_SSE4_MATH
	hq_sse_float4 m0 , m1 , m2 , m3 , m4;

	m1 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m2 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m3 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
			
		//load vector	
		m4=_mm_load_ps(v[i].v);//x y z w
		m4 = _mm_and_ps(m4 , _mm_load_ps((hq_float32*)Mask));//x y z 0
		m4 = _mm_or_ps(m4 , _3Zeros_1One);// x y z 1.0f

		m0 = _mm_dp_ps(m1 , m4 , SSE4_DP_MASK(1,1,1,1,0,0,0,1));
		m0 = _mm_or_ps(m0 , _mm_dp_ps(m2 , m4 , SSE4_DP_MASK(1,1,1,1,0,0,1,0)));
		m0 = _mm_or_ps(m0 , _mm_dp_ps(m3 , m4 , SSE4_DP_MASK(1,1,1,1,0,1,0,0)));
		m0 = _mm_or_ps(m0 , _3Zeros_1One);
		
		_mm_store_ps(out[i].v,m0);
	}
#else
	//hq_float32 oldW=v1->w;
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m4 , m5 , m6 , m7;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44

	//transpose matrix
	m5 = _mm_unpacklo_ps(m0 , m1);//11 21 12 22
	m6 = _mm_unpacklo_ps(m2 , _3Zeros_1One);//31 41 32 42
	m4 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//11 21 31 41
	m5 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//12 22 32 42

	m7 = _mm_unpackhi_ps(m0 , m1);//13 23 14 24
	m0 = _mm_unpackhi_ps(m2 , _3Zeros_1One);//33 43 34 44
	m6 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//13 23 33 43
	m7 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//14 24 34 44
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
		//load vector
		m0=_mm_load_ps(v[i].v);

		m1 = hq_mm_copy_ps(m0,0x55);//y y y y
		m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
		m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

		m0=_mm_mul_ps(m0,m4);
		m1=_mm_mul_ps(m1,m5);
		m2=_mm_mul_ps(m2,m6);
		
		m0=_mm_add_ps(m0,m1);
		m2=_mm_add_ps(m2,m7);
		m0=_mm_add_ps(m0,m2);
		
		_mm_store_ps(out[i].v,m0);
	}
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4TransformNormal(const HQVector4* v1,const HQMatrix3x4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v1)
	{
		hq_float32 X=v1->x*mat->_11+v1->y*mat->_12+v1->z*mat->_13;
		hq_float32 Y=v1->x*mat->_21+v1->y*mat->_22+v1->z*mat->_23;
		hq_float32 Z=v1->x*mat->_31+v1->y*mat->_32+v1->z*mat->_33;
		out->w = 0.0f;
		out->x = X;
		out->y = Y;
		out->z = Z;
	}
	else {
		out->x=v1->x*mat->_11+v1->y*mat->_12+v1->z*mat->_13;
		out->y=v1->x*mat->_21+v1->y*mat->_22+v1->z*mat->_23;
		out->z=v1->x*mat->_31+v1->y*mat->_32+v1->z*mat->_33;
		out->w = 0.0f;
	}
#elif defined HQ_NEON_MATH
	
	HQNeonVector4TransformNormalMatrix3x4(v1->v, mat->m, out->v);

#elif defined HQ_DIRECTX_MATH

	HQDXVector4TransformNormalMatrix3x4(v1->v, mat->m, out->v);
	
#elif defined HQ_SSE4_MATH
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m3 ;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44
	
	//load vector	
	m3 = _mm_load_ps(v1->v);
	
	m0 = _mm_dp_ps(m0 , m3 , SSE4_DP_MASK(0,1,1,1,0,0,0,1));
	m0 = _mm_or_ps(m0 , _mm_dp_ps(m1 , m3 , SSE4_DP_MASK(0,1,1,1,0,0,1,0)));
	m0 = _mm_or_ps(m0 , _mm_dp_ps(m2 , m3 , SSE4_DP_MASK(0,1,1,1,0,1,0,0)));
	
	_mm_store_ps(out->v,m0);	
#else
	//hq_float32 oldW=v1->w;
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m4 , m5 , m6 , m7;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44

	//transpose matrix
	m5 = _mm_unpacklo_ps(m0 , m1);//11 21 12 22
	m6 = _mm_unpacklo_ps(m2 , _3Zeros_1One);//31 41 32 42
	m4 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//11 21 31 41
	m5 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//12 22 32 42

	m7 = _mm_unpackhi_ps(m0 , m1);//13 23 14 24
	m0 = _mm_unpackhi_ps(m2 , _3Zeros_1One);//33 43 34 44
	m6 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//13 23 33 43
	
	//load vector
	m0=_mm_load_ps(v1->v);
	
	m1 = hq_mm_copy_ps(m0,0x55);//y y y y
	m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
	m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

	m0=_mm_mul_ps(m0,m4);
	m1=_mm_mul_ps(m1,m5);
	m2=_mm_mul_ps(m2,m6);

	m0=_mm_add_ps(m0,m1);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(out->v,m0);
	//out->w=oldW;
#endif
	return out;
}

HQVector4* HQVector4MultiTransformNormal(const HQVector4* v , hquint32 numVec ,const HQMatrix3x4* mat,HQVector4* out){
#ifdef HQ_CMATH
	if (out == v)
	{
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			hq_float32 X=v[i].x*mat->_11+v[i].y*mat->_12+v[i].z*mat->_13;
			hq_float32 Y=v[i].x*mat->_21+v[i].y*mat->_22+v[i].z*mat->_23;
			hq_float32 Z=v[i].x*mat->_31+v[i].y*mat->_32+v[i].z*mat->_33;
			out[i].w = 0.0f;
			out[i].x = X;
			out[i].y = Y;
			out[i].z = Z;
		}
	}
	else {
		for (hquint32 i = 0 ; i < numVec ; ++i)
		{
			out[i].x=v[i].x*mat->_11+v[i].y*mat->_12+v[i].z*mat->_13;
			out[i].y=v[i].x*mat->_21+v[i].y*mat->_22+v[i].z*mat->_23;
			out[i].z=v[i].x*mat->_31+v[i].y*mat->_32+v[i].z*mat->_33;
			out[i].w = 0.0f;
		}
	}
	
#elif defined HQ_NEON_MATH
	
	HQNeonMultiVector4TransformNormalMatrix3x4(v->v, numVec, mat->m, out->v);

#elif defined HQ_DIRECTX_MATH

	HQDXMultiVector4TransformNormalMatrix3x4(v->v, numVec, mat->m, out->v);
	
#elif defined HQ_SSE4_MATH
	hq_sse_float4 m0 , m1 , m2 , m3 , m4;

	m1 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m2 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m3 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
			
		//load vector	
		m4=_mm_load_ps(v[i].v);

		m0 = _mm_dp_ps(m1 , m4 , SSE4_DP_MASK(0,1,1,1,0,0,0,1));
		m0 = _mm_or_ps(m0 , _mm_dp_ps(m2 , m4 , SSE4_DP_MASK(0,1,1,1,0,0,1,0)));
		m0 = _mm_or_ps(m0 , _mm_dp_ps(m3 , m4 , SSE4_DP_MASK(0,1,1,1,0,1,0,0)));
		
		_mm_store_ps(out[i].v,m0);
	}	
#else
	//hq_float32 oldW=v1->w;
	//load matrix
	hq_sse_float4 m0 , m1 , m2 , m4 , m5 , m6 , m7;

	m0 = _mm_load_ps(&mat->m[0]);//11 12 13 14
	m1 = _mm_load_ps(&mat->m[4]);//21 22 23 24
	m2 = _mm_load_ps(&mat->m[8]);//31 32 33 34
	//_3Zeros_1One	{0  0  0  1}  41 42 43 44

	//transpose matrix
	m5 = _mm_unpacklo_ps(m0 , m1);//11 21 12 22
	m6 = _mm_unpacklo_ps(m2 , _3Zeros_1One);//31 41 32 42
	m4 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//11 21 31 41
	m5 = _mm_shuffle_ps(m5 , m6 , SSE_SHUFFLEL(2 , 3 , 2 , 3));//12 22 32 42

	m7 = _mm_unpackhi_ps(m0 , m1);//13 23 14 24
	m0 = _mm_unpackhi_ps(m2 , _3Zeros_1One);//33 43 34 44
	m6 = _mm_shuffle_ps(m7 , m0 , SSE_SHUFFLEL(0 , 1 , 0 , 1));//13 23 33 43
	
	for (hquint32 i = 0 ; i < numVec ; ++i)
	{
		//load vector
		m0=_mm_load_ps(v[i].v);
		
		m1 = hq_mm_copy_ps(m0,0x55);//y y y y
		m2 = hq_mm_copy_ps(m0,0xaa);//z z z z
		m0 = _mm_shuffle_ps(m0,m0,0x00);//x x x x

		m0=_mm_mul_ps(m0,m4);
		m1=_mm_mul_ps(m1,m5);
		m2=_mm_mul_ps(m2,m6);

		m0=_mm_add_ps(m0,m1);
		m0=_mm_add_ps(m0,m2);

		_mm_store_ps(out[i].v,m0);
	}
#endif
	return out;
}

//*************************************
//Biến đổi vector với quaternion
//*************************************
HQVector4* HQVector4Transform(const HQVector4* v1,const HQQuaternion* quat,HQVector4* out)
{
#ifdef HQ_EXPLICIT_ALIGN
	HQA16ByteStorageArrayPtr<HQQuaternion , 3> q;
	q[0].Set(v1->x,v1->y,v1->z,0);
	HQQuaternion &quat1 = q[0];
	HQQuaternion &quat2 = q[1];
	HQQuaternion &temp =  q[2];
#else
	HQQuaternion quat1(v1->x,v1->y,v1->z,0);
	HQQuaternion quat2(NULL);
	HQQuaternion temp(NULL);
#endif

	//HQQuatMultiply(&quat1 , &(-(*quat)) , &quat2);
	HQQuatMultiply(&quat1 , HQQuatNegate(quat, &temp) , &quat2);

	//quat2=(*quat)*quat2;
	HQQuatMultiply(quat, &quat2, &quat2);

	out->x=quat2.x;
	out->y=quat2.y;
	out->z=quat2.z;
	out->w=v1->w;
	return out;
}

HQVector4* HQVector4Transform(const HQVector4* v1,const HQOBB* box,HQVector4* out)
{
	if (v1->w != 0.0f)
	{
		HQVector4Sub(v1, &box->vCenter, out);//translation
	}
	out->x = *out * box->vA[0];
	out->y = *out * box->vA[1];
	out->z = *out * box->vA[2];
	out->w = v1->w;

	return out;
}
HQVector4* HQVector4TransformCoord(const HQVector4* v1,const HQOBB* box,HQVector4* out)
{
	HQVector4Sub(v1, &box->vCenter, out);//translation

	out->x = *out * box->vA[0];
	out->y = *out * box->vA[1];
	out->z = *out * box->vA[2];
	out->w = 1.0f;

	return out;
}
HQVector4* HQVector4TransformNormal(const HQVector4* v1,const HQOBB* box,HQVector4* out)
{
	out->x = *out * box->vA[0];
	out->y = *out * box->vA[1];
	out->z = *out * box->vA[2];
	out->w = 0.0f;

	return out;
}

//*************************************
//In vector lên màn hình
//*************************************
void HQPrintVector4(const HQVector4* pV)
{
	printf("%f %f %f %f\n",
		pV->x,pV->y,pV->z,pV->w);
}
//*************************************
//Góc giữa 2 vector
//*************************************
hq_float32 HQVector4::AngleWith(HQVector4& v2){
	return (hq_float32)acos(((*this)*v2)/(this->Length()*v2.Length()));
}
