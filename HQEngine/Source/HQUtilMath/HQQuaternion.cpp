/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
#ifdef HQ_NEON_MATH
#include "arm_neon_math/HQNeonQuaternion.h"
#elif defined HQ_DIRECTX_MATH
#include "directx_math/HQDXQuaternion.h"
#endif

#ifdef LINUX
#include <string.h>//for memset
#endif

#ifdef ANDROID
//#	define TRACE(...) __android_log_print(ANDROID_LOG_DEBUG, "test", __VA_ARGS__)
#	define TRACE(...)
#else
#	define TRACE(...)
#endif

#ifdef HQ_SSE_MATH

const float4 _4Halves= {.5f,.5f,.5f,.5f};
const float4 _4Quarts= {.25f,.25f,.25f,.25f};
const float4 _1One_3Zeros = {1.0f , 0.0f , 0.0f , 0.0f};
const HQ_ALIGN16 hq_uint32 NotMask[]={0xffffffff,0xffffffff,0xffffffff,0xffffffff};
const HQ_ALIGN16 hq_uint32 QuatMask1[]={0x00000000,0x80000000,0x00000000,0x80000000};//+ - + -
const HQ_ALIGN16 hq_uint32 QuatMask2[]={0x00000000,0x00000000,0x80000000,0x80000000};//+ + - -
const HQ_ALIGN16 hq_uint32 QuatMask3[]={0x80000000,0x00000000,0x00000000,0x80000000};//- + + -
const HQ_ALIGN16 hq_uint32 QuatMask4[]={0x00000000,0x80000000,0x80000000,0x80000000};//+ - - -
const HQ_ALIGN16 hq_uint32 QuatMask5[]={0x80000000,0x00000000,0x80000000,0x80000000};//- + - -
const HQ_ALIGN16 hq_uint32 QuatMask6[]={0x80000000,0x80000000,0x00000000,0x00000000};//- - + +
const HQ_ALIGN16 hq_uint32 NegSignMask[]={0x80000000,0x80000000,0x80000000,0x80000000};//- - - -

#endif

//*************************
//tạo ma trận từ quaternion
//*************************
void HQQuaternion::QuatToMatrix4r(HQMatrix4 *matOut){
	this->Normalize();
	this->QuatUnitToMatrix4r(matOut);
}
void HQQuaternion::QuatToMatrix3x4c(HQMatrix3x4 *matOut){
	this->Normalize();
	this->QuatUnitToMatrix3x4c(matOut);
}
//**************************************
//tạo ma trận từ quaternion đã chuẩn hóa
//**************************************
void HQQuaternion::QuatUnitToMatrix4r(HQMatrix4 *matOut)const{
#ifdef HQ_CMATH
	hq_float32 _2x=x+x;
	hq_float32 _2y=y+y;
	hq_float32 _2z=z+z;

	hq_float32 _2x2=_2x*x;
	hq_float32 _2y2=_2y*y;
	hq_float32 _2z2=_2z*z;
	hq_float32 _2xy=_2x*y;
	hq_float32 _2xz=_2x*z;
	hq_float32 _2yz=_2y*z;
	hq_float32 _2xw=_2x*w;
	hq_float32 _2yw=_2y*w;
	hq_float32 _2zw=_2z*w;

	matOut->_11=1-(_2y2+_2z2);
	matOut->_22=1-(_2x2+_2z2);
	matOut->_33=1-(_2x2+_2y2);
	matOut->_44=1.0f;
	matOut->_14=matOut->_24=matOut->_34
		=matOut->_41=matOut->_42=matOut->_43=0.0f;

	matOut->_21=_2xy-_2zw;
	matOut->_31=_2xz+_2yw;

	matOut->_12=_2xy+_2zw;
	matOut->_32=_2yz-_2xw;

	matOut->_13=_2xz-_2yw;
	matOut->_23=_2yz+_2xw;

#elif defined HQ_NEON_MATH
	
	HQNeonQuatUnitToMatrix4r(this->q, matOut->m);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatUnitToMatrix4r(this->q, matOut->m);

#else
	/*SSE*/
	float4 m0,m1,m2,m3,m4,m5,m7;
	m0 = _mm_load_ps(this->q);// x y z w
	m1 = _mm_add_ps(m0 , m0); //2x 2y 2z 2w
	
	m2 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,0,1));// y x x y
	m3 = hq_mm_copy_ps(m1,SSE_SHUFFLEL(1,1,2,2));// 2y 2y 2z 2z
	m2 = _mm_mul_ps(m2,m3);// 2yy 2xy 2xz 2yz

	m4 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,3,3));// z w w w
	m5 = hq_mm_copy_ps(m1,SSE_SHUFFLEL(2,2,1,0));// 2z 2z 2y 2x
	m4 = _mm_mul_ps(m4,m5);// 2zz 2zw 2yw 2xw

	m0 = _mm_mul_ps(m0,m1);// 2xx 2yy 2zz 2ww
	m7 = _1One_3Zeros; // 1 0 0 0
	m7 = _mm_sub_ss(m7,m0);// -2xx+1 0 0 0
	m7 = _mm_sub_ss(m7,m2);// -2yy-2xx+1 0 0 0
	
	m5 = _mm_load_ps((hq_float32*) QuatMask4);

	//first row
	m2 = _mm_xor_ps(m2,m5); // 2yy -2xy -2xz -2yz
	m4 = _mm_xor_ps(m4,_mm_load_ps((hq_float32*) QuatMask5)); // -2zz 2zw -2yw -2xw
	m4 = _mm_add_ss(m4,_1One_3Zeros); // -2zz+1 2zw -2yw -2xw
	m3 = _mm_sub_ps(m4,m2); // -2zz-2yy+1 2zw+2xy -2yw+2xz -2xw+2yz
	_mm_store_ps(matOut->m,m3);//hàng thứ 1
	matOut->_14 = 0.0f;
	
	//second row
	m2 = _mm_move_ss(m2,m0); //2xx -2xy -2xz -2yz
	m4 = _mm_xor_ps(m4,m5);//-2zz+1, -2zw, 2yw, 2xw
	m4 = _mm_sub_ps(m4,m2);// -2zz-2xx+1 2xy-2zw 2yw+2xz 2xw+2yz
	m4 = _mm_shuffle_ps(m4,m4,SSE_SHUFFLEL(1,0,3,2));//2xy-2zw -2zz-2xx+1 2xw+2yz 2yw+2xz
	_mm_store_ps(&matOut->m[4],m4);//hàng thứ 2
	matOut->_24 = 0.0f;

	//third row
	m3 = _mm_movehl_ps(m3,m4);//2xw+2yz 2yw+2xz -2yw+2xz -2xw+2yz
	m3 = _mm_shuffle_ps(m3,m7,SSE_SHUFFLEL(1,3,0,2));//2yw+2xz -2xw+2yz -2yy-2xx+1 0
	_mm_store_ps(&matOut->m[8],m3);//hàng thứ 3
	
	matOut->_41 = matOut->_42 = matOut->_43 = 0.0f;
	matOut->_44 =1.0f;
#endif
}

void HQQuaternion::QuatUnitToMatrix3x4c(HQMatrix3x4 *matOut)const{
#ifdef HQ_CMATH
	hq_float32 _2x=x+x;
	hq_float32 _2y=y+y;
	hq_float32 _2z=z+z;

	hq_float32 _2x2=_2x*x;
	hq_float32 _2y2=_2y*y;
	hq_float32 _2z2=_2z*z;
	hq_float32 _2xy=_2x*y;
	hq_float32 _2xz=_2x*z;
	hq_float32 _2yz=_2y*z;
	hq_float32 _2xw=_2x*w;
	hq_float32 _2yw=_2y*w;
	hq_float32 _2zw=_2z*w;

	matOut->_11=1-(_2y2+_2z2);
	matOut->_22=1-(_2x2+_2z2);
	matOut->_33=1-(_2x2+_2y2);
	matOut->_14=matOut->_24=matOut->_34=0.0f;

	matOut->_12=_2xy-_2zw;
	matOut->_13=_2xz+_2yw;

	matOut->_21=_2xy+_2zw;
	matOut->_23=_2yz-_2xw;

	matOut->_31=_2xz-_2yw;
	matOut->_32=_2yz+_2xw;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatUnitToMatrix3x4c(this->q, matOut->m);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatUnitToMatrix3x4c(this->q, matOut->m);

#else
	/*SSE*/
	float4 m0,m1,m2,m3,m4,m5,m7;
	m0 = _mm_load_ps(this->q);// x y z w
	m1 = _mm_add_ps(m0 , m0); //2x 2y 2z 2w
	
	m2 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,0,1));// y x x y
	m3 = hq_mm_copy_ps(m1,SSE_SHUFFLEL(1,1,2,2));// 2y 2y 2z 2z
	m2 = _mm_mul_ps(m2,m3);// 2yy 2xy 2xz 2yz

	m4 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,3,3));// z w w w
	m5 = hq_mm_copy_ps(m1,SSE_SHUFFLEL(2,2,1,0));// 2z 2z 2y 2x
	m4 = _mm_mul_ps(m4,m5);// 2zz 2zw 2yw 2xw

	m0 = _mm_mul_ps(m0,m1);// 2xx 2yy 2zz 2ww
	m7 = _1One_3Zeros; // 1 0 0 0
	m7 = _mm_sub_ss(m7,m0);// -2xx+1 0 0 0
	m7 = _mm_sub_ss(m7,m2);// -2yy-2xx+1 0 0 0
	
	m5 = _mm_load_ps((hq_float32*) QuatMask4);

	//first row
	m2 = _mm_xor_ps(m2,m5); // 2yy -2xy -2xz -2yz
	m4 = _mm_xor_ps(m4,_mm_load_ps((hq_float32*) QuatMask6)); // -2zz -2zw 2yw 2xw
	m4 = _mm_add_ss(m4,_1One_3Zeros); // -2zz+1 -2zw 2yw 2xw
	m3 = _mm_sub_ps(m4,m2); // -2zz-2yy+1 -2zw+2xy 2yw+2xz 2xw+2yz
	_mm_store_ps(matOut->m,m3);//hàng thứ 1
	matOut->_14 = 0.0f;
	
	//second row
	m2 = _mm_move_ss(m2,m0); //2xx -2xy -2xz -2yz
	m4 = _mm_xor_ps(m4,m5);//-2zz+1, 2zw, -2yw, -2xw
	m4 = _mm_sub_ps(m4,m2);// -2zz-2xx+1 2xy+2zw -2yw+2xz -2xw+2yz
	m4 = _mm_shuffle_ps(m4,m4,SSE_SHUFFLEL(1,0,3,2));//2xy+2zw -2zz-2xx+1 -2xw+2yz -2yw+2xz
	_mm_store_ps(&matOut->m[4],m4);//hàng thứ 2
	matOut->_24 = 0.0f;

	//third row
	m3 = _mm_movehl_ps(m3,m4);//-2xw+2yz -2yw+2xz 2yw+2xz 2xw+2yz
	m3 = _mm_shuffle_ps(m3,m7,SSE_SHUFFLEL(1,3,0,2));//-2yw+2xz 2xw+2yz -2yy-2xx+1 0
	_mm_store_ps(&matOut->m[8],m3);//hàng thứ 3
	
#endif
}

#ifdef HQ_SSE_MATH


#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#define ALIGN4_INIT1( X, I ) static const HQ_ALIGN16 X[4] = { I, I, I, I }
#else
#define ALIGN4_INIT1( X, I ) static const X[4] HQ_ALIGN16_POSTFIX = { I, I, I, I }
#endif 

ALIGN4_INIT1( hq_uint32 SIMD_DW_mat2quatShuffle0, (3<<0)|(2<<8)|(1<<16)|(0<<24) );
ALIGN4_INIT1( hq_uint32 SIMD_DW_mat2quatShuffle1, (0<<0)|(1<<8)|(2<<16)|(3<<24) );
ALIGN4_INIT1( hq_uint32 SIMD_DW_mat2quatShuffle2, (1<<0)|(0<<8)|(3<<16)|(2<<24) ); 
ALIGN4_INIT1( hq_uint32 SIMD_DW_mat2quatShuffle3, (2<<0)|(3<<8)|(0<<16)|(1<<24) );

#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
struct HQ_ALIGN16 A16_Bytes
#else
typedef struct A16_Bytes
#endif
{
	unsigned char bytes[16];
	unsigned char& operator[] (hq_int32 i)
	{
		return bytes[i];
	}
#if defined LINUX || defined HQ_MAC_PLATFORM || defined ANDROID || defined HQ_IPHONE_PLATFORM
}  A16_Bytes HQ_ALIGN16 ;
#else
};
#endif

#endif

void HQQuaternion::QuatFromMatrix4r(const HQMatrix4 &m)
{
#ifdef HQ_CMATH
	//normal version
	hq_float32 s0, s1, s2;
	hq_int32 k0, k1, k2, k3;
	if ( m(0,0) + m(1,1) + m(2,2) > 0.0f ) {
		k0 = 3;
		k1 = 2;
		k2 = 1;
		k3 = 0;
		s0 = 1.0f;
		s1 = 1.0f;
		s2 = 1.0f;
	} else if ( m(0,0) > m(1,1) && m(0,0) > m(2,2) ) {
		k0 = 0;
		k1 = 1;
		k2 = 2;
		k3 = 3;
		s0 = 1.0f;
		s1 = -1.0f;
		s2 = -1.0f;
	} else if ( m(1,1) > m(2,2) ) {
		k0 = 1;
		k1 = 0;
		k2 = 3;
		k3 = 2;
		s0 = -1.0f;
		s1 = 1.0f;
		s2 = -1.0f;
	} else {
		k0 = 2;
		k1 = 3;
		k2 = 0;
		k3 = 1;
		s0 = -1.0f;
		s1 = -1.0f;
		s2 = 1.0f;
	}
	hq_float32 t = s0 * m(0,0) + s1 * m(1,1) + s2 * m(2,2) + 1.0f;
	hq_float32 s = 1.0f / sqrt(t) * 0.5f;
	q[k0] = s * t;
	q[k1] = ( m(0,1) - s2 * m(1,0) ) * s;
	q[k2] = ( m(2,0) - s1 * m(0,2) ) * s;
	q[k3] = ( m(1,2) - s0 * m(2,1) ) * s;
	
#elif defined HQ_NEON_MATH
	
	HQNeonQuatFromMatrix4r(m, this->q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatFromMatrix4r(m, this->q);
	
#else
	/*SSE*/
	float4 m0,m1,m2,m3,m4,m5,m6,m7;

	m5 = _mm_load_ps1(&m(0,0));// 11 11 11 11
	m6 = _mm_load_ps1(&m(1,1));// 22 22 22 22
	m7 = _mm_load_ps1(&m(2,2));// 33 33 33 33
	
	m0 = _mm_add_ps(m5,m6);
	m0 = _mm_add_ps(m0,m7);//  11 + 22 + 33 , _ , _ , _
	m0 = _mm_cmpnlt_ps( m0 , _4Zeros);  // (11 + 22 + 33) >= 0

	m1 = _mm_cmpnlt_ps( m5, m6) ;// 11 >= 22
	m2 = _mm_cmpnlt_ps( m5, m7) ;// 11 >= 33
	m2 = _mm_and_ps (m2,m1); // max = 11

	m4 = _mm_cmpnlt_ps(m6 , m7);// 22 >= 33
	
	m1 = _mm_andnot_ps(m0,m2) ; // (11 + 22 + 33) < 0 && max = 11
	m2 = _mm_or_ps(m2,m0); //(11 + 22 + 33) >= 0 || max = 11
	m3 = m2;
	m2 = _mm_andnot_ps(m2 , m4); //!((11 + 22 + 33) >= 0 || max = 11) && 22 >= 33
	m3 = _mm_or_ps(m3,m2);
	m4 = _mm_load_ps((hq_float32*) NotMask);
	m3 = _mm_xor_ps(m3, m4) ;//else

	A16_Bytes shuffle;
	m0 = _mm_and_ps(m0 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle0));
	m4 = _mm_and_ps(m1 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle1));
	m0 = _mm_or_ps(m0,m4);
	m4 = _mm_and_ps(m2 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle2));
	m0 = _mm_or_ps(m0,m4);
	m4 = _mm_and_ps(m3 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle3));
	m4 = _mm_or_ps(m4,m0);

	_mm_store_ps((hq_float32*)shuffle.bytes,m4);


	m0 = _mm_or_ps(m2,m3);//(max = 22 || max = 33) && sum < 0
	m2 = _mm_or_ps(m2,m1);//(max = 11 || max = 22) && sum < 0
	m1 = _mm_or_ps(m1,m3);//(max = 11 || max = 33) && sum < 0
	
	float4 mmNegSignMask = _mm_load_ps((hq_float32*) NegSignMask);

	m0 = _mm_and_ps(m0, mmNegSignMask);//s0
	m1 = _mm_and_ps(m1, mmNegSignMask);//s1
	m2 = _mm_and_ps(m2, mmNegSignMask);//s2

	m5 = _mm_xor_ps(m5,m0);
	m6 = _mm_xor_ps(m6,m1);
	m7 = _mm_xor_ps(m7,m2);
	m4 = _4Ones;
	m5 = _mm_add_ps(m5,m6);
	m7 = _mm_add_ps(m7,m4);
	m5 = _mm_add_ps(m5,m7);//s0 ^ 11 + s1 ^ 22 + s2 ^ 33 + 1.0f = t

	m4=_mm_rsqrt_ps(m5);//	tính gần đúng 1/căn(t) ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	//Newton Raphson iteration gần đúng căn x = y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	//m6 = t(n+1) * 0.5 = 1/4 *(y(n)*(3-x*y(n)^2)) = s
	m6=_mm_mul_ps(_mm_mul_ps(_4Quarts,m4),_mm_sub_ps(_4Threes,_mm_mul_ps(m5,_mm_mul_ps(m4,m4))));
	
	m7 = _mm_mul_ps(m5,m6);//s * t

	_mm_store_ss(&q[shuffle[0]] , m7); // q[k0] = s* t

	m4 = _mm_load_ss(&m(0,1));
	m5 = _mm_load_ss(&m(1,0));
	m5 = _mm_xor_ps(m5,m2);
	m4 = _mm_sub_ss(m4,m5);
	m4 = _mm_mul_ss(m4,m6);
	_mm_store_ss(&q[shuffle[1]] , m4);// q[k1] = ( m(0,1) - s2 ^ m(1,0) ) * s;

	m4 = _mm_load_ss(&m(2,0));
	m5 = _mm_load_ss(&m(0,2));
	m5 = _mm_xor_ps(m5,m1);
	m4 = _mm_sub_ss(m4,m5);
	m4 = _mm_mul_ss(m4,m6);
	_mm_store_ss(&q[shuffle[2]] , m4);// q[k2] = ( m(2,0) - s1 ^ m(0,2) ) * s;

	m4 = _mm_load_ss(&m(1,2));
	m5 = _mm_load_ss(&m(2,1));
	m5 = _mm_xor_ps(m5,m0);
	m4 = _mm_sub_ss(m4,m5);
	m4 = _mm_mul_ss(m4,m6);
	_mm_store_ss(&q[shuffle[3]] , m4);// q[k3] = ( m(1,2) - s0 ^ m(2,1) ) * s;
#endif
}

void HQQuaternion::QuatFromMatrix3x4c(const HQMatrix3x4 &m)
{
#ifdef HQ_CMATH
	//normal version
	hq_float32 s0, s1, s2;
	hq_int32 k0, k1, k2, k3;
	if ( m(0,0) + m(1,1) + m(2,2) > 0.0f ) {
		k0 = 3;
		k1 = 2;
		k2 = 1;
		k3 = 0;
		s0 = 1.0f;
		s1 = 1.0f;
		s2 = 1.0f;
	} else if ( m(0,0) > m(1,1) && m(0,0) > m(2,2) ) {
		k0 = 0;
		k1 = 1;
		k2 = 2;
		k3 = 3;
		s0 = 1.0f;
		s1 = -1.0f;
		s2 = -1.0f;
	} else if ( m(1,1) > m(2,2) ) {
		k0 = 1;
		k1 = 0;
		k2 = 3;
		k3 = 2;
		s0 = -1.0f;
		s1 = 1.0f;
		s2 = -1.0f;
	} else {
		k0 = 2;
		k1 = 3;
		k2 = 0;
		k3 = 1;
		s0 = -1.0f;
		s1 = -1.0f;
		s2 = 1.0f;
	}
	hq_float32 t = s0 * m(0,0) + s1 * m(1,1) + s2 * m(2,2) + 1.0f;
	hq_float32 s = 1.0f / sqrt(t) * 0.5f;
	q[k0] = s * t;
	q[k1] = ( m(1,0) - s2 * m(0,1) ) * s;
	q[k2] = ( m(0,2) - s1 * m(2,0) ) * s;
	q[k3] = ( m(2,1) - s0 * m(1,2) ) * s;
	
#elif defined HQ_NEON_MATH
	
	HQNeonQuatFromMatrix3x4c(m, this->q);
	
#elif defined HQ_DIRECTX_MATH

	HQDXQuatFromMatrix3x4c(m, this->q);

#else
	/*SSE*/
	TRACE("HERE %s, %d", __FILE__, __LINE__);

	float4 m0,m1,m2,m3,m4,m5,m6,m7;

	m5 = _mm_load_ps1(&m(0,0));// 11 11 11 11
	m6 = _mm_load_ps1(&m(1,1));// 22 22 22 22
	m7 = _mm_load_ps1(&m(2,2));// 33 33 33 33
	
	m0 = _mm_add_ps(m5,m6);
	m0 = _mm_add_ps(m0,m7);//  11 + 22 + 33 , _ , _ , _
	m0 = _mm_cmpnlt_ps( m0 , _4Zeros);  // (11 + 22 + 33) >= 0

	m1 = _mm_cmpnlt_ps( m5, m6) ;// 11 >= 22
	m2 = _mm_cmpnlt_ps( m5, m7) ;// 11 >= 33
	m2 = _mm_and_ps (m2,m1); // max = 11

	m4 = _mm_cmpnlt_ps(m6 , m7);// 22 >= 33
	
	m1 = _mm_andnot_ps(m0,m2) ; // (11 + 22 + 33) < 0 && max = 11
	m2 = _mm_or_ps(m2,m0); //(11 + 22 + 33) >= 0 || max = 11
	m3 = m2;
	m2 = _mm_andnot_ps(m2 , m4); //!((11 + 22 + 33) >= 0 || max = 11) && 22 >= 33
	m3 = _mm_or_ps(m3,m2);
	m4 = _mm_load_ps((hq_float32*) NotMask);
	m3 = _mm_xor_ps(m3, m4) ;//else
	
	TRACE("HERE %s, %d", __FILE__, __LINE__);

	A16_Bytes shuffle;
	m0 = _mm_and_ps(m0 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle0));
	m4 = _mm_and_ps(m1 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle1));
	m0 = _mm_or_ps(m0,m4);
	m4 = _mm_and_ps(m2 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle2));
	m0 = _mm_or_ps(m0,m4);
	m4 = _mm_and_ps(m3 , _mm_load_ps((hq_float32*)SIMD_DW_mat2quatShuffle3));
	m4 = _mm_or_ps(m4,m0);

	_mm_store_ps((hq_float32*)shuffle.bytes,m4);

	
	TRACE("HERE %s, %d", __FILE__, __LINE__);

	m0 = _mm_or_ps(m2,m3);//(max = 22 || max = 33) && sum < 0
	m2 = _mm_or_ps(m2,m1);//(max = 11 || max = 22) && sum < 0
	m1 = _mm_or_ps(m1,m3);//(max = 11 || max = 33) && sum < 0
	
	float4 mmNegSignMask = _mm_load_ps((hq_float32*) NegSignMask);
	m0 = _mm_and_ps(m0, mmNegSignMask);//s0
	m1 = _mm_and_ps(m1, mmNegSignMask);//s1
	m2 = _mm_and_ps(m2, mmNegSignMask);//s2

	m5 = _mm_xor_ps(m5,m0);
	m6 = _mm_xor_ps(m6,m1);
	m7 = _mm_xor_ps(m7,m2);
	m4 = _4Ones;
	m5 = _mm_add_ps(m5,m6);
	m7 = _mm_add_ps(m7,m4);
	m5 = _mm_add_ps(m5,m7);//s0 ^ 11 + s1 ^ 22 + s2 ^ 33 + 1.0f = t

	m4=_mm_rsqrt_ps(m5);//	tính gần đúng 1/căn(t) ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	//Newton Raphson iteration gần đúng căn x = y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	//m6 = t(n+1) * 0.5 = 1/4 *(y(n)*(3-x*y(n)^2)) = s
	m6=_mm_mul_ps(_mm_mul_ps(_4Quarts,m4),_mm_sub_ps(_4Threes,_mm_mul_ps(m5,_mm_mul_ps(m4,m4))));
	
	m7 = _mm_mul_ps(m5,m6);//s * t

	_mm_store_ss(&q[shuffle[0]] , m7); // q[k0] = s* t

	m4 = _mm_load_ss(&m(1,0));
	m5 = _mm_load_ss(&m(0,1));
	m5 = _mm_xor_ps(m5,m2);
	m4 = _mm_sub_ss(m4,m5);
	m4 = _mm_mul_ss(m4,m6);
	_mm_store_ss(&q[shuffle[1]] , m4);// q[k1] = ( m(1,0) - s2 ^ m(0,1) ) * s;

	m4 = _mm_load_ss(&m(0,2));
	m5 = _mm_load_ss(&m(2,0));
	m5 = _mm_xor_ps(m5,m1);
	m4 = _mm_sub_ss(m4,m5);
	m4 = _mm_mul_ss(m4,m6);
	_mm_store_ss(&q[shuffle[2]] , m4);// q[k2] = ( m(0,2) - s1 ^ m(2,0) ) * s;

	m4 = _mm_load_ss(&m(2,1));
	m5 = _mm_load_ss(&m(1,2));
	m5 = _mm_xor_ps(m5,m0);
	m4 = _mm_sub_ss(m4,m5);
	m4 = _mm_mul_ss(m4,m6);
	_mm_store_ss(&q[shuffle[3]] , m4);// q[k3] = ( m(2,1) - s0 ^ m(1,2) ) * s;

	
	TRACE("HERE %s, %d", __FILE__, __LINE__);
#endif
}

HQQuaternion& HQQuaternion::Slerp(const HQQuaternion& quat1,const HQQuaternion& quat2,hq_float32 t)//nội suy giữa 2 quaternion,kết quả lưu trong đối tượng này
{
	hq_float32 Cosf = quat1.Dot(quat2);
	hq_float32 d1 = 1.0f - t;
	if(Cosf < 0.0f)
	{
		Cosf = -Cosf;
		HQ_DECL_STACK_QUATERNION_CTOR_PARAMS( quat, (-quat2.x , -quat2.y , -quat2.z , -quat2.w));
		if((1.0f - Cosf) < EPSILON)//góc quá nhỏ
		{
#ifdef HQ_NEON_MATH
			//prevent alignment problem in gcc
			HQNeonQuatMultiplyScalar(quat1.q, d1,this->q);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQNeonQuatMultiplyScalar(quat.q, t, temp.q);

			*this += temp;
			
#else
			//*this = d1 * quat1;
			//*this += t * quat;

			HQQuatMultiply(&quat1, d1,this);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat, t, &temp);

			*this += temp;

#endif
		}
		else
		{
			hq_float32 angle = acosf(Cosf);
			hq_float32 Sinf = 1.0f / sinf(angle);
			d1 = sinf(d1 * angle);
			hq_float32 d2 = sinf(t * angle);
#ifdef HQ_NEON_MATH
			//prevent alignment problem in gcc
			HQNeonQuatMultiplyScalar(quat1.q, d1,this->q);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQNeonQuatMultiplyScalar(quat.q, d2, temp.q);

			*this += temp;
			*this *= Sinf;
			
#else
			//*this = d1 * quat1;
			//*this += d2 * quat;
			HQQuatMultiply(&quat1, d1,this);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat, d2, &temp);

			*this += temp;


			*this *= Sinf;
#endif
		}
	}
	else
	{
		if((1.0f - Cosf) < EPSILON)//góc quá nhỏ
		{
#ifdef HQ_NEON_MATH
			//prevent alignment problem in gcc
			HQNeonQuatMultiplyScalar(quat1.q, d1,this->q);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQNeonQuatMultiplyScalar(quat2.q, t, temp.q);

			*this += temp;
			
#else
			//*this = d1 * quat1;
			//*this += t * quat2;
			HQQuatMultiply(&quat1, d1,this);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat2, t, &temp);

			*this += temp;
#endif
		}
		else
		{
			hq_float32 angle = acosf(Cosf);
			hq_float32 Sinf = 1.0f / sinf(angle);
			d1 = sinf(d1 * angle);
			hq_float32 d2 = sinf(t * angle);
#ifdef HQ_NEON_MATH
			//prevent alignment problem in gcc
			HQNeonQuatMultiplyScalar(quat1.q, d1,this->q);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQNeonQuatMultiplyScalar(quat2.q, d2, temp.q);

			*this += temp;
			*this *= Sinf;
			
#else
			//*this = d1 * quat1;
			//*this += d2 * quat2;
			HQQuatMultiply(&quat1, d1,this);
			
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat2, d2, &temp);

			*this += temp;

			*this *= Sinf;
#endif
		}
	}
	return *this;
}
HQQuaternion& HQQuaternion::Slerp(const HQQuaternion& quat2,hq_float32 t)//nội suy giữa quaternion này và quaternion <quat2>
{
	hq_float32 Cosf = this->Dot(quat2);;
	hq_float32 d1 = 1.0f - t;
	if(Cosf < 0.0f)
	{
		Cosf = -Cosf;
		HQ_DECL_STACK_QUATERNION_CTOR_PARAMS( quat, (-quat2.x , -quat2.y , -quat2.z , -quat2.w));
		if((1.0f - Cosf) < EPSILON)//góc quá nhỏ
		{
			//((*this)*= d1) += (t * quat);
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat, t, &temp);
			
			((*this)*= d1) += temp;
		}
		else
		{
			hq_float32 angle = acosf(Cosf);
			hq_float32 Sinf = 1.0f / sinf(angle);
			d1 = sinf(d1 * angle);
			hq_float32 d2 = sinf(t * angle);

			//(((*this)*= d1) += (d2 * quat)) *= Sinf;
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat, d2, &temp);
			(((*this)*= d1) += temp) *= Sinf;
		}
	}
	else
	{
		if((1.0f - Cosf) < EPSILON)//góc quá nhỏ
		{
			//((*this)*= d1) += (t * quat2);
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat2, t, &temp);
			
			((*this)*= d1) += temp;
		}
		else
		{
			hq_float32 angle = acosf(Cosf);
			hq_float32 Sinf = 1.0f / sinf(angle);
			d1 = sinf(d1 * angle);
			hq_float32 d2 = sinf(t * angle);

			//(((*this)*= d1) += (d2 * quat2)) *= Sinf;
			HQ_DECL_STACK_QUATERNION_CTOR_PARAMS(temp, (NULL));
			HQQuatMultiply(&quat2, d2, &temp);
			(((*this)*= d1) += temp) *= Sinf;
		}
	}
	return *this;
}

//***************************************
//tạo quat từ các góc quay yaw pitch roll
//***************************************
void HQQuaternion::QuatFromRollPitchYaw(hq_float32 roll, hq_float32 pitch, hq_float32 yaw){
	roll*=0.5f;
	pitch*=0.5f;
	yaw*=0.5f;

	hq_float32 rC,pC,yC,rS,pS,yS;
	HQSincosf(roll,&rS,&rC);
	HQSincosf(pitch,&pS,&pC);
	HQSincosf(yaw,&yS,&yC);

	hq_float32 pCyC=pC*yC;
	hq_float32 pSyS=pS*yS;
	hq_float32 pCyS=pC*yS;
	hq_float32 pSyC=pS*yC;

	x=rC*pSyC+rS*pCyS;
	w=rC*pCyC+rS*pSyS;

	y=rC*pCyS-rS*pSyC;
	z=rS*pCyC-rC*pSyS;
}
//********************************************************************************************************
//tạo quaternion phép biến đổi quay quanh trục axis góc angle
//********************************************************************************************************
void HQQuaternion::QuatFromRotAxis(hq_float32 angle, HQVector4 &axis){
	axis.Normalize();
	hq_float32 sin;
	HQSincosf(angle*0.5f,&sin,&w);
	x=axis.x*sin;
	y=axis.y*sin;
	z=axis.z*sin;
}
//********************************************************************************************************
//tạo quaternion phép biến đổi quay quanh trục axis góc angle,vector chỉ phương của trục phải đã chuẩn hóa
//********************************************************************************************************
void HQQuaternion::QuatFromRotAxisUnit(hq_float32 angle, const HQVector4 &axis){
	hq_float32 sin;
	HQSincosf(angle*0.5f,&sin,&w);
	x=axis.x*sin;
	y=axis.y*sin;
	z=axis.z*sin;
}

void HQQuaternion::QuatFromRotAxisOx(hq_float32 angle){
	HQSincosf(angle*0.5f,&x,&w);
	y=0.0f;
	z=0.0f;
}

void HQQuaternion::QuatFromRotAxisOy(hq_float32 angle){
	HQSincosf(angle*0.5f,&y,&w);
	x=0.0f;
	z=0.0f;
}

void HQQuaternion::QuatFromRotAxisOz(hq_float32 angle){
	HQSincosf(angle*0.5f,&z,&w);
	x=0.0f;
	y=0.0f;
}

//********************************************
//truy vấn góc quay và trục quay từ quaternion
//********************************************
void HQQuaternion::QuatToRotAxis(hq_float32* pAngle,HQVector4* pAxis){
	this->Normalize();
#ifdef HQ_CMATH
	hq_float32 f = 1.0f / sqrtf(x * x + y * y + z * z);
	pAxis->x = x * f;
	pAxis->y = y * f;
	pAxis->z = z * f;
	
	pAxis->w=0.0f;
	
#elif defined HQ_NEON_MATH
	
	HQNeonQuatUnitToRotAxis(this->q, pAxis->v);
	
#elif defined HQ_DIRECTX_MATH

	HQDXQuatUnitToRotAxis(this->q, pAxis->v);

#else
	float4 m0,m1,m2,m3;
	m0=_mm_load_ps(q);//copy quat data to 16 bytes(128 bit) xmm register
	m2=m0;
	m0=_mm_mul_ps(m0,m0);//nhân quat data với chính nó x^2 y^2 z^2 w^2

	m1=hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,0,0));//	y^2 x^2 x^2 x^2
	m3=hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,1,0));//	z^2 z^2 y^2 x^2 

	m0=_mm_add_ps(m0,m1);//x^2+y^2		y^2+x^2		z^2+x^2		x^2+x^2
	m0=_mm_add_ps(m0,m3);//x^2+y^2+z^2		x^2+y^2+z^2		x^2+y^2+z^2		x^2+x^2+x^2   

	float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_4Halves,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));

	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài

	_mm_store_ps(pAxis->v,m2);
	
	pAxis->w=0.0f;
#endif

	*pAngle=2.0f*acosf(w);
}
//*********************************************************
//truy vấn góc quay và trục quay từ quaternion đã chuẩn hóa
//*********************************************************
void HQQuaternion::QuatUnitToRotAxis(hq_float32* pAngle,HQVector4* pAxis)const{
#ifdef HQ_CMATH
	hq_float32 f = 1.0f / sqrtf(x * x + y * y + z * z);
	pAxis->x = x * f;
	pAxis->y = y * f;
	pAxis->z = z * f;
	pAxis->w=0.0f;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatUnitToRotAxis(this->q, pAxis->v);
	
#elif defined HQ_DIRECTX_MATH

	HQDXQuatUnitToRotAxis(this->q, pAxis->v);

#else
	float4 m0,m1,m2,m3;
	m0=_mm_load_ps(q);//copy quat data to 16 bytes(128 bit) xmm register
	m2=m0;
	m0=_mm_mul_ps(m0,m0);//nhân quat data với chính nó x^2 y^2 z^2 w^2

	m1=hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,0,0));//	y^2 x^2 x^2 x^2
	m3=hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,1,0));//	z^2 z^2 y^2 x^2 

	m0=_mm_add_ps(m0,m1);//x^2+y^2		y^2+x^2		z^2+x^2		x^2+x^2
	m0=_mm_add_ps(m0,m3);//x^2+y^2+z^2		x^2+y^2+z^2		x^2+y^2+z^2		x^2+x^2+x^2   

	float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_4Halves,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));

	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài

	_mm_store_ps(pAxis->v,m2);
	pAxis->w=0.0f;
#endif

	*pAngle=2.0f*acosf(w);
}
//*************************************
//sqrt(x^2+y^2+z^2+w^2)
//*************************************
hq_float32 HQQuaternion::GetMagnitude(){
#ifdef HQ_CMATH
	return sqrtf(x * x + y * y + z * z + w * w);
#elif defined HQ_NEON_MATH
	
	return HQNeonQuatMagnitude(this->q);

#elif defined HQ_DIRECTX_MATH

	return HQDXQuatMagnitude(this->q);

#else
	float4 m0=_mm_load_ps(q);//copy quat data to 128 bit xmm register
	m0=_mm_mul_ps(m0,m0);//nhân quat data với chính nó=> x*x y*y z*z w*w
	float4 m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//z*z w*w x*x y*y
	m0=_mm_add_ps(m0,m1);//=> x*x+z*z, y*y+w*w, z*z+x*x, w*w+y*y
	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y*y+w*w, x*x+z*z, y*y+w*w, x*x+z*z
	m0=_mm_add_ps(m0,m1);//=> x*x+z*z+y*y+z*z+w*w, _, _, _
	m0=_mm_sqrt_ss(m0);//căn phần tử đầu

	hq_float32 result;
	_mm_store_ss(&result,m0);
	return result;
#endif
}
//*************************************
//(x^2+y^2+z^2+w^2)
//*************************************
hq_float32 HQQuaternion::GetSumSquares(){
#ifdef HQ_CMATH
	return x * x + y * y + z * z + w * w;
#elif defined HQ_NEON_MATH
	
	return HQNeonQuatSumSquares(this->q);

#elif defined HQ_DIRECTX_MATH

	return HQDXQuatSumSquares(this->q);

#else
	float4 m0=_mm_load_ps(q);//copy quat data to 128 bit xmm register
	m0=_mm_mul_ps(m0,m0);//nhân quat data với chính nó=> x*x y*y z*z w*w
	float4 m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//z*z w*w x*x y*y
	m0=_mm_add_ps(m0,m1);//=> x*x+z*z, y*y+w*w, z*z+x*x, w*w+y*y
	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y*y+w*w, x*x+z*z, y*y+w*w, x*x+z*z
	m0=_mm_add_ps(m0,m1);//=> x*x+z*z+y*y+z*z+w*w, _, _, _

	hq_float32 result;
	_mm_store_ss(&result,m0);
	return result;
#endif
}
//***************************************
//phép nhân
//***************************************

HQQuaternion& HQQuaternion::operator *=(const HQQuaternion &quat){
#ifdef HQ_CMATH
	hq_float32 _x,_y,_z;

	_x=w*quat.x+x*quat.w+y*quat.z-z*quat.y;
	_y=w*quat.y+y*quat.w+z*quat.x-x*quat.z;
	_z=w*quat.z+z*quat.w+x*quat.y-y*quat.x;
	w=w*quat.w-x*quat.x-y*quat.y-z*quat.z;

	x=_x;
	y=_y;
	z=_z;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatMultiply(this->q, quat.q, this->q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatMultiply(this->q, quat.q, this->q);
	
#else	
	/*SSE*/

	float4 m0,m1,m2,m3,m4,m5,m6,m7;
	m0=_mm_load_ps(q);//x1 y1 z1 w1
	m4=_mm_load_ps(quat.q);//x2 y2 z2 w2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(0,0,0,0));//x1 x1 x1 x1
	m2 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,1,1,1));//y1 y1 y1 y1
	m3 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,2,2));//z1 z1 z1 z1
	m0 = _mm_shuffle_ps(m0,m0,SSE_SHUFFLEL(3,3,3,3));//w1 w1 w1 w1

	m5 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(3,2,1,0));//w2 z2 y2 x2
	m6 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(2,3,0,1));//z2 w2 x2 y2
	m7 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(1,0,3,2));//y2 x2 w2 z2

	m0=_mm_mul_ps(m0,m4);//w1*x2 w1*y2 w1*z2 w1*w2
	m1=_mm_mul_ps(m1,m5);//x1*w2 x1*z2 x1*y2 x1*x2
	m2=_mm_mul_ps(m2,m6);//y1*z2 y1*w2 y1*x2 y1*y2
	m3=_mm_mul_ps(m3,m7);//z1*y2 z1*x2 z1*w2 z1*z2

	m4=_mm_load_ps((hq_float32*) QuatMask1);
	m5=_mm_load_ps((hq_float32*) QuatMask2);
	m6=_mm_load_ps((hq_float32*) QuatMask3);

	m1=_mm_xor_ps(m1,m4);//x1*w2 -x1*z2 x1*y2 -x1*x2
	m2=_mm_xor_ps(m2,m5);//y1*z2 y1*w2 -y1*x2 -y1*y2
	m3=_mm_xor_ps(m3,m6);//-z1*y2 z1*x2 z1*w2 -z1*z2
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(q,m0);
#endif
	return *this;
}

#ifndef HQ_EXPLICIT_ALIGN
HQQuaternion HQQuaternion::operator *(const HQQuaternion &quat) const{
	HQ_DECL_STACK_VAR( HQQuaternion , result);
#ifdef HQ_CMATH
	result.x=w*quat.x+x*quat.w+y*quat.z-z*quat.y;
	result.y=w*quat.y+y*quat.w+z*quat.x-x*quat.z;
	result.z=w*quat.z+z*quat.w+x*quat.y-y*quat.x;
	result.w=w*quat.w-x*quat.x-y*quat.y-z*quat.z;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatMultiply(this->q, quat.q, result.q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatMultiply(this->q, quat.q, result.q);

#else
	/*SSE*/

	float4 m0,m1,m2,m3,m4,m5,m6,m7;
	m0=_mm_load_ps(q);//x1 y1 z1 w1
	m4=_mm_load_ps(quat.q);//x2 y2 z2 w2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(0,0,0,0));//x1 x1 x1 x1
	m2 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,1,1,1));//y1 y1 y1 y1
	m3 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,2,2));//z1 z1 z1 z1
	m0 = _mm_shuffle_ps(m0,m0,SSE_SHUFFLEL(3,3,3,3));//w1 w1 w1 w1

	m5 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(3,2,1,0));//w2 z2 y2 x2
	m6 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(2,3,0,1));//z2 w2 x2 y2
	m7 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(1,0,3,2));//y2 x2 w2 z2

	m0=_mm_mul_ps(m0,m4);//w1*x2 w1*y2 w1*z2 w1*w2
	m1=_mm_mul_ps(m1,m5);//x1*w2 x1*z2 x1*y2 x1*x2
	m2=_mm_mul_ps(m2,m6);//y1*z2 y1*w2 y1*x2 y1*y2
	m3=_mm_mul_ps(m3,m7);//z1*y2 z1*x2 z1*w2 z1*z2

	m4=_mm_load_ps((hq_float32*) QuatMask1);
	m5=_mm_load_ps((hq_float32*) QuatMask2);
	m6=_mm_load_ps((hq_float32*) QuatMask3);

	m1=_mm_xor_ps(m1,m4);//x1*w2 -x1*z2 x1*y2 -x1*x2
	m2=_mm_xor_ps(m2,m5);//y1*z2 y1*w2 -y1*x2 -y1*y2
	m3=_mm_xor_ps(m3,m6);//-z1*y2 z1*x2 z1*w2 -z1*z2
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(result.q,m0);
#endif
	return result;
}

#endif//#ifndef HQ_EXPLICIT_ALIGN

HQQuaternion* HQQuatMultiply(const HQQuaternion* q1,const HQQuaternion* q2 , HQQuaternion* out)
{
#ifdef HQ_CMATH
	hq_float32 _x,_y,_z;

	_x=q1->w*q2->x+q1->x*q2->w+q1->y*q2->z-q1->z*q2->y;
	_y=q1->w*q2->y+q1->y*q2->w+q1->z*q2->x-q1->x*q2->z;
	_z=q1->w*q2->z+q1->z*q2->w+q1->x*q2->y-q1->y*q2->x;
	out->w=q1->w*q2->w-q1->x*q2->x-q1->y*q2->y-q1->z*q2->z;

	out->x=_x;
	out->y=_y;
	out->z=_z;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatMultiply(q1->q, q2->q, out->q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatMultiply(q1->q, q2->q, out->q);
	
#else	
	/*SSE*/

	float4 m0,m1,m2,m3,m4,m5,m6,m7;
	m0=_mm_load_ps(q1->q);//x1 y1 z1 w1
	m4=_mm_load_ps(q2->q);//x2 y2 z2 w2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(0,0,0,0));//x1 x1 x1 x1
	m2 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,1,1,1));//y1 y1 y1 y1
	m3 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,2,2,2));//z1 z1 z1 z1
	m0 = _mm_shuffle_ps(m0,m0,SSE_SHUFFLEL(3,3,3,3));//w1 w1 w1 w1

	m5 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(3,2,1,0));//w2 z2 y2 x2
	m6 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(2,3,0,1));//z2 w2 x2 y2
	m7 = hq_mm_copy_ps(m4,SSE_SHUFFLEL(1,0,3,2));//y2 x2 w2 z2

	m0=_mm_mul_ps(m0,m4);//w1*x2 w1*y2 w1*z2 w1*w2
	m1=_mm_mul_ps(m1,m5);//x1*w2 x1*z2 x1*y2 x1*x2
	m2=_mm_mul_ps(m2,m6);//y1*z2 y1*w2 y1*x2 y1*y2
	m3=_mm_mul_ps(m3,m7);//z1*y2 z1*x2 z1*w2 z1*z2

	m4=_mm_load_ps((hq_float32*) QuatMask1);
	m5=_mm_load_ps((hq_float32*) QuatMask2);
	m6=_mm_load_ps((hq_float32*) QuatMask3);

	m1=_mm_xor_ps(m1,m4);//x1*w2 -x1*z2 x1*y2 -x1*x2
	m2=_mm_xor_ps(m2,m5);//y1*z2 y1*w2 -y1*x2 -y1*y2
	m3=_mm_xor_ps(m3,m6);//-z1*y2 z1*x2 z1*w2 -z1*z2
	
	m0=_mm_add_ps(m0,m1);
	m2=_mm_add_ps(m2,m3);
	m0=_mm_add_ps(m0,m2);

	_mm_store_ps(out->q,m0);
#endif
	return out;
}
//**************************************
//chuẩn hóa
//**************************************
HQQuaternion& HQQuaternion::Normalize(){
#ifdef HQ_CMATH
	hq_float32 f = sqrtf(x * x + y * y + z * z + w * w);
#	ifdef INFCHECK	
	if(f < EPSILON)//too close to zero
	{
		hq_float32 w = 1.0f;
		*((hq_int32*)&w) |= 0x80000000 &  *((hq_int32*)&this->w);
		this->w = w;
		return *this;
	}
#	endif
	
	f = 1.0f / f;
	this->x *= f;
	this->y *= f;
	this->z *= f;
	this->w *= f;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatNormalize(this->q, this->q);	

#elif defined HQ_DIRECTX_MATH

	HQDXQuatNormalize(this->q, this->q);	
#else
	/*SSE*/
	float4 m0,m1,m2;
	m0=_mm_load_ps(this->q);//copy quat data to 16 bytes(128 bit) xmm register
	m2=m0;//copy m0 vào m2
	m0=_mm_mul_ps(m0,m0);//nhân quat với chính nó x^2 y^2 z^2 w^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//	z^2 w^2 x^2 y^2

	m0=_mm_add_ps(m0,m1);//x^2+z^2		y^2+w^2		z^2+x^2		w^2+y^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y^2+w^2		x^2+z^2		y^2+w^2		x^2+z^2

	m0=_mm_add_ps(m0,m1);//x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+x^2+x^2+w^2
#	ifdef INFCHECK	
	hq_float32 lengthSqr;
	_mm_store_ss(&lengthSqr , m0);
	if(lengthSqr < EPSILON)//too close to zero
	{
		hq_float32 w = 1.0f;
		*((hq_int32*)&w) |= 0x80000000 &  *((hq_int32*)&this->w);
		this->w = w;
		return *this;
	}
#	endif

	float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn

	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_4Halves,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));

	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài
	_mm_store_ps(this->q,m2);
#endif
	return *this;
}
HQQuaternion* HQQuatNormalize(const HQQuaternion* in,HQQuaternion* out)
{
#ifdef HQ_CMATH
	hq_float32 f = sqrtf(in->x * in->x + in->y * in->y + in->z * in->z + in->w * in->w);
#	ifdef INFCHECK	
	if(f < EPSILON)//too close to zero
	{
		out->x = in->x;
		out->y = in->y;
		out->z = in->z;
		hq_float32 w = 1.0f;
		*((hq_int32*)&w) |= 0x80000000 &  *((hq_int32*)&in->w);
		out->w = w;
		return *this;
	}
#	endif
	
	f = 1.0f / f;
	out->x = in->x * f;
	out->y = in->y * f;
	out->z = in->z * f;
	out->w = in->w * f;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatNormalize(in->q, out->q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatNormalize(in->q, out->q);
#else
	/*SSE*/
	float4 m0,m1,m2;
	m0=_mm_load_ps(in->q);//copy quat data to 16 bytes(128 bit) xmm register
	m2=m0;//copy m0 vào m2
	m0=_mm_mul_ps(m0,m0);//nhân quat với chính nó x^2 y^2 z^2 w^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//	z^2 w^2 x^2 y^2
	
	m0=_mm_add_ps(m0,m1);//x^2+z^2		y^2+w^2		z^2+x^2		w^2+y^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y^2+w^2		x^2+z^2		y^2+w^2		x^2+z^2 

	m0=_mm_add_ps(m0,m1);//x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+x^2+x^2+w^2   
	
#	ifdef INFCHECK
	hq_float32 lengthSqr;
	_mm_store_ss(&lengthSqr , m0);
	if(lengthSqr < EPSILON)//too close to zero
	{
		out->x = in->x;
		out->y = in->y;
		out->z = in->z;
		hq_float32 w = 1.0f;
		*((hq_int32*)&w) |= 0x80000000 &  *((hq_int32*)&in->w);
		out->w = w;
		return out;
	}
#	endif
	float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn
	
	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_4Halves,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));

	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài
	_mm_store_ps(out->q,m2);
#endif
	return out;
}
//*************************************
//dot product
//*************************************
hq_float32 HQQuaternion::Dot(const HQQuaternion& q2)const{
#ifdef HQ_CMATH
	return x * q2.x + y * q2.y + z * q2.z + w * q2.w;
#elif defined HQ_NEON_MATH
	
	return HQNeonQuatDot(this->q, q2.q);

#elif defined HQ_DIRECTX_MATH

	return HQDXQuatDot(this->q, q2.q);
#else /*SSE*/
	hq_float32 result;
	float4 m0=_mm_load_ps(q);
	float4 m1=_mm_load_ps(q2.q);

	m0=_mm_mul_ps(m0,m1);//x1*x2 y1*y2 z1*z2 w1*w2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//z1*z2 w1*w2 x1*x2 y1*y2

	m0=_mm_add_ps(m0,m1);//x1*x2+z1*z2 y1*y2+w1*w2 z1*z2+x1*x2 w1*w2+y1*y2
	
	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y1*y2+w1*w2 x1*x2+z1*z2 y1*y2+w1*w2 x1*x2+z1*z2

	m0=_mm_add_ps(m0,m1);//x1*x2+y1*y2+z1*z2+w1*w2, _, _, _

	_mm_store_ss(&result,m0);
	
	return result;
#endif
}
//*************************************
//Nghịch đảo
//*************************************

static const HQ_ALIGN16 hq_uint32 Quatmask[4]={0x80000000,0x80000000,0x80000000,0x00000000};//- - - +
HQQuaternion& HQQuaternion::Inverse(){

#ifdef HQ_CMATH
	hq_float32 f = 1.0f / (x * x + y * y + z * z + w * w);
	x = -x * f;
	y = -y * f;
	z = -z * f;
	w = w * f;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatInverse(this->q, this->q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatInverse(this->q, this->q);
#else /*SSE*/
	float4 m0=_mm_load_ps(this->q);
	float4 m1=_mm_load_ps((hq_float32*)Quatmask);

	float4 m2=_mm_xor_ps(m0,m1);//-x,-y,-z,w
	m0=_mm_mul_ps(m0,m0);//nhân quat với chính nó x^2 y^2 z^2 w^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//	z^2 w^2 x^2 y^2

	m0=_mm_add_ps(m0,m1);//x^2+z^2		y^2+w^2		z^2+x^2		w^2+y^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y^2+w^2		x^2+z^2		y^2+w^2		x^2+z^2

	m0=_mm_add_ps(m0,m1);//x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+x^2+x^2+w^2

	float4 tmp = _mm_rcp_ps(m0);//tính gần đúng 1/m0
	//Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 - giảm sai số
	m0  = _mm_sub_ps(_mm_add_ps(tmp, tmp), _mm_mul_ps(m0, _mm_mul_ps(tmp, tmp)));

	m0=_mm_mul_ps(m0,m2);

	_mm_store_ps(this->q,m0);
#endif
	return *this;
}
HQQuaternion* HQQuatInverse(const HQQuaternion* in,HQQuaternion* out){
#ifdef HQ_CMATH
	hq_float32 f = 1.0f / (in->x * in->x + in->y * in->y + in->z * in->z + in->w * in->w);
	out->x = -in->x * f;
	out->y = -in->y * f;
	out->z = -in->z * f;
	out->w = in->w * f;
#elif defined HQ_NEON_MATH
	
	HQNeonQuatInverse(in->q, out->q);

#elif defined HQ_DIRECTX_MATH

	HQDXQuatInverse(in->q, out->q);
#else /*SSE*/
	float4 m0=_mm_load_ps(in->q);
	float4 m1=_mm_load_ps((hq_float32*)Quatmask);
	
	float4 m2=_mm_xor_ps(m0,m1);//-x,-y,-z,w
	m0=_mm_mul_ps(m0,m0);//nhân quat với chính nó x^2 y^2 z^2 w^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(2,3,0,1));//	z^2 w^2 x^2 y^2
	
	m0=_mm_add_ps(m0,m1);//x^2+z^2		y^2+w^2		z^2+x^2		w^2+y^2

	m1 = hq_mm_copy_ps(m0,SSE_SHUFFLEL(1,0,1,0));//y^2+w^2		x^2+z^2		y^2+w^2		x^2+z^2 

	m0=_mm_add_ps(m0,m1);//x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+y^2+z^2+w^2		x^2+x^2+x^2+w^2 

	float4 tmp = _mm_rcp_ps(m0);//tính gần đúng 1/m0
	//Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 - giảm sai số
	m0  = _mm_sub_ps(_mm_add_ps(tmp, tmp), _mm_mul_ps(m0, _mm_mul_ps(tmp, tmp)));

	m0=_mm_mul_ps(m0,m2);

	_mm_store_ps(out->q,m0);
#endif
	return out;
}
