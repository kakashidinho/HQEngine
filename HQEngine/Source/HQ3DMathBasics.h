/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _3D_MATH_BASICS_H_
#define _3D_MATH_BASICS_H_

#include "HQUtilMathCommon.h"
#include "HQPrimitiveDataType.h"

struct HQBaseMatrix4;
struct HQBaseMatrix3x4;


//=======================================================
//Matrix 4x4 - default row major
//=======================================================
struct HQBaseMatrix4
{
	HQBaseMatrix4();//construct an identity matrix
	explicit HQBaseMatrix4(const void * null) {}//this constructor does nothing
	HQBaseMatrix4(hq_float32 _11, hq_float32 _12, hq_float32 _13, hq_float32 _14,
				hq_float32 _21, hq_float32 _22, hq_float32 _23, hq_float32 _24,
				hq_float32 _31, hq_float32 _32, hq_float32 _33, hq_float32 _34,
				hq_float32 _41, hq_float32 _42, hq_float32 _43, hq_float32 _44);

	HQBaseMatrix4(const HQBaseMatrix4 &matrix);
	HQBaseMatrix4(const HQBaseMatrix3x4 &matrix);//append (0,0,0,1) as last row

	operator hq_float32*() {return m;}//casting operator
	operator const hq_float32*() const {return m;}//casting operator

	union{
		struct{
			hq_float32 _11,_12,_13,_14,
					  _21,_22,_23,_24,
					  _31,_32,_33,_34,
					  _41,_42,_43,_44;
		};
		hq_float32 m[16];
		hq_float32 mt[4][4];
	};
};



//=======================================================
//Matrix 3x4
//=======================================================
struct HQBaseMatrix3x4
{
	HQBaseMatrix3x4();//construct an identity matrix
	explicit HQBaseMatrix3x4(const void * null) {}//this constructor does nothing
	HQBaseMatrix3x4(hq_float32 _11, hq_float32 _12, hq_float32 _13,hq_float32 _14,
				hq_float32 _21, hq_float32 _22, hq_float32 _23, hq_float32 _24,
				hq_float32 _31, hq_float32 _32, hq_float32 _33 ,hq_float32 _34);

	HQBaseMatrix3x4( const HQBaseMatrix3x4& src);

	
	operator hq_float32*() {return m;}//casting operator
	operator const hq_float32*() const {return m;}//casting operator

	union{
		struct{
			hq_float32 _11,_12,_13,_14,
					  _21,_22,_23,_24,
					  _31,_32,_33,_34;
		};
		hq_float32 m[12];
		hq_float32 mt[3][4];
	};
};

/*----------------Matrix4 implementation ---------------*/
HQ_FORCE_INLINE HQBaseMatrix4::HQBaseMatrix4()
	: _11(1.0f) , _12(0.0f) , _13(0.0f) , _14(0.0f) ,
	  _21(0.0f) , _22(1.0f) , _23(0.0f) , _24(0.0f) , 
	  _31(0.0f) , _32(0.0f) , _33(1.0f) , _34(0.0f) ,
	  _41(0.0f) , _42(0.0f) , _43(0.0f) , _44(1.0f)
{
}
HQ_FORCE_INLINE HQBaseMatrix4::HQBaseMatrix4(hq_float32 m11, hq_float32 m12, hq_float32 m13, hq_float32 m14, 
				hq_float32 m21, hq_float32 m22, hq_float32 m23, hq_float32 m24, 
				hq_float32 m31, hq_float32 m32, hq_float32 m33, hq_float32 m34, 
				hq_float32 m41, hq_float32 m42, hq_float32 m43, hq_float32 m44)
	: _11(m11) , _12(m12) , _13(m13) , _14(m14) ,
	  _21(m21) , _22(m22) , _23(m23) , _24(m24) , 
	  _31(m31) , _32(m32) , _33(m33) , _34(m34) ,
	  _41(m41) , _42(m42) , _43(m43) , _44(m44)
{
}

HQ_FORCE_INLINE HQBaseMatrix4::HQBaseMatrix4(const HQBaseMatrix3x4 &matrix)
	: _11(matrix._11) , _12(matrix._12) , _13(matrix._13) , _14(matrix._14) ,
	  _21(matrix._21) , _22(matrix._22) , _23(matrix._23) , _24(matrix._24) , 
	  _31(matrix._31) , _32(matrix._32) , _33(matrix._33) , _34(matrix._34) ,
	  _41(0.0f) , _42(0.0f) , _43(0.0f) , _44(1.0f)
{
}

HQ_FORCE_INLINE HQBaseMatrix4::HQBaseMatrix4(const HQBaseMatrix4 &matrix)
	: _11(matrix._11) , _12(matrix._12) , _13(matrix._13) , _14(matrix._14) ,
	  _21(matrix._21) , _22(matrix._22) , _23(matrix._23) , _24(matrix._24) , 
	  _31(matrix._31) , _32(matrix._32) , _33(matrix._33) , _34(matrix._34) ,
	  _41(matrix._41) , _42(matrix._42) , _43(matrix._43) , _44(matrix._44)
{
}

/*------------------Matrix3 implementation------------------*/
HQ_FORCE_INLINE HQBaseMatrix3x4::HQBaseMatrix3x4()
	: _11(1.0f) , _12(0.0f) , _13(0.0f), _14(0.0f),
	  _21(0.0f) , _22(1.0f) , _23(0.0f), _24(0.0f),
	  _31(0.0f) , _32(0.0f) , _33(1.0f) ,_34(0.0f) 
{
}
HQ_FORCE_INLINE HQBaseMatrix3x4::HQBaseMatrix3x4(hq_float32 m11, hq_float32 m12, hq_float32 m13, hq_float32 m14,
				hq_float32 m21, hq_float32 m22, hq_float32 m23, hq_float32 m24,
				hq_float32 m31, hq_float32 m32, hq_float32 m33, hq_float32 m34)
	: _11(m11) , _12(m12) , _13(m13), _14(m14),
	  _21(m21) , _22(m22) , _23(m23), _24(m24) ,
	  _31(m31) , _32(m32) , _33(m33), _34(m34)
{
}

HQ_FORCE_INLINE HQBaseMatrix3x4::HQBaseMatrix3x4(const HQBaseMatrix3x4 &matrix)
	: _11(matrix._11) , _12(matrix._12) , _13(matrix._13) , _14(matrix._14) ,
	  _21(matrix._21) , _22(matrix._22) , _23(matrix._23) , _24(matrix._24) , 
	  _31(matrix._31) , _32(matrix._32) , _33(matrix._33) , _34(matrix._34)
{
}


#endif