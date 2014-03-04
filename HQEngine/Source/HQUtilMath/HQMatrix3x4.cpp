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


//************************************************
//ma trận tịnh tiến
//************************************************
HQMatrix3x4* HQMatrix3x4cTranslate(hq_float32 x, hq_float32 y, hq_float32 z, HQMatrix3x4 *out){
	out->_11=out->_22=out->_33=1.0f;
	out->_12=out->_13=out->_21=out->_23=out->_31=out->_32=0.0f;
	out->_14=x;out->_24=y;out->_34=z;
	return out;
}

//***********************************************
//ma trận quay quanh Ox
//***********************************************
HQMatrix3x4* HQMatrix3x4cRotateX(hq_float32 angle, HQMatrix3x4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	out->_14 = out->_24 = out->_34 = 0.0f;
	out->_11=1.0f;
	out->_12=out->_13=out->_21=out->_31=0.0f;
	out->_22=fCos;
	out->_32=fSin;
	out->_23=-fSin;
	out->_33=fCos;
	return out;
}
//***********************************************
//ma trận quay quanh Oy
//***********************************************
HQMatrix3x4* HQMatrix3x4cRotateY(hq_float32 angle, HQMatrix3x4 *out ){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_14 = out->_24 = out->_34 = 0.0f;
	out->_22=1.0f;
	out->_12=out->_21=out->_23
		=out->_32=0.0f;
	out->_11=fCos;
	out->_31=-fSin;
	out->_13=fSin;
	out->_33=fCos;
	return out;
}
//***********************************************
//ma trận quay quanh Oz
//***********************************************
HQMatrix3x4* HQMatrix3x4cRotateZ(hq_float32 angle, HQMatrix3x4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);
	out->_14 = out->_24 = out->_34 = 0.0f;
	out->_33=1.0f;
	out->_13=out->_23=out->_31
		=out->_32=0.0f;
	out->_11=fCos;
	out->_21=fSin;
	out->_12=-fSin;
	out->_22=fCos;
	return out;
}
//******************************************************
//ma trận tỷ lệ
//******************************************************
HQMatrix3x4* HQMatrix3x4Scale(hq_float32 sx, hq_float32 sy, hq_float32 sz, HQMatrix3x4 *out){
	out->_11=sx;out->_22=sy;out->_33=sz;
	out->_12=out->_13=out->_21=out->_23=out->_31=
		out->_32=out->_14 = out->_24 = out->_34 = 0.0f;
	return out;
}

HQMatrix3x4* HQMatrix3x4Scale(hq_float32 s[3], HQMatrix3x4 *out){
	out->_11=s[0];out->_22=s[1];out->_33=s[2];
	out->_12=out->_13=out->_21=out->_23=out->_31=
		out->_32=out->_14 = out->_24 = out->_34 = 0.0f;
	return out;
}

//******************************************************
//nhân 2 ma trận
//******************************************************
static const HQ_ALIGN16 hq_int32 _3Zeros_1One_Masks[4]={0x00000000,0x00000000,0x00000000,0xffffffff};

HQMatrix3x4& HQMatrix3x4::operator *=(const HQMatrix3x4 &m){
	
#ifdef CMATH
	HQMatrix3x4 result( 0.0f , 0.0f , 0.0f , this->_14 ,
					0.0f , 0.0f , 0.0f , this->_24 ,
					0.0f , 0.0f , 0.0f , this->_34 );
	for (hq_int32 i = 0; i < 3 ; ++i)
	{
		for(hq_int32 j = 0; j < 4 ; ++j)
			for( hq_int32 k = 0 ; k < 3 ; ++k)
				result.mt[i][j] += mt[i][k] * m.mt[k][j];
	}
	memcpy(this, &result, sizeof(HQMatrix3x4));
	
#elif defined NEON_MATH
	
	HQNeonMatrix3x4Multiply(this->m , m , this->m);

#elif defined HQ_DIRECTX_MATH

	HQDXMatrix3x4Multiply(this->m, m, this->m);

#else
	float4 xmm[3],re,row , e ,masks;
	
	masks = _mm_load_ps((hq_float32*)_3Zeros_1One_Masks);
	//load 3 hàng của matix m
	xmm[0]=_mm_load_ps(&m.m[0]);
	xmm[1]=_mm_load_ps(&m.m[4]);
	xmm[2]=_mm_load_ps(&m.m[8]);
	
	//first row
	row=_mm_load_ps(this->m);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));
	
	row = _mm_and_ps(row , masks);// 0 0 0 m[3]
	re = _mm_add_ps(re , row);
	
	_mm_store_ps(this->m,re);

	//second row
	row=_mm_load_ps(&this->m[4]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));
	
	row = _mm_and_ps(row , masks);// 0 0 0 m[7]
	re = _mm_add_ps(re , row);
	
	_mm_store_ps(&this->m[4],re);

	//third row
	row=_mm_load_ps(&this->m[8]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));
	
	row = _mm_and_ps(row , masks);// 0 0 0 m[11]
	re = _mm_add_ps(re , row);
	
	_mm_store_ps(&this->m[8],re);
#endif

	return *this;
}

#ifndef HQ_EXPLICIT_ALIGN
HQMatrix3x4 HQMatrix3x4::operator *(const HQMatrix3x4 &m) const{
	
#ifdef CMATH
	HQMatrix3x4 result( 0.0f , 0.0f , 0.0f , this->_14 ,
					0.0f , 0.0f , 0.0f , this->_24 ,
					0.0f , 0.0f , 0.0f , this->_34 );
	for (hq_int32 i = 0; i < 3 ; ++i)
	{
		for(hq_int32 j = 0; j < 4 ; ++j)
			for( hq_int32 k = 0 ; k < 3 ; ++k)
				result.mt[i][j] += mt[i][k] * m.mt[k][j];
	}
	
#elif defined NEON_MATH
	HQA16ByteMatrix3x4Ptr pResult( NULL );
	HQMatrix3x4 &result = *pResult;
	
	HQNeonMatrix3x4Multiply(this->m , m , result);
	
#elif defined HQ_DIRECTX_MATH

	HQMatrix3x4 result;
	
	HQDXMatrix3x4Multiply(this->m , m , result);

#else

	HQMatrix3x4 result( NULL);

	float4 xmm[3],re,row ,e,masks;
	
	masks = _mm_load_ps((hq_float32*)_3Zeros_1One_Masks);
	//load 3 hàng của matix m
	xmm[0]=_mm_load_ps(&m.m[0]);
	xmm[1]=_mm_load_ps(&m.m[4]);
	xmm[2]=_mm_load_ps(&m.m[8]);

	//first row
	row=_mm_load_ps(this->m);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));
	
	row = _mm_and_ps(row , masks);// 0 0 0 m[3]
	re = _mm_add_ps(re , row);
	
	_mm_store_ps(result.m,re);

	//second row
	row=_mm_load_ps(&this->m[4]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));
	
	row = _mm_and_ps(row , masks);// 0 0 0 m[7]
	re = _mm_add_ps(re , row);
	
	_mm_store_ps(&result.m[4],re);

	//third row
	row=_mm_load_ps(&this->m[8]);
	e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
	re=_mm_mul_ps(e,xmm[0]);

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[1]));

	e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
	re=_mm_add_ps(re,_mm_mul_ps(e,xmm[2]));
	
	row = _mm_and_ps(row , masks);// 0 0 0 m[11]
	re = _mm_add_ps(re , row);
	
	_mm_store_ps(&result.m[8],re);
#endif
	return result;
}

#endif//#ifndef HQ_EXPLICIT_ALIGN

HQMatrix3x4* HQMatrix3x4Multiply(const HQMatrix3x4* pM1,const HQMatrix3x4* pM2,HQMatrix3x4* pOut){
#ifdef CMATH
	if (pOut == pM1 || pOut == pM2)
	{
		HQMatrix3x4 result( 0.0f , 0.0f , 0.0f , pM1->_14 ,
						0.0f , 0.0f , 0.0f , pM1->_24 ,
						0.0f , 0.0f , 0.0f , pM1->_34 );
		for (hq_int32 i = 0; i < 3 ; ++i)
		{
			for(hq_int32 j = 0; j < 4 ; ++j)
				for( hq_int32 k = 0 ; k < 3 ; ++k)
					result.mt[i][j] += pM1->mt[i][k] * pM2->mt[k][j];
		}
		memcpy(pOut, &result, sizeof(HQMatrix3x4));
	}
	else
	{
		pOut->_11 = pOut->_12 = pOut->_13 =
		pOut->_21 = pOut->_22 = pOut->_23 =
		pOut->_31 = pOut->_32 = pOut->_33 = 0.0f;
		pOut->_14 = pM1->_14;
		pOut->_24 = pM1->_24;
		pOut->_34 = pM1->_34;

		for (hq_int32 i = 0; i < 3 ; ++i)
		{
			for(hq_int32 j = 0; j < 4 ; ++j)
				for( hq_int32 k = 0 ; k < 3 ; ++k)
					pOut->mt[i][j] += pM1->mt[i][k] * pM2->mt[k][j];
		}
	}

#elif defined NEON_MATH
	
	HQNeonMatrix3x4Multiply(pM1->m, pM2->m, pOut->m);
	
#elif defined HQ_DIRECTX_MATH

	HQDXMatrix3x4Multiply(pM1->m, pM2->m, pOut->m);

#else
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
	
#endif
	return pOut;
}

HQ_UTIL_MATH_API HQMatrix3x4* HQMatrix3x4MultiMultiply(const HQMatrix3x4* pM, hq_uint32 numMatrices ,HQMatrix3x4* pOut)
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
		
		HQNeonMultiMatrix3x4Multiply(pM->m, numMatrices, pOut->m);
#elif defined HQ_DIRECTX_MATH

		HQDXMultiMatrix3x4Multiply(pM->m, numMatrices, pOut->m);
		
#else/*SSE*/
		float4 xmm[3] , reRow[3] , row , e , masks;
		masks = _mm_load_ps((hq_float32*)_3Zeros_1One_Masks);

		//load 3 hàng của matix 0
		reRow[0]=_mm_load_ps(&pM[0].m[0]);
		reRow[1]=_mm_load_ps(&pM[0].m[4]);
		reRow[2]=_mm_load_ps(&pM[0].m[8]);

		for (hq_uint32 i = 1 ; i < numMatrices ; ++i)
		{
			//load 3 hàng của matix i
			xmm[0]=_mm_load_ps(&pM[i].m[0]);
			xmm[1]=_mm_load_ps(&pM[i].m[4]);
			xmm[2]=_mm_load_ps(&pM[i].m[8]);

			//first row
			row = reRow[0];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[0]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[0]=_mm_add_ps(reRow[0],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[0]=_mm_add_ps(reRow[0],_mm_mul_ps(e,xmm[2]));

			row = _mm_and_ps(row , masks);// 0 0 0 m[3]
			reRow[0] = _mm_add_ps(reRow[0] , row);


			//second row
			row = reRow[1];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[1]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[1]=_mm_add_ps(reRow[1],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[1]=_mm_add_ps(reRow[1],_mm_mul_ps(e,xmm[2]));

			row = _mm_and_ps(row , masks);// 0 0 0 m[7]
			reRow[1] = _mm_add_ps(reRow[1] , row);

			//third row
			row = reRow[2];
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(0,0,0,0));
			reRow[2]=_mm_mul_ps(e,xmm[0]);
			
			e = hq_mm_copy_ps( row , SSE_SHUFFLE(1,1,1,1));
			reRow[2]=_mm_add_ps(reRow[2],_mm_mul_ps(e,xmm[1]));

			e = hq_mm_copy_ps( row , SSE_SHUFFLE(2,2,2,2));
			reRow[2]=_mm_add_ps(reRow[2],_mm_mul_ps(e,xmm[2]));

			row = _mm_and_ps(row , masks);// 0 0 0 m[11]
			reRow[2] = _mm_add_ps(reRow[2] , row);

		}

		_mm_store_ps(pOut->m , reRow[0]);
		_mm_store_ps(pOut->m + 4, reRow[1]);
		_mm_store_ps(pOut->m + 8, reRow[2]);
#endif
	}
	return pOut;
}

//**********************************************************
//ma trận quay quanh trục bất kỳ
//**********************************************************
HQMatrix3x4* HQMatrix3x4cRotateAxis(HQVector4 &axis, hq_float32 angle, HQMatrix3x4 *out){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	axis.Normalize();

	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;
	
	out->_14 = out->_24 = out->_34 = 0.0f;
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
HQMatrix3x4* HQMatrix3x4cRotateAxisUnit(const HQVector4 &axis, hq_float32 angle, HQMatrix3x4 *out ){
	hq_float32 fCos,fSin;
	HQSincosf(angle,&fSin,&fCos);

	hq_float32 fSum = 1.0f - fCos;
	hq_float32 fXY=axis.x*axis.y;
	hq_float32 fYZ=axis.y*axis.z;
	hq_float32 fXZ=axis.x*axis.z;

	out->_14 = out->_24 = out->_34 = 0.0f;
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


/*-----------------------------
Matrix inversion
-----------------------------*/
HQMatrix4* HQMatrix3x4Inverse(const HQMatrix3x4* pM,hq_float32* Determinant,HQMatrix4* pOut)
{
	#ifdef CMATH
	hq_float32     f[12],det;
	
	// transpose matrix
	HQMatrix4 trans(pM->_11 , pM->_21 , pM->_31 , 0.0f ,
					pM->_12 , pM->_22 , pM->_32 , 0.0f ,
					pM->_13 , pM->_23 , pM->_33 , 0.0f ,
					pM->_14 , pM->_24 , pM->_34 , 1.0f );
	
	
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
	
	HQNeonMatrix3x4InverseToMatrix4(pM->m, pOut->m, Determinant);
	
#elif defined HQ_DIRECTX_MATH

	HQDXMatrix3x4InverseToMatrix4(pM->m, pOut->m, Determinant);

#else /*SSE version*/
	float4 temp1;
	float4 det;
	float4 f0 , f1 , f2;
	float4 m0 , m1 , m2 , m3;
	static const float4 _3Zeros_1One = {0.0f , 0.0f ,0.0f , 1.0f};

	m0 = _mm_load_ps(pM->m);//row0		m11 m12 m13 m14
	m1 = _mm_load_ps(pM->m + 4);//row1  m21 m22 m23 m24
	m2 = _mm_load_ps(pM->m + 8);//row2  m31 m32 m33 m34
	m3 = _3Zeros_1One;//{0  0  0  1}	m41 m42 m43 m44


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

HQMatrix4* HQMatrix3x4Inverse(const HQMatrix3x4* in,HQMatrix4*out)
{
	return HQMatrix3x4Inverse(in , NULL , out);
}


//*************************************
//in matrix lên màn hình
//*************************************
void HQPrintMatrix3x4(const HQMatrix3x4* pMat)
{
	
	printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
		pMat->_11,pMat->_12,pMat->_13,pMat->_14,
		pMat->_21,pMat->_22,pMat->_23,pMat->_24,
		pMat->_31,pMat->_32,pMat->_33,pMat->_34
		);
}
