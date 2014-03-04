/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DX_MATRIX_H
#define HQ_DX_MATRIX_H

#include "../../HQUtilMathCommon.h"

using namespace DirectX;

	
HQ_FORCE_INLINE XMMATRIX HQDXLoadMatrix3x4(const float* matrix1)
{
	XMMATRIX simdMatrix4;

	simdMatrix4.r[0] = XMLoadFloat4A((XMFLOAT4A*) matrix1);//row0
	simdMatrix4.r[1] = XMLoadFloat4A((XMFLOAT4A*) (matrix1 + 4));//row1
	simdMatrix4.r[2] = XMLoadFloat4A((XMFLOAT4A*) (matrix1 + 8));//row2
	simdMatrix4.r[3] = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);//0 0 0 1

	return simdMatrix4;
}

HQ_FORCE_INLINE void HQDXStoreMatrix3x4(float* matrix, const XMMATRIX &simdMatrix4)
{
	XMStoreFloat4A((XMFLOAT4A*) matrix, simdMatrix4.r[0]);
	XMStoreFloat4A((XMFLOAT4A*) (matrix + 4), simdMatrix4.r[1]);
	XMStoreFloat4A((XMFLOAT4A*) (matrix + 8), simdMatrix4.r[2]);
}

HQ_FORCE_INLINE void HQDXMatrix4Inverse(const float *matrix1 , float *result , float * pDeterminant)
{
	XMMATRIX simdMat = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix1);
	XMVECTOR determinant;
	XMMATRIX inverse =  XMMatrixInverse(&determinant, simdMat);

	if (pDeterminant != NULL)
		XMVectorGetXPtr(pDeterminant, determinant);

	XMStoreFloat4x4A((XMFLOAT4X4A*) result, inverse);
}
	
HQ_FORCE_INLINE void HQDXMatrix3x4InverseToMatrix4(const float *matrix1 , float *result , float * pDeterminant)
{
	XMMATRIX simdMatrix4 = HQDXLoadMatrix3x4(matrix1);

	XMVECTOR determinant;
	XMMATRIX inverse = XMMatrixInverse(&determinant, simdMatrix4);

	if (pDeterminant != NULL)
		XMVectorGetXPtr(pDeterminant, determinant);

	XMStoreFloat4x4A((XMFLOAT4X4A*) result, inverse);
}
	
HQ_FORCE_INLINE void HQDXMatrix4Multiply(const float * matrix1 , const float *matrix2 , float * result )
{
	XMMATRIX simdMat1 = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix1);
	XMMATRIX simdMat2 = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix2);
		 
	XMMATRIX simdResult = XMMatrixMultiply(simdMat1, simdMat2);

	XMStoreFloat4x4A((XMFLOAT4X4A*) result, simdResult);
}
	
HQ_FORCE_INLINE void HQDXMultiMatrix4Multiply(const float * matrix , hq_uint32 numMat , float * result )
{

	XMMATRIX simdResult = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);//first matrix

	XMMATRIX nextMatrix;

	for (hq_uint32 i = 1 ; i < numMat ; ++i)
	{
		nextMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) (matrix + i * 16));//first matrix

		simdResult = XMMatrixMultiply(simdResult, nextMatrix);
	}

	XMStoreFloat4x4A((XMFLOAT4X4A*) result, simdResult);
		
}
	
HQ_FORCE_INLINE void HQDXVector4MultiplyMatrix4(const float *vector ,const float * matrix ,  float * resultVector )
{
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)vector);
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix

	XMVECTOR simdResult = XMVector4Transform(simdVec, simdMatrix);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);	
}
	
HQ_FORCE_INLINE void HQDXMultiVector4MultiplyMatrix4(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
{
		
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix

	for (hquint32 i = 0; i < numVec; ++i)
	{
		XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)(vector + i * 4));

		XMVECTOR simdResult = XMVector4Transform(simdVec, simdMatrix);

		XMStoreFloat4A((XMFLOAT4A*) (resultVector + i * 4), simdResult);
	}
}
	
HQ_FORCE_INLINE void HQDXVector4TransformCoord(const float *vector ,const float * matrix ,  float * resultVector )
{
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)vector);
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix

	XMVECTOR simdResult = XMVector3TransformCoord(simdVec, simdMatrix);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);	
		
}
	
HQ_FORCE_INLINE void HQDXMultiVector4TransformCoord(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
{
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix

	for (hquint32 i = 0; i < numVec; ++i)
	{
		XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)(vector + i * 4));

		XMVECTOR simdResult = XMVector3TransformCoord(simdVec, simdMatrix);

		XMStoreFloat4A((XMFLOAT4A*) (resultVector + i * 4), simdResult);
	}
		
}
	
HQ_FORCE_INLINE void HQDXVector4TransformNormal(const float *vector ,const float * matrix ,  float * resultVector )
{
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)vector);
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix

	XMVECTOR simdResult = XMVector3TransformNormal(simdVec, simdMatrix);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);	
		
}
	
HQ_FORCE_INLINE void HQDXMultiVector4TransformNormal(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
{
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix

	for (hquint32 i = 0; i < numVec; ++i)
	{
		XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)(vector + i * 4));

		XMVECTOR simdResult = XMVector3TransformNormal(simdVec, simdMatrix);

		XMStoreFloat4A((XMFLOAT4A*) (resultVector + i * 4), simdResult);
	}
		
}
	
HQ_FORCE_INLINE void HQDXMatrix4MultiplyVector4(const float * matrix , const float *vector , float * resultVector )
{
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*)vector);
	XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix);// matrix
	XMMATRIX transpose = XMMatrixTranspose(simdMatrix);

	XMVECTOR simdResult = XMVector3TransformNormal(simdVec, transpose);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);	
		
}
	
	
HQ_FORCE_INLINE void HQDXMatrix3x4Multiply(const float * matrix1 , const float *matrix2 , float * result )
{
	XMMATRIX simdMat1 = HQDXLoadMatrix3x4(matrix1);
	XMMATRIX simdMat2 = HQDXLoadMatrix3x4(matrix2);

	XMMATRIX simdResult = XMMatrixMultiply(simdMat1, simdMat2);

	HQDXStoreMatrix3x4( result, simdResult);
		
}
	
HQ_FORCE_INLINE void HQDXMatrix4MultiplyMatrix3x4(const float * matrix1 , const float *matrix2 , float * result )
{
	XMMATRIX simdMat1 = XMLoadFloat4x4A((XMFLOAT4X4A*) matrix1);// matrix1
	XMMATRIX simdMat2 = HQDXLoadMatrix3x4(matrix2);

	XMMATRIX simdResult = XMMatrixMultiply(simdMat1, simdMat2);

	XMStoreFloat4x4A((XMFLOAT4X4A*) result, simdResult);	
}
	
HQ_FORCE_INLINE void HQDXMultiMatrix3x4Multiply(const float * matrix , hq_uint32 numMat , float * result )
{
	XMMATRIX nextMatrix;

	XMMATRIX simdResult = HQDXLoadMatrix3x4(matrix);//first matrix

	for (hquint32 i = 0 ;i < numMat; ++i)
	{
		nextMatrix = HQDXLoadMatrix3x4(matrix + i * 12);

		simdResult = XMMatrixMultiply(simdResult, nextMatrix);
	}

	HQDXStoreMatrix3x4( result, simdResult);
		
}
	
HQ_FORCE_INLINE void HQDXMatrix3x4MultiplyVector4(const float * matrix , const float *vector , float * resultVector )
{
	XMMATRIX simdMat = HQDXLoadMatrix3x4(matrix);
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*) vector);

	XMMATRIX transpose = XMMatrixTranspose(simdMat);

	XMVECTOR simdResult = XMVector4Transform(simdVec, transpose);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);
		
}
	
HQ_FORCE_INLINE void HQDXMatrix3x4MultiplyMultiVector4(const float * matrix , const float *vector ,hq_uint32 numVec , float * resultVector )
{
	XMMATRIX simdMat = HQDXLoadMatrix3x4(matrix);
	XMMATRIX transpose = XMMatrixTranspose(simdMat);

	for (hquint32 i = 0; i < numVec; ++i)
	{
		XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*) (vector + i * 4));


		XMVECTOR simdResult = XMVector4Transform(simdVec, transpose);

		XMStoreFloat4A((XMFLOAT4A*) (resultVector + i * 4), simdResult);
	}
		
}
	
HQ_FORCE_INLINE void HQDXVector4TransformCoordMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector )
{
	XMMATRIX simdMat = HQDXLoadMatrix3x4(matrix);
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*) vector);

	XMMATRIX transpose = XMMatrixTranspose(simdMat);

	XMVECTOR simdResult = XMVector3TransformCoord(simdVec, transpose);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);
		
}

HQ_FORCE_INLINE void HQDXMultiVector4TransformCoordMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix , float * resultVector )
{
	XMMATRIX simdMat = HQDXLoadMatrix3x4(matrix);
	XMMATRIX transpose = XMMatrixTranspose(simdMat);

	for (hquint32 i = 0; i < numVec; ++i)
	{
		XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*) (vector + i * 4));


		XMVECTOR simdResult = XMVector3TransformCoord(simdVec, transpose);

		XMStoreFloat4A((XMFLOAT4A*) (resultVector + i * 4), simdResult);
	}
		
}
	
HQ_FORCE_INLINE void HQDXVector4TransformNormalMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector )
{
		
	XMMATRIX simdMat = HQDXLoadMatrix3x4(matrix);
	XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*) vector);

	XMMATRIX transpose = XMMatrixTranspose(simdMat);

	XMVECTOR simdResult = XMVector3TransformNormal(simdVec, transpose);

	XMStoreFloat4A((XMFLOAT4A*) resultVector, simdResult);
		
}
	
HQ_FORCE_INLINE void HQDXMultiVector4TransformNormalMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix ,  float * resultVector )
{
		
	XMMATRIX simdMat = HQDXLoadMatrix3x4(matrix);
	XMMATRIX transpose = XMMatrixTranspose(simdMat);

	for (hquint32 i = 0; i < numVec; ++i)
	{
		XMVECTOR simdVec = XMLoadFloat4A((XMFLOAT4A*) (vector + i * 4));


		XMVECTOR simdResult = XMVector3TransformNormal(simdVec, transpose);

		XMStoreFloat4A((XMFLOAT4A*) (resultVector + i * 4), simdResult);
	}
		
}



#endif

