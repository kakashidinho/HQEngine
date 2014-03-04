/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DX_QUATERNION_H
#define HQ_DX_QUATERNION_H

#include "../../HQUtilMathCommon.h"
#include "HQDXVector.h"
#include "HQDXMatrix.h"

using namespace DirectX;

#ifdef __cplusplus
extern "C" {
#endif

	HQ_FORCE_INLINE float HQDXQuatMagnitude(float * quat)
	{

		XMVECTOR simdQuat = DirectX::XMLoadFloat4A((XMFLOAT4A*) quat);
		
		XMVECTOR magnitude = XMVector4Length(simdQuat);

		return XMVectorGetX(magnitude);
	}
	
	HQ_FORCE_INLINE float HQDXQuatSumSquares(float * quat)
	{
		XMVECTOR simdQuat = DirectX::XMLoadFloat4A((XMFLOAT4A*) quat);
		
		XMVECTOR magnitudeSq = XMVector4LengthSq(simdQuat);

		return XMVectorGetX(magnitudeSq);
	}
	
	HQ_FORCE_INLINE float HQDXQuatDot(const float * quat1 ,const float *quat2)
	{
		XMVECTOR simdQuat1 = DirectX::XMLoadFloat4A((XMFLOAT4A*) quat1);
		XMVECTOR simdQuat2 = DirectX::XMLoadFloat4A((XMFLOAT4A*) quat2);
		
		XMVECTOR simdDot = XMVector4Dot(simdQuat1, simdQuat2);

		return XMVectorGetX(simdDot);
	}
	
	HQ_FORCE_INLINE void HQDXQuatNormalize(const float* q , float *normalizedQuat)
	{
		XMVECTOR simdQuat = DirectX::XMLoadFloat4A((XMFLOAT4A*) q);

		XMVECTOR simdNormalized = DirectX::XMQuaternionNormalize(simdQuat);
		
		DirectX::XMStoreFloat4A((XMFLOAT4A*) normalizedQuat, simdNormalized);
	}
	
	HQ_FORCE_INLINE void HQDXQuatInverse(const float* q , float *result)
	{
		XMVECTOR simdQuat = DirectX::XMLoadFloat4A((XMFLOAT4A*) q);

		XMVECTOR simdInverse = DirectX::XMQuaternionInverse(simdQuat);
		
		DirectX::XMStoreFloat4A((XMFLOAT4A*) result, simdInverse);
		
	}
	
	HQ_FORCE_INLINE void HQDXQuatMultiply(const float * quat1 ,const  float *quat2 , float* result)
	{
		XMVECTOR simdQuat1 = DirectX::XMLoadFloat4A((XMFLOAT4A*) quat1);
		XMVECTOR simdQuat2 = DirectX::XMLoadFloat4A((XMFLOAT4A*) quat2);
		
		XMVECTOR simdMultiply = XMQuaternionMultiply(simdQuat2, simdQuat1);

		DirectX::XMStoreFloat4A((XMFLOAT4A*) result, simdMultiply);
	}
	
	HQ_FORCE_INLINE void HQDXQuatUnitToRotAxis(const float* q , float *axisVector)
	{
		HQDXVector4Normalize(q, axisVector);
	}
	
	HQ_FORCE_INLINE void HQDXQuatUnitToMatrix3x4c(const float* q , float *matrix)
	{
		XMVECTOR simdQuat = DirectX::XMLoadFloat4A((XMFLOAT4A*) q);

		XMMATRIX simdMatrix = XMMatrixRotationQuaternion(simdQuat);

		simdMatrix = XMMatrixTranspose(simdMatrix);

		XMStoreFloat4A((XMFLOAT4A*) matrix, simdMatrix.r[0]);//row0
		XMStoreFloat4A((XMFLOAT4A*) (matrix + 4), simdMatrix.r[1]);//row1
		XMStoreFloat4A((XMFLOAT4A*) (matrix + 8), simdMatrix.r[2]);//row2
	}
	
	
	HQ_FORCE_INLINE void HQDXQuatUnitToMatrix4r(const float* q , float *matrix)
	{
		XMVECTOR simdQuat = DirectX::XMLoadFloat4A((XMFLOAT4A*) q);

		XMMATRIX simdMatrix = XMMatrixRotationQuaternion(simdQuat);

		XMStoreFloat4x4A((XMFLOAT4X4A*) matrix, simdMatrix);
	}
	
	HQ_FORCE_INLINE void HQDXQuatFromMatrix3x4c(const float *matrix , float * quaternion)
	{
		XMMATRIX simdMatrix = HQDXLoadMatrix3x4(matrix);
		simdMatrix = XMMatrixTranspose(simdMatrix);

		XMVECTOR simdQuat = XMQuaternionRotationMatrix(simdMatrix);

		XMStoreFloat4A((XMFLOAT4A*) quaternion, simdQuat);
	}


	HQ_FORCE_INLINE void HQDXQuatFromMatrix4r(const float *matrix , float * quaternion)
	{
		XMMATRIX simdMatrix = XMLoadFloat4x4A((XMFLOAT4X4A*)matrix);

		XMVECTOR simdQuat = XMQuaternionRotationMatrix(simdMatrix);

		XMStoreFloat4A((XMFLOAT4A*) quaternion, simdQuat);
	}
	
#ifdef __cplusplus
}
#endif

#endif

