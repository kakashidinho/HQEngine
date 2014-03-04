/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DX_VECTOR_H
#define HQ_DX_VECTOR_H

#include "../../HQUtilMathCommon.h"

using namespace DirectX;

#ifdef __cplusplus
extern "C" {
#endif
	
	HQ_FORCE_INLINE float HQDXVector4Dot(const float* v1 , const float* v2)
	{
		XMVECTOR simdVec1 = DirectX::XMLoadFloat4A((XMFLOAT4A*) v1);
		XMVECTOR simdVec2 = DirectX::XMLoadFloat4A((XMFLOAT4A*) v2);

		XMVECTOR simdDot = DirectX::XMVector3Dot(simdVec1, simdVec2);

		return DirectX::XMVectorGetX(simdDot);
	}
	
	HQ_FORCE_INLINE float HQDXVector4Length(const float* v)
	{
		XMVECTOR simdVec = DirectX::XMLoadFloat4A((XMFLOAT4A*) v);

		return DirectX::XMVectorGetX( DirectX::XMVector3Length(simdVec) );
	}
		
	HQ_FORCE_INLINE void HQDXVector4Cross(const float* v1 , const float* v2 , float *cross)
	{
		XMVECTOR simdVec1 = DirectX::XMLoadFloat4A((XMFLOAT4A*) v1);
		XMVECTOR simdVec2 = DirectX::XMLoadFloat4A((XMFLOAT4A*) v2);

		XMVECTOR simdCross = DirectX::XMVector3Cross(simdVec1, simdVec2);
		
		DirectX::XMStoreFloat4A( (XMFLOAT4A*) cross, simdCross);
	}
	
	HQ_FORCE_INLINE void HQDXVector4Normalize(const float* v , float *normalizedVec)
	{
		XMVECTOR simdVec = DirectX::XMLoadFloat4A((XMFLOAT4A*) v);

		XMVECTOR simdNormalized = DirectX::XMVector3Normalize(simdVec);

		DirectX::XMStoreFloat4A( (XMFLOAT4A*) normalizedVec, simdNormalized);	
	}

	
	
#ifdef __cplusplus
}
#endif


#endif
