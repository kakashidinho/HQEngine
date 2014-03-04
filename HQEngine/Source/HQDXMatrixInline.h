/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DX_MATRIX_INLINE_H
#define HQ_DX_MATRIX_INLINE_H

#include "HQUtilMathCommon.h"

#ifdef __cplusplus
extern "C" {
#endif


	HQ_FORCE_INLINE void HQDXMatrix4Transpose(const float * matrix , float * result)
	{
		using namespace DirectX;

		XMMATRIX simdMatrix = XMLoadFloat4x4A( (XMFLOAT4X4A*) matrix);
		simdMatrix = XMMatrixTranspose(simdMatrix);

		XMStoreFloat4x4A((XMFLOAT4X4A*)result, simdMatrix);
	}
	
#ifdef __cplusplus
}
#endif


#endif

