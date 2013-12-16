/*
 *  HQNeonMatrixInline.h
 *
 *  Created by Kakashidinho on 5/29/11.
 *  Copyright 2011 LHQ. All rights reserved.
 *
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

