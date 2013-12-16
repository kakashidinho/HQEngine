#ifndef HQ_DX_QUAT_INLINE_H
#define HQ_DX_QUAT_INLINE_H

#include "HQUtilMathCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

	HQ_FORCE_INLINE void HQDXQuatAdd(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	{
		using namespace DirectX;

		XMVECTOR simdQuat1 = XMLoadFloat4A((XMFLOAT4A*) quat1);
		XMVECTOR simdQuat2 = XMLoadFloat4A((XMFLOAT4A*) quat2);

		XMVECTOR simdResult = simdQuat1 + simdQuat2;

		XMStoreFloat4A((XMFLOAT4A*) result, simdResult);
	}

	HQ_FORCE_INLINE void HQDXQuatSub(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	{
		using namespace DirectX;

		XMVECTOR simdQuat1 = XMLoadFloat4A((XMFLOAT4A*) quat1);
		XMVECTOR simdQuat2 = XMLoadFloat4A((XMFLOAT4A*) quat2);

		XMVECTOR simdResult = simdQuat1 - simdQuat2;

		XMStoreFloat4A((XMFLOAT4A*) result, simdResult);
	}

	HQ_FORCE_INLINE void HQDXQuatMultiplyScalar(const hqfloat32 *quat1, hqfloat32 f, hqfloat32 *result)
	{
		using namespace DirectX;

		XMVECTOR simdQuat1 = XMLoadFloat4A((XMFLOAT4A*) quat1);
		XMVECTOR simScalarDup = XMVectorReplicate(f);


		XMVECTOR simdResult = simdQuat1 * simScalarDup;

		XMStoreFloat4A((XMFLOAT4A*) result, simdResult);
	}

#ifdef __cplusplus
}
#endif

#endif