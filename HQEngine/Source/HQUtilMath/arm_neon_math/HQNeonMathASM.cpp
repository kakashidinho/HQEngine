/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../../HQUtilMathCommon.h"

#ifdef HQ_NEON_ASM
#include "../../HQNeonMatrixInline.h"
#include "../../HQNeonQuaternionInline.h"

extern "C"
{
	//assembly implementations
	void HQNeonQuatAddImpl(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result);

	void HQNeonQuatSubImpl(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result);

	void HQNeonQuatMultiplyScalarImpl(const hqfloat32 *quat1, hqfloat32 f, hqfloat32 *result);

	void HQNeonMatrix4TransposeImpl(const float * matrix , float * result);

	//C dll export wrapper
	void HQNeonMatrix4Transpose(const float * matrix , float * result)
	{
		//call the assembly function
		HQNeonMatrix4TransposeImpl(matrix, result);
	}

	void HQNeonQuatAdd(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	{
		//call the assembly function
		HQNeonQuatAddImpl(quat1, quat2, result);
	}

	void HQNeonQuatSub(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	{
		//call the assembly function
		HQNeonQuatSubImpl(quat1, quat2, result);
	}

	void HQNeonQuatMultiplyScalar(const hqfloat32 *quat1, hqfloat32 f, hqfloat32 *result)
	{
		//call the assembly function
		HQNeonQuatMultiplyScalarImpl(quat1, f, result);
	}

}//extern "C"


#endif//#ifdef HQ_NEON_ASM
