/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_GLSL_INLINE_H
#define HQ_SHADER_GL_GLSL_INLINE_H

#include "HQShaderGL_GLSLController.h"
#include <string.h>



inline HQReturnVal HQBaseGLSLShaderController::DeActiveProgramGLSL()
{
	glUseProgram(0);
	return HQ_OK;
}
inline HQReturnVal HQBaseGLSLShaderController::ActiveProgramGLSL(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
	glUseProgram(pProgram->programGLHandle);
	return HQ_OK;
}


/*-----------------------*/

inline HQReturnVal HQBaseGLSLShaderController::SetUniformIntGLSL(GLint param , const hq_int32* pValues,
										hq_uint32 numElements)
{
	glUniform1iv(param , (int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniform2IntGLSL(GLint param , const hq_int32* pValues,
										hq_uint32 numElements)
{
	glUniform2iv(param , (int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniform3IntGLSL(GLint param , const hq_int32* pValues,
										hq_uint32 numElements)
{
	glUniform3iv(param , (int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniform4IntGLSL(GLint param , const hq_int32* pValues,
										hq_uint32 numElements)
{
	glUniform4iv(param , (int)numElements,pValues);

	return HQ_OK;
}



inline HQReturnVal HQBaseGLSLShaderController::SetUniformFloatGLSL(GLint param , const hq_float32* pValues,
										hq_uint32 numElements)
{
	glUniform1fv(param , (int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniform2FloatGLSL(GLint param , const hq_float32* pValues,
										hq_uint32 numElements)
{
	glUniform2fv(param , (int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniform3FloatGLSL(GLint param , const hq_float32* pValues,
										hq_uint32 numElements)
{
	glUniform3fv(param , (int)numElements ,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniform4FloatGLSL(GLint param , const hq_float32* pValues,
										hq_uint32 numElements)
{
	glUniform4fv(param , (int)numElements ,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniformMatrixGLSL(GLint param , const HQBaseMatrix4* pMatrices,
										hq_uint32 numMatrices)
{
	glUniformMatrix4fv(	param, (int)numMatrices,0 , pMatrices->m);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::SetUniformMatrixGLSL(GLint param , const HQBaseMatrix3x4* pMatrices,
										hq_uint32 numMatrices)
{
#ifdef GLES
	return HQ_FAILED;
#else
	if(GLEW_VERSION_2_1)
		glUniformMatrix3x4fv(	param, (int)numMatrices, GL_FALSE , pMatrices->m);
	else
		return HQ_FAILED;
#endif

	return HQ_OK;
}


#endif
