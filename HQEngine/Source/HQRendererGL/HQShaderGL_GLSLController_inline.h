/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_GLSL_INLINE_H
#define HQ_SHADER_GL_GLSL_INLINE_H

#include "HQShaderGL_GLSLController.h"
#include <string.h>


/*------------------------HQBaseGLSLShaderController---------------------------*/
inline HQReturnVal HQBaseGLSLShaderController::ActiveProgramGLSL(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
	glUseProgram(pProgram->programGLHandle);
#if 0
    GLint OK;
    glValidateProgram(pProgram->programGLHandle);
	glGetProgramiv(pProgram->programGLHandle, GL_VALIDATE_STATUS, &OK);
	if(OK == GL_FALSE)
	{
		int infologLength = 0;
		int charsWritten  = 0;
		char *infoLog;
		glGetProgramiv(pProgram->programGLHandle, GL_INFO_LOG_LENGTH, &infologLength);
		if (infologLength > 0)
		{
			infoLog = (char *)malloc(infologLength);
			glGetProgramInfoLog(pProgram->programGLHandle, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL program validate error: %s", infoLog);
			free(infoLog);
		}
		return HQ_FAILED;
	}
#endif
	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderController::DeactiveProgramGLSL(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
	return HQ_OK;
}


inline HQReturnVal HQBaseGLSLShaderController::ActiveComputeShaderGLSL(HQSharedPtr<HQShaderObjectGL> &pShader)
{
	//not supported
	return HQ_FAILED;
}

inline HQReturnVal HQBaseGLSLShaderController::DeactiveComputeShaderGLSL(HQSharedPtr<HQShaderObjectGL> &pShader)
{
	//not supported
	return HQ_FAILED;
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
	if(GLEW_VERSION_2_1)
		glUniformMatrix3x4fv(	param, (int)numMatrices, GL_FALSE , pMatrices->m);
	else
		return HQ_FAILED;

	return HQ_OK;
}


#ifdef HQ_GLSL_SHADER_PIPELINE_DEFINED
#define HQ_GLSL_SHADER_VERIFY_PIPELINE 0
/*------------------------HQBaseGLSLShaderPipelineController---------------------------*/
inline HQReturnVal HQBaseGLSLShaderPipelineController::ActiveProgramGLSL(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
#if HQ_GLSL_SHADER_VERIFY_PIPELINE
	GLint pipeline;
	glGetIntegerv(GL_PROGRAM_PIPELINE_BINDING, &pipeline);
#endif

	GLbitfield activeStages = 0;
	GLbitfield inactiveStages = GL_VERTEX_SHADER_BIT | GL_GEOMETRY_SHADER_BIT | GL_FRAGMENT_SHADER_BIT;

	if (pProgram->vertexShader != NULL)
	{
		activeStages |= GL_VERTEX_SHADER_BIT;
	}
	if (pProgram->geometryShader != NULL)
	{
		activeStages |= GL_GEOMETRY_SHADER_BIT;
	}
	if (pProgram->pixelShader != NULL)
	{
		activeStages |= GL_FRAGMENT_SHADER_BIT;
	}

	inactiveStages &= ~activeStages;

	glUseProgramStages(HQ_GLSL_SHADER_PIPELINE_ID, activeStages, pProgram->programGLHandle);
	glUseProgramStages(HQ_GLSL_SHADER_PIPELINE_ID, inactiveStages, 0);

	this->activeProgram = pProgram;
	return HQ_OK;
}


inline HQReturnVal HQBaseGLSLShaderPipelineController::DeactiveProgramGLSL(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
	this->activeProgram.ToNull();
	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::ActiveComputeShaderGLSL(HQSharedPtr<HQShaderObjectGL> &pShader)
{
#ifdef GL_COMPUTE_SHADER_BIT
	glUseProgramStages(HQ_GLSL_SHADER_PIPELINE_ID, GL_COMPUTE_SHADER_BIT, pShader->shader);//shader is self-contained program
	return HQ_OK;
#else
	return HQ_FAILED;
#endif
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::DeactiveComputeShaderGLSL(HQSharedPtr<HQShaderObjectGL> &pShader)
{
#ifdef GL_COMPUTE_SHADER_BIT
	glUseProgramStages(HQ_GLSL_SHADER_PIPELINE_ID, GL_COMPUTE_SHADER_BIT, 0);
	return HQ_OK;
#else
	return HQ_FAILED;
#endif
}


/*-----------------------*/

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniformIntGLSL(GLint param, const hq_int32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform1iv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniform2IntGLSL(GLint param, const hq_int32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform2iv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniform3IntGLSL(GLint param, const hq_int32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform3iv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniform4IntGLSL(GLint param, const hq_int32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform4iv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}



inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniformFloatGLSL(GLint param, const hq_float32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform1fv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniform2FloatGLSL(GLint param, const hq_float32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform2fv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniform3FloatGLSL(GLint param, const hq_float32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform3fv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniform4FloatGLSL(GLint param, const hq_float32* pValues,
	hq_uint32 numElements)
{
	glProgramUniform4fv(this->activeProgram->programGLHandle, param, (int)numElements, pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniformMatrixGLSL(GLint param, const HQBaseMatrix4* pMatrices,
	hq_uint32 numMatrices)
{
	glProgramUniformMatrix4fv(this->activeProgram->programGLHandle, param, (int)numMatrices, 0, pMatrices->m);

	return HQ_OK;
}

inline HQReturnVal HQBaseGLSLShaderPipelineController::SetUniformMatrixGLSL(GLint param, const HQBaseMatrix3x4* pMatrices,
	hq_uint32 numMatrices)
{
	glProgramUniformMatrix3x4fv(this->activeProgram->programGLHandle, param, (int)numMatrices, GL_FALSE, pMatrices->m);

	return HQ_OK;
}

#endif//#ifdef HQ_GLSL_SHADER_PIPELINE_DEFINED

/*----------HQGLSLShaderControllerTemplate----------------*/
template <class BaseGLSLShaderController>
HQReturnVal HQGLSLShaderControllerTemplate<BaseGLSLShaderController>::CreateShaderFromStream(HQShaderType type,
	HQDataReaderStream* dataStream,
	const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
	bool isPreCompiled,
	const char* entryFunctionName,
	HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromStreamGLSL(type, dataStream, pDefines, ppShaderObjectOut);
}

template <class BaseGLSLShaderController>
HQReturnVal HQGLSLShaderControllerTemplate<BaseGLSLShaderController>::CreateShaderFromMemory(HQShaderType type,
	const char* pSourceData,
	const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
	bool isPreCompiled,
	const char* entryFunctionName,
	HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromMemoryGLSL(type, pSourceData, pDefines, ppShaderObjectOut);
}

template <class BaseGLSLShaderController>
HQReturnVal HQGLSLShaderControllerTemplate<BaseGLSLShaderController>::CreateShaderFromStream(HQShaderType type,
	HQShaderCompileMode compileMode,
	HQDataReaderStream* dataStream,
	const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
	const char* entryFunctionName,
	HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_GLSL:case HQ_SCM_GLSL_DEBUG:
		return this->CreateShaderFromStreamGLSL(type, dataStream, pDefines, ppShaderObjectOut);

	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
}

template <class BaseGLSLShaderController>
HQReturnVal HQGLSLShaderControllerTemplate<BaseGLSLShaderController>::CreateShaderFromMemory(HQShaderType type,
	HQShaderCompileMode compileMode,
	const char* pSourceData,
	const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
	const char* entryFunctionName,
	HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_GLSL:case HQ_SCM_GLSL_DEBUG:
		return this->CreateShaderFromMemoryGLSL(type, pSourceData, pDefines, ppShaderObjectOut);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}

}

template <class BaseGLSLShaderController>
HQReturnVal HQGLSLShaderControllerTemplate<BaseGLSLShaderController>::CreateProgram(
	HQBaseShaderProgramGL *pNewProgramObj,
	HQSharedPtr<HQShaderObjectGL>& pVShader,
	HQSharedPtr<HQShaderObjectGL>& pGShader,
	HQSharedPtr<HQShaderObjectGL>& pFShader)
{
	if (pNewProgramObj->isGLSL == false)
		return HQ_FAILED;

	HQReturnVal re = this->CreateProgramGLSL(pNewProgramObj, pVShader, pGShader, pFShader);

	//store shaders' IDs
	if (!HQFailed(re))
	{
		pNewProgramObj->vertexShader = pVShader.GetRawPointer();
		pNewProgramObj->geometryShader = pGShader.GetRawPointer();
		pNewProgramObj->pixelShader = pFShader.GetRawPointer();
	}

	return re;
}

#endif
