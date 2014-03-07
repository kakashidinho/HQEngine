/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_TEMPLATE_H
#define HQ_SHADER_GL_TEMPLATE_H

#include "HQShaderGL.h"
#include <string.h>

/*---------HQShaderManagerGL<ShaderController , BaseShaderManagerClass>--------------*/
template <class ShaderController , class BaseShaderManagerClass>
HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::HQShaderManagerGL(HQLogStream* logFileStream , const char *logPrefix , bool flushLog)
:BaseShaderManagerClass(logFileStream , logPrefix , flushLog)
{
	this->Log("Init done!");
}

template <class ShaderController , class BaseShaderManagerClass>
HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::~HQShaderManagerGL()
{
	this->DestroyAllResource();
	this->Log("Released!");
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::ActiveProgram(hq_uint32 programID)
{
	if( programID == this->activeProgram)
		return HQ_OK;
	HQSharedPtr<HQBaseShaderProgramGL> pProgram ;
	HQReturnVal re;
	switch(programID)
	{
	case HQ_NOT_USE_SHADER:
		pProgram = this->GetItemPointerNonCheck(this->activeProgram);
		re = this->shaderController.DeActiveProgram(pProgram->isGLSL , pProgram);

		this->activeProgram = HQ_NOT_USE_SHADER;

		this->ActiveFFEmu();

		return re;
	default:
		pProgram = this->GetItemPointer(programID);
#if defined _DEBUG || defined DEBUG
		if(pProgram==NULL)
			return HQ_FAILED;
#endif
		this->DeActiveFFEmu();

		re = this->shaderController.ActiveProgram(pProgram->isGLSL , pProgram);
		this->activeProgram = programID;

		return re;
	}
}

/*--------------------------*/
template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::CreateShaderFromStream(HQShaderType type,
										HQDataReaderStream* dataStream,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										hq_uint32 *pID)
{
	HQShaderObjectGL *pNewShader;
	HQReturnVal re = this->shaderController.CreateShaderFromStream(
					type ,
					dataStream ,
					pDefines ,
					isPreCompiled ,
					entryFunctionName ,
					&pNewShader);

	if (HQFailed(re))
		return re;

	if (!this->shaderObjects.AddItem(pNewShader, pID))
	{
		delete pNewShader;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  hq_uint32 *pID)
{
	HQShaderObjectGL *pNewShader;
	HQReturnVal re = this->shaderController.CreateShaderFromMemory(
					type ,
					pSourceData ,
					pDefines ,
					isPreCompiled ,
					entryFunctionName ,
					&pNewShader);

	if (HQFailed(re))
		return re;

	if (!this->shaderObjects.AddItem(pNewShader, pID))
	{
		delete pNewShader;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::CreateShaderFromStream(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	HQShaderObjectGL *pNewShader;
	HQReturnVal re = this->shaderController.CreateShaderFromStream(
					type ,
					compileMode,
					dataStream ,
					pDefines ,
					entryFunctionName ,
					&pNewShader);

	if (HQFailed(re))
		return re;

	if (!this->shaderObjects.AddItem(pNewShader, pID))
	{
		delete pNewShader;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::CreateShaderFromMemory(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	HQShaderObjectGL *pNewShader;
	HQReturnVal re = this->shaderController.CreateShaderFromMemory(
					type ,
					compileMode,
					pSourceData ,
					pDefines ,
					entryFunctionName ,
					&pNewShader);

	if (HQFailed(re))
		return re;

	if (!this->shaderObjects.AddItem(pNewShader, pID))
	{
		delete pNewShader;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

/*-----------------------*/
template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::CreateProgram(hq_uint32 vertexShaderID,
							  hq_uint32 pixelShaderID,
							  hq_uint32 geometryShaderID,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)
{
	HQSharedPtr<HQShaderObjectGL> pVShader = HQSharedPtr<HQShaderObjectGL> :: null;
	HQSharedPtr<HQShaderObjectGL> pFShader = HQSharedPtr<HQShaderObjectGL> :: null;
	HQSharedPtr<HQShaderObjectGL> pGShader = HQSharedPtr<HQShaderObjectGL> :: null;
	if (vertexShaderID != HQ_NOT_USE_VSHADER)
		pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pixelShaderID != HQ_NOT_USE_PSHADER)
		pFShader = this->shaderObjects.GetItemPointer(pixelShaderID);
	if (geometryShaderID != HQ_NOT_USE_GSHADER)
		pGShader = this->shaderObjects.GetItemPointer(geometryShaderID);

	bool GLSL = false;
	if(pVShader != NULL)
	{
		if(pVShader->isGLSL == true)
			GLSL = true;
	}
	else
		vertexShaderID = HQ_NOT_USE_VSHADER;

	if(!GLSL && pFShader != NULL)
	{
		if(pFShader->isGLSL == true)
			GLSL = true;
	}
	else
		geometryShaderID = HQ_NOT_USE_GSHADER;

	if(!GLSL && pGShader != NULL)
	{
		if(pGShader->isGLSL == true)
			GLSL = true;
	}
	else
		pixelShaderID = HQ_NOT_USE_PSHADER;

	return this->shaderController.CreateProgram(
		GLSL,
		vertexShaderID, geometryShaderID, pixelShaderID,
		pVShader , pGShader ,pFShader ,
		uniformParameterNames ,
		pID);
}

/*-----------------------*/

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformInt(const char* parameterName,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	return this->shaderController.SetUniformInt(pProgram->isGLSL , param , pValues , numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform2Int(const char* parameterName,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniform2Int(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform3Int(const char* parameterName,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniform3Int(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform4Int(const char* parameterName,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniform4Int(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformFloat(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniformFloat(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform2Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniform2Float(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform3Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniform3Float(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform4Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniform4Float(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformMatrix(const char* parameterName,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniformMatrix(
		pProgram->isGLSL ,
		param ,
		pMatrices,
		numMatrices);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformMatrix(const char* parameterName,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	return this->shaderController.SetUniformMatrix(
		pProgram->isGLSL ,
		param ,
		pMatrices,
		numMatrices);
}
/*-------parameter index version---------------*/

#if defined _DEBUG || defined DEBUG
#	define GETPARAM(program , paramIndex) (program->parameters.GetItemRawPointer(paramIndex))
#else
#	define GETPARAM(program , paramIndex) (program->parameters.GetItemRawPointerNonCheck(paramIndex))
#endif

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformInt(hq_uint32 parameterIndex,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
	if(this->activeProgram==HQ_NOT_USE_SHADER)
	{
		if (this->IsFFEmuActive())
			return this->SetFFRenderState((HQFFRenderState) parameterIndex, pValues);
		return HQ_FAILED;
	}
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniformInt(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform2Int(hq_uint32 parameterIndex,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniform2Int(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform3Int(hq_uint32 parameterIndex,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniform3Int(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform4Int(hq_uint32 parameterIndex,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniform4Int(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformFloat(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniformFloat(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform2Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniform2Float(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform3Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniform3Float(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniform4Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniform4Float(
		pProgram->isGLSL ,
		param ,
		pValues,
		numElements);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformMatrix(hq_uint32 parameterIndex,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices)
{
	if(this->activeProgram==HQ_NOT_USE_SHADER)
	{
		if (this->IsFFEmuActive())
			return this->SetFFTransform((HQFFTransformMatrix) parameterIndex, pMatrices);
		return HQ_FAILED;
	}
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniformMatrix(
		pProgram->isGLSL ,
		param ,
		pMatrices,
		numMatrices);
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformMatrix(hq_uint32 parameterIndex,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==HQ_NOT_USE_SHADER)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->GetItemRawPointerNonCheck(this->activeProgram);
	HQShaderParameterGL* param = GETPARAM(pProgram , parameterIndex);

#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
#endif

	return this->shaderController.SetUniformMatrix(
		pProgram->isGLSL ,
		param ,
		pMatrices,
		numMatrices);
}


template <class ShaderController , class BaseShaderManagerClass>
void HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::NotifyFFRenderIfNeeded()
{
	if (this->IsFFEmuActive())
	{
		this->NotifyFFRender();
	}
}

template <class ShaderController , class BaseShaderManagerClass>
void HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::OnLost() {
	this->DestroyAllResource();
	this->ReleaseFFEmu();
}

template <class ShaderController , class BaseShaderManagerClass>
void HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::OnReset() {
	this->RestoreFFEmu();
}

#endif
