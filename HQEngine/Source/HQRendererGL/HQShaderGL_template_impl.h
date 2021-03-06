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
	this->RemoveAllResource();
	this->Log("Released!");
}

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::ActiveProgram(HQShaderProgram* program)
{
	if( program == this->activeProgram.GetRawPointer())
		return HQ_OK;
	HQSharedPtr<HQBaseShaderProgramGL> pProgramSharedPtr ;
	HQReturnVal re;
	if (program == NULL)
	{
		this->shaderController.DeactiveProgram(this->activeProgram);
		this->activeProgram = HQSharedPtr<HQBaseShaderProgramGL>::null;

		this->ActiveFFEmu();

		re = HQ_OK;
	}

	else {
		pProgramSharedPtr = this->GetItemPointer(program);
#if defined _DEBUG || defined DEBUG
		if (pProgramSharedPtr == NULL)
			return HQ_FAILED;
#endif
		this->DeActiveFFEmu();

		re = this->shaderController.ActiveProgram(pProgramSharedPtr);
		this->activeProgram = pProgramSharedPtr;
		this->BaseShaderManagerClass::OnProgramActivated();//tell parent class
	}

	return re;
}

/*-----------------------------*/
template <class ShaderController, class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController, BaseShaderManagerClass>::ActiveComputeShader(HQShaderObject *shader)
{
	HQSharedPtr<HQShaderObjectGL> pShader = this->shaderObjects.GetItemPointer(shader);

	if (this->activeCShader != pShader)
	{
		if (pShader == NULL)
		{
			this->shaderController.DeactiveComputeShader(this->activeCShader);
		}
		else
		{
			if (pShader->type != HQ_COMPUTE_SHADER)
			{
				this->Log("Error : invalid compute shader!");
				return HQ_FAILED_INVALID_ID;
			}

			this->shaderController.ActiveComputeShader(pShader);
		}

		this->activeCShader = pShader;
	}

	return HQ_OK;
}

/*--------------------------*/
template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::CreateShaderFromStream(HQShaderType type,
										HQDataReaderStream* dataStream,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										HQShaderObject** pID)
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
										  HQShaderObject**pID)
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
								 HQShaderObject**pID)
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
								 HQShaderObject**pID)
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
HQReturnVal HQShaderManagerGL<ShaderController, BaseShaderManagerClass>::CreateProgram(HQShaderObject* vertexShaderID,
								HQShaderObject* pixelShaderID,
								HQShaderObject* geometryShaderID,
								HQShaderProgram **pID)
{
	HQSharedPtr<HQShaderObjectGL> pVShader = HQSharedPtr<HQShaderObjectGL> :: null;
	HQSharedPtr<HQShaderObjectGL> pFShader = HQSharedPtr<HQShaderObjectGL> :: null;
	HQSharedPtr<HQShaderObjectGL> pGShader = HQSharedPtr<HQShaderObjectGL> :: null;
	if (vertexShaderID != NULL)
		pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pixelShaderID != NULL)
		pFShader = this->shaderObjects.GetItemPointer(pixelShaderID);
	if (geometryShaderID != NULL)
		pGShader = this->shaderObjects.GetItemPointer(geometryShaderID);

	bool GLSL = false;
	if(pVShader != NULL)
	{
		if(pVShader->isGLSL == true)
			GLSL = true;
	}
	else
		vertexShaderID = NULL;

	if(pFShader != NULL)
	{
		if(pFShader->isGLSL == true)
			GLSL = true;
	}
	else
		pixelShaderID = NULL;

	if(pGShader != NULL)
	{
		if(pGShader->isGLSL == true)
			GLSL = true;
	}
	else
		geometryShaderID = NULL;

	HQBaseShaderProgramGL * pNewProgram = this->CreateNewProgramObject();
	pNewProgram->isGLSL = GLSL;

	HQReturnVal re = this->shaderController.CreateProgram(
		pNewProgram,
		pVShader , pGShader ,pFShader );

	if (HQFailed(re))
		delete pNewProgram;
	else
	{
		this->OnProgramCreated(pNewProgram);

		if (pID != NULL)
			*pID = pNewProgram;
	}

	return re;

}

/*-----------------------*/

template <class ShaderController , class BaseShaderManagerClass>
HQReturnVal HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::SetUniformInt(const char* parameterName,
					 const hq_int32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
	const HQShaderParameterGL* param = pProgram->GetParameter( parameterName);
#if defined _DEBUG || defined DEBUG	
	if (param == NULL)
	{
		this->Log("error :parameter \"%s\" not found from program (%u)!",parameterName,this->activeProgram->GetID());
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
	if(this->activeProgram==NULL)
	{
		if (this->IsFFEmuActive())
			return this->SetFFRenderState((HQFFRenderState) parameterIndex, pValues);
		return HQ_FAILED;
	}
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
	{
		if (this->IsFFEmuActive())
			return this->SetFFTransform((HQFFTransformMatrix) parameterIndex, pMatrices);
		return HQ_FAILED;
	}
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
	if(this->activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	HQBaseShaderProgramGL* pProgram = this->activeProgram.GetRawPointer();
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
void HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::OnDraw()
{
	if (this->IsFFEmuActive())
	{
		this->NotifyFFRender();
	}
	else
		BaseShaderManagerClass::OnDraw();
}

template <class ShaderController , class BaseShaderManagerClass>
void HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::OnLost() {
	this->RemoveAllResource();
	this->ReleaseFFEmu();
}

template <class ShaderController , class BaseShaderManagerClass>
void HQShaderManagerGL<ShaderController , BaseShaderManagerClass>::OnReset() {
	this->RestoreFFEmu();
}

#endif
