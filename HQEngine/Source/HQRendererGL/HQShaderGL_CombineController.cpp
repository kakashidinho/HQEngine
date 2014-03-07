/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQShaderGL_CgController_inline.h"
#include "HQShaderGL_GLSLController_inline.h"
#include "HQShaderGL_CombineController.h"
#ifndef GLES
/*----------HQCombineShaderController----------------*/

HQReturnVal HQCombineShaderController::CreateShaderFromStream(HQShaderType type,
										HQDataReaderStream* dataStream,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromStreamCg(type,dataStream,pDefines ,isPreCompiled , entryFunctionName ,ppShaderObjectOut);
}

HQReturnVal HQCombineShaderController::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromMemoryCg(type , pSourceData,pDefines ,isPreCompiled,entryFunctionName, ppShaderObjectOut);
}

HQReturnVal HQCombineShaderController::CreateShaderFromStream(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_GLSL:case HQ_SCM_GLSL_DEBUG:
		return this->CreateShaderFromStreamGLSL(type , dataStream,pDefines,ppShaderObjectOut);

	case HQ_SCM_CG:case HQ_SCM_CG_DEBUG:
		return this->CreateShaderFromStreamCg(type , dataStream ,pDefines,false , entryFunctionName, ppShaderObjectOut);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
}

HQReturnVal HQCombineShaderController::CreateShaderFromMemory(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_GLSL:case HQ_SCM_GLSL_DEBUG:
		return this->CreateShaderFromMemoryGLSL(type , pSourceData ,pDefines,ppShaderObjectOut);

	case HQ_SCM_CG:case HQ_SCM_CG_DEBUG:
		return this->CreateShaderFromMemoryCg(type , pSourceData ,pDefines,false , entryFunctionName, ppShaderObjectOut);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}

}

HQReturnVal HQCombineShaderController::CreateProgram(	bool isGLSL ,
								hq_uint32 vertexShaderID,
								hq_uint32 pixelShaderID,
								hq_uint32 geometryShaderID,
								HQSharedPtr<HQShaderObjectGL>& pVShader,
								HQSharedPtr<HQShaderObjectGL>& pGShader,
								HQSharedPtr<HQShaderObjectGL>& pFShader,
								const char** uniformParameterNames,
								hq_uint32 *pID)
{
	hquint32 programID;
	HQReturnVal re ;

	if (isGLSL)
		re = this->CreateProgramGLSL(
			pVShader , pGShader , pFShader , 
			uniformParameterNames , 
			&programID);
	else
		re = this->CreateProgramCg(
			pVShader , pGShader , pFShader , 
			uniformParameterNames , 
			&programID);
	
	
	if (pID != NULL)
		*pID = programID;

	//store shaders' IDs
	if (!HQFailed(re))
	{
		HQSharedPtr<HQBaseShaderProgramGL> pProgram = g_pShaderMan->GetItemPointer(programID);
		pProgram->vertexShaderID = vertexShaderID;
		pProgram->geometryShaderID = geometryShaderID;
		pProgram->pixelShaderID = pixelShaderID;
	}

	return re;
}

#endif //ifndef GLES
