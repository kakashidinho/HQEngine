/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_COMBINE_H
#define HQ_SHADER_GL_COMBINE_H

#include "HQShaderGL_CgController.h"
#include "HQShaderGL_GLSLController.h"

#ifndef HQ_OPENGLES


//this controller accepts both Cg and GLSL based shader
class HQCombineShaderController : public HQBaseCgShaderController , public HQBaseGLSLShaderController
{
public:
	HQCombineShaderController(){};
	~HQCombineShaderController(){};

	
	HQ_FORCE_INLINE HQReturnVal DeActiveProgram(bool isGLSL ,HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
	{
		if (isGLSL)
			return this->DeActiveProgramGLSL();
		return this->DeActiveProgramCg(pProgram);
	}
	HQ_FORCE_INLINE HQReturnVal ActiveProgram(bool isGLSL ,HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
	{
		if (isGLSL)
			return this->ActiveProgramGLSL(pProgram);
		return this->ActiveProgramCg(pProgram);
	}
	
	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut);
	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut);

	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut);

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut);

	HQReturnVal CreateProgram(	bool isGLSL ,
								hq_uint32 vertexShaderID,
								hq_uint32 pixelShaderID,
								hq_uint32 geometryShaderID,
								HQSharedPtr<HQShaderObjectGL>& pVShader,
								HQSharedPtr<HQShaderObjectGL>& pGShader,
								HQSharedPtr<HQShaderObjectGL>& pFShader,
								const char** uniformParameterNames,
								hq_uint32 *pID);

	
	
	HQ_FORCE_INLINE HQReturnVal SetUniformInt(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniformIntGLSL(parameter->location , pValues , numElements);
		return this->SetUniformIntCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform2Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniform2IntGLSL(parameter->location , pValues , numElements);
		return this->SetUniform2IntCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform3Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniform3IntGLSL(parameter->location , pValues , numElements);
		return this->SetUniform3IntCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform4Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniform4IntGLSL(parameter->location , pValues , numElements);
		return this->SetUniform4IntCg(parameter->parameter , pValues , numElements);
	}

	
	HQ_FORCE_INLINE HQReturnVal SetUniformFloat(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniformFloatGLSL(parameter->location , pValues , numElements);
		return this->SetUniformFloatCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform2Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniform2FloatGLSL(parameter->location , pValues , numElements);
		return this->SetUniform2FloatCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform3Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniform3FloatGLSL(parameter->location , pValues , numElements);
		return this->SetUniform3FloatCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform4Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		if (isGLSL)
			return this->SetUniform4FloatGLSL(parameter->location , pValues , numElements);
		return this->SetUniform4FloatCg(parameter->parameter , pValues , numElements);
	}



	HQ_FORCE_INLINE HQReturnVal SetUniformMatrix(bool isGLSL,
										const HQShaderParameterGL *parameter,
										const HQBaseMatrix4* pMatrices,
										hq_uint32 numMatrices=1)
	{
		if (isGLSL)
			return this->SetUniformMatrixGLSL(parameter->location , pMatrices , numMatrices);
		return this->SetUniformMatrixCg(parameter->parameter , pMatrices , numMatrices);
	}
	HQ_FORCE_INLINE HQReturnVal SetUniformMatrix(bool isGLSL,
										const HQShaderParameterGL *parameter,
										const HQBaseMatrix3x4* pMatrices,
										hq_uint32 numMatrices=1)
	{
		if (isGLSL)
			return this->SetUniformMatrixGLSL(parameter->location , pMatrices , numMatrices);
		return this->SetUniformMatrixCg(parameter->parameter , pMatrices , numMatrices);
	}

	

	
};


#endif//ifndef HQ_OPENGLES

#endif
