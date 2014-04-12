/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_GLSL_H
#define HQ_SHADER_GL_GLSL_H

#include "HQShaderGL_Common.h"
#include <string>

/*---------------------------*/

class HQBaseGLSLShaderController
{
public:
	HQBaseGLSLShaderController();
	virtual ~HQBaseGLSLShaderController();

protected:
	HQVarParserGL *pVParser;
	
	void BindAttributeLocationGLSL(GLuint program , HQLinkedList<HQShaderAttrib>& attribList);
	void BindUniformBlockGLSL(GLuint program);
	void BindSamplerUnitGLSL(HQBaseShaderProgramGL* pProgram , HQLinkedList<HQUniformSamplerGL>& samplerList);
	
	void GetPredefineMacroGLSL(std::string & macroDefList , const HQShaderMacro * pDefines, std::string &version_string);//convert HQShaderMacro array to GLSL macro definition


	HQReturnVal DeActiveProgramGLSL();
	HQReturnVal ActiveProgramGLSL(HQSharedPtr<HQBaseShaderProgramGL>& pProgram);

	HQReturnVal CreateShaderFromStreamGLSL(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 HQShaderObjectGL **ppShaderObjectOut);
	HQReturnVal CreateShaderFromMemoryGLSL(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 HQShaderObjectGL **ppShaderObjectOut);
	HQReturnVal CreateProgramGLSL(
								HQBaseShaderProgramGL *pNewProgramObj,
							  HQSharedPtr<HQShaderObjectGL>& pVShader,
							  HQSharedPtr<HQShaderObjectGL>& pGShader,
							  HQSharedPtr<HQShaderObjectGL>& pFShader);
	
	

	HQReturnVal SetUniformIntGLSL(GLint param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform2IntGLSL(GLint param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform3IntGLSL(GLint param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform4IntGLSL(GLint param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniformFloatGLSL(GLint param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform2FloatGLSL(GLint param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform3FloatGLSL(GLint param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform4FloatGLSL(GLint param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniformMatrixGLSL( GLint param,
							const HQBaseMatrix4* pMatrices,
							hq_uint32 numMatrices);
	HQReturnVal SetUniformMatrixGLSL( GLint param,
							const HQBaseMatrix3x4* pMatrices,
							hq_uint32 numMatrices);
};


/*----------HQGLSLShaderController------------------*/
//this controller only accepts GLSL based shader
class HQGLSLShaderController : public HQBaseGLSLShaderController
{
public:
	HQGLSLShaderController(){};
	~HQGLSLShaderController(){};

	
	HQ_FORCE_INLINE HQReturnVal DeActiveProgram(bool isGLSL ,HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
	{
		return this->DeActiveProgramGLSL();
	}
	HQ_FORCE_INLINE HQReturnVal ActiveProgram(bool isGLSL ,HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
	{
		return this->ActiveProgramGLSL(pProgram);
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

	HQReturnVal CreateProgram(  HQBaseShaderProgramGL *pNewProgramObj,
								HQSharedPtr<HQShaderObjectGL>& pVShader,
								HQSharedPtr<HQShaderObjectGL>& pGShader,
								HQSharedPtr<HQShaderObjectGL>& pFShader);

	
	HQ_FORCE_INLINE HQReturnVal SetUniformInt(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniformIntGLSL(parameter->location , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform2Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform2IntGLSL(parameter->location , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform3Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform3IntGLSL(parameter->location , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform4Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform4IntGLSL(parameter->location , pValues , numElements);
	}

	
	HQ_FORCE_INLINE HQReturnVal SetUniformFloat(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniformFloatGLSL(parameter->location , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform2Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform2FloatGLSL(parameter->location , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform3Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform3FloatGLSL(parameter->location , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform4Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform4FloatGLSL(parameter->location , pValues , numElements);
	}



	HQ_FORCE_INLINE HQReturnVal SetUniformMatrix(bool isGLSL,
										const HQShaderParameterGL *parameter,
										const HQBaseMatrix4* pMatrices,
										hq_uint32 numMatrices=1)
	{
		return this->SetUniformMatrixGLSL(parameter->location , pMatrices , numMatrices);
	}
	HQ_FORCE_INLINE HQReturnVal SetUniformMatrix(bool isGLSL,
										const HQShaderParameterGL *parameter,
										const HQBaseMatrix3x4* pMatrices,
										hq_uint32 numMatrices=1)
	{
		return this->SetUniformMatrixGLSL(parameter->location , pMatrices , numMatrices);
	}

	

	
};

#endif
