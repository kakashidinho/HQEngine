/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_CG_H
#define HQ_SHADER_GL_CG_H

#include "HQShaderGL_Common.h"

#ifndef GLES


#ifdef WIN32
typedef HMODULE libHandle_t;
#else
#include <dlfcn.h>
typedef void * libHandle_t;
#endif

struct HQShaderProgramGL_Cg : public HQBaseShaderProgramGL
{
	HQShaderParameterGL* TryCreateParameterObject(const char *parameterName);//implements base class virtual method
};
/*---------HQBaseCgShaderController---------------*/
class HQBaseCgShaderController
{
public:
	HQBaseCgShaderController();
	virtual ~HQBaseCgShaderController();
protected:
	CGcontext cgContext;
	CGprofile cgVertexProfile,cgFragmentProfile,cgGeometryProfile;

	bool CompareSemanticCg(const char *semantic1 , const char *semantic2 , size_t len);
	void RecursiveChangeSemanticCg(CGparameter param);//change semantic of vertex attribute
	char ** GetPredefineMacroArgumentsCg(const HQShaderMacro * pDefines);//convert HQShaderMacro array to Cg compiler command arguments
	void DeAllocCgArgs(char **ppC);//delete arguments array

	HQReturnVal DeActiveProgramCg(HQSharedPtr<HQBaseShaderProgramGL>& pProgram);
	HQReturnVal ActiveProgramCg(HQSharedPtr<HQBaseShaderProgramGL>& pProgram);

	HQReturnVal CreateShaderFromFileCg(HQShaderType type,
									 const char* fileName,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut);
	HQReturnVal CreateShaderFromMemoryCg(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut);

	HQReturnVal CreateProgramCg(HQSharedPtr<HQShaderObjectGL>& pVShader,
							  HQSharedPtr<HQShaderObjectGL>& pGShader,
							  HQSharedPtr<HQShaderObjectGL>& pFShader,
							  const char** uniformParameterNames,
							  hq_uint32 *pID);


	HQReturnVal SetUniformIntCg(CGparameter param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform2IntCg(CGparameter param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform3IntCg(CGparameter param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform4IntCg(CGparameter param,
						 const hq_int32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniformFloatCg(CGparameter param,const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform2FloatCg(CGparameter param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform3FloatCg(CGparameter param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniform4FloatCg(CGparameter param,
						 const hq_float32* pValues,
						 hq_uint32 numElements);
	HQReturnVal SetUniformMatrixCg( CGparameter param,
							const HQBaseMatrix4* pMatrices,
							hq_uint32 numMatrices);
	HQReturnVal SetUniformMatrixCg( CGparameter param,
							const HQBaseMatrix3x4* pMatrices,
							hq_uint32 numMatrices);

private:
#ifndef CG_IMPLICIT_LINK
	libHandle_t cgLibHandle;
#	ifndef __APPLE__
	libHandle_t cgGLLibHandle;
#	endif
#endif
	void InitCgLibrary();
};

//this controller only accepts Cg based shader
class HQCgShaderController : public HQBaseCgShaderController
{
public:
	HQCgShaderController(){};
	~HQCgShaderController(){};


	HQ_FORCE_INLINE HQReturnVal DeActiveProgram(bool isGLSL ,HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
	{
		return this->DeActiveProgramCg(pProgram);
	}
	HQ_FORCE_INLINE HQReturnVal ActiveProgram(bool isGLSL ,HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
	{
		return this->ActiveProgramCg(pProgram);
	}

	HQReturnVal CreateShaderFromFile(HQShaderType type,
									 const char* fileName,
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

	HQReturnVal CreateShaderFromFile(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* fileName,
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
		return this->SetUniformIntCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform2Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform2IntCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform3Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform3IntCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform4Int(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_int32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform4IntCg(parameter->parameter , pValues , numElements);
	}


	HQ_FORCE_INLINE HQReturnVal SetUniformFloat(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniformFloatCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform2Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform2FloatCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform3Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform3FloatCg(parameter->parameter , pValues , numElements);
	}

	HQ_FORCE_INLINE HQReturnVal SetUniform4Float(bool isGLSL,
									 const HQShaderParameterGL *parameter,
									 const hq_float32* pValues,
									 hq_uint32 numElements=1)
	{
		return this->SetUniform4FloatCg(parameter->parameter , pValues , numElements);
	}



	HQ_FORCE_INLINE HQReturnVal SetUniformMatrix(bool isGLSL,
										const HQShaderParameterGL *parameter,
										const HQBaseMatrix4* pMatrices,
										hq_uint32 numMatrices=1)
	{
		return this->SetUniformMatrixCg(parameter->parameter , pMatrices , numMatrices);
	}
	HQ_FORCE_INLINE HQReturnVal SetUniformMatrix(bool isGLSL,
										const HQShaderParameterGL *parameter,
										const HQBaseMatrix3x4* pMatrices,
										hq_uint32 numMatrices=1)
	{
		return this->SetUniformMatrixCg(parameter->parameter , pMatrices , numMatrices);
	}




};


#endif//ifndef GLES

#endif
