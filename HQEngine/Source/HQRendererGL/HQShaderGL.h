/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _SHADER_H_
#define _SHADER_H_
#include "../HQRenderDevice.h"
#include "HQShaderGL_Common.h"
#include "HQShaderGL_FakeUBO.h"
#include "HQShaderGL_UBO.h"
#include "HQShaderGL_GLSLController.h"


#define GLSL_SHADER_MANAGER 1

struct HQFixedFunctionShaderGL;

// for fixed function emulation 
class HQFFShaderControllerGL
{
private:
	HQFixedFunctionShaderGL * pFFEmu;

protected:
	HQFFShaderControllerGL();
	~HQFFShaderControllerGL();

	void ReleaseFFEmu();
	void RestoreFFEmu();

	HQReturnVal ActiveFFEmu();
	HQReturnVal DeActiveFFEmu();
	bool IsFFEmuActive();
	HQReturnVal SetFFTransform(HQFFTransformMatrix type, const HQBaseMatrix4 *pMatrix);
	HQReturnVal SetFFRenderState(HQFFRenderState stateType, const void* pValue);

	void NotifyFFRender();
};

/*-----------------template class Shader manager---------------*/
///
///Note: due to historical reason, I used Cg before, hence I used controller class to 
///separate between 2 controller type, one for Cg shader and one for GLSL shader
///
template <class ShaderController , class BaseShaderManagerClass>
class HQShaderManagerGL: public BaseShaderManagerClass, public HQFFShaderControllerGL
{
private:
	ShaderController shaderController;
public:
	HQShaderManagerGL(HQLogStream* logFileStream , const char*logPrefix , bool flushLog);
	~HQShaderManagerGL();
	
	
	

	HQReturnVal ActiveProgram(HQShaderProgram* programID);
	HQReturnVal ActiveComputeShader(HQShaderObject *shader);

	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject** pID);
	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject** pID);

	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 HQShaderObject** pID);

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 HQShaderObject** pID);
	HQReturnVal CreateProgram(HQShaderObject* vertexShaderID,
								HQShaderObject* pixelShaderID,
								HQShaderObject* geometryShaderID,
								HQShaderProgram** pID);

	HQReturnVal SetUniformBool(const char* parameterName,
						 const HQBool* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform2Bool(const char* parameterName,
						 const HQBool* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform3Bool(const char* parameterName,
						 const HQBool* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform4Bool(const char* parameterName,
						 const HQBool* pValues,
						 hq_uint32 numElements=1);
	
	HQReturnVal SetUniformInt(const char* parameterName,
						 const int* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform2Int(const char* parameterName,
						 const int* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform3Int(const char* parameterName,
						 const int* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform4Int(const char* parameterName,
						 const int* pValues,
						 hq_uint32 numElements=1);
	
	HQReturnVal SetUniformFloat(const char* parameterName,
						 const hq_float32* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform2Float(const char* parameterName,
						 const hq_float32* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform3Float(const char* parameterName,
						 const hq_float32* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniform4Float(const char* parameterName,
						 const hq_float32* pValues,
						 hq_uint32 numElements=1);
	HQReturnVal SetUniformMatrix( const char* parameterName,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices);
	HQReturnVal SetUniformMatrix( const char* parameterName,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices);

	/*-------parameter index version-----------*/

	
	HQReturnVal SetUniformBool(hq_uint32 parameterIndex,
								 const HQBool* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform2Bool(hq_uint32 parameterIndex,
								 const HQBool* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform3Bool(hq_uint32 parameterIndex,
								 const HQBool* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform4Bool(hq_uint32 parameterIndex,
								 const HQBool* pValues,
								 hq_uint32 numElements=1);
	
	HQReturnVal SetUniformInt(hq_uint32 parameterIndex,
								 const int* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform2Int(hq_uint32 parameterIndex,
								 const int* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform3Int(hq_uint32 parameterIndex,
								 const int* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform4Int(hq_uint32 parameterIndex,
								 const int* pValues,
								 hq_uint32 numElements=1);
	
	HQReturnVal SetUniformFloat(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform2Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform3Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1);
	HQReturnVal SetUniform4Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1);


	HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix4* pMatrices,
								   hq_uint32 numMatrices=1);
	HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix3x4* pMatrices,
								   hq_uint32 numMatrices=1);


	void Commit();

	//device lost handling methods
	void OnLost() ;
	void OnReset() ;
	
};


HQBaseShaderManagerGL * HQCreateShaderManager(int shaderManagerType , HQLogStream *logFileStream , bool flushLog);



#include "HQShaderGL_template_impl.h"

#endif
