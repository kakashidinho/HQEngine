/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_FF_SHADER_MAN_H
#define HQ_FF_SHADER_MAN_H
#include "HQShaderGL_Common.h"
#include "../HQRendererCoreTypeFixedFunction.h"
#include "../HQLoggableObject.h"

class HQFixedFunctionShaderManagerGL : public HQBaseShaderManagerGL , public HQA16ByteObject , public HQLoggableObject
{
public:
	HQFixedFunctionShaderManagerGL(HQLogStream *logStream, bool flushLog);
	~HQFixedFunctionShaderManagerGL();

	virtual bool IsUsingVShader() {return HQ_FAILED;}
	
	virtual bool IsUsingGShader(){return HQ_FAILED;}
	
	virtual bool IsUsingPShader(){return HQ_FAILED;}
	
	virtual hq_uint32 GetShader(hq_uint32 programID, HQShaderType shaderType) 
	{
		return NULL_VSHADER;
	}
	
	virtual HQReturnVal ActiveProgram(hq_uint32 programID){return HQ_FAILED;}
	
	
	virtual HQReturnVal CreateShaderFromFile(HQShaderType type,
									 const char* fileName,
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID)
	{return HQ_FAILED;}
	
	
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,//null terminated string
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID)
	{return HQ_FAILED;}

	
	
	virtual HQReturnVal CreateShaderFromFile(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* fileName,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 hq_uint32 *pID)
	{return HQ_FAILED;}
	
	
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,//nul terminated string
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 hq_uint32 *pID)
	{return HQ_FAILED;}
	

	virtual HQReturnVal CreateProgram(hq_uint32 vertexShaderID,
							  hq_uint32 pixelShaderID,
							  hq_uint32 geometryShaderID,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)
	{return HQ_FAILED;}

	virtual HQReturnVal DestroyProgram(hq_uint32 programID){return HQ_FAILED;}
	virtual HQReturnVal DestroyShader(hq_uint32 shaderID) {return HQ_FAILED;}
	virtual void DestroyAllProgram() {}
	virtual void DestroyAllShader() {}
	virtual void DestroyAllResource() {}
	
	virtual hq_uint32 GetParameterIndex(hq_uint32 programID , 
											const char *parameterName)
	{return NOT_AVAIL_ID;}

	
	virtual HQReturnVal SetUniformInt(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform2Int(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform3Int(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform4Int(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	
	virtual HQReturnVal SetUniformFloat(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform2Float(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform3Float(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform4Float(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}


	virtual HQReturnVal SetUniformMatrix(const char* parameterName,
								   const HQBaseMatrix4* pMatrices,
								   hq_uint32 numMatrices=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniformMatrix(const char* parameterName,
								   const HQBaseMatrix3x4* pMatrices,
								   hq_uint32 numMatrices=1)
	{return HQ_FAILED;}
	

	/*Set Uniform by parameter index version*/
	//set render state
	virtual HQReturnVal SetUniformInt(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1);
	virtual HQReturnVal SetUniform2Int(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform3Int(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform4Int(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	
	virtual HQReturnVal SetUniformFloat(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform2Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform3Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}
	virtual HQReturnVal SetUniform4Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)
	{return HQ_FAILED;}

	
	//set state matrix
	virtual HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix4* pMatrices,
								   hq_uint32 numMatrices=1);
	
	virtual HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix3x4* pMatrices,
								   hq_uint32 numMatrices=1)
	{
		return HQ_FAILED;
	}
private:

	void SetLightPosition(unsigned int light);//set light position in world space
	void SetLight(unsigned int light , HQFFLight* lightInfo);
	void SetMaterial(HQFFMaterial *material);
	void SetModelViewMatrix();

	hq_float32 lightPosition[8][4];//for 8 lights
	HQColor materialSpecular;

	HQMatrix4 *world;
	HQMatrix4 *view;
};

#endif
