/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
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


	virtual HQReturnVal SetIncludeFileManager(HQFileManager* fileManager) { return HQ_FAILED; }
	
	virtual HQReturnVal ActiveProgram(HQShaderProgram* programID){return HQ_FAILED;}
	virtual HQReturnVal ActiveComputeShader(HQShaderObject *shader) { return HQ_FAILED; }
	virtual HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ) { return HQ_FAILED; }
	
	
	virtual HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject** pID)
	{return HQ_FAILED;}
	
	
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,//null terminated string
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject** pID)
	{return HQ_FAILED;}

	
	
	virtual HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 HQShaderObject** pID)
	{return HQ_FAILED;}
	
	
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,//nul terminated string
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 HQShaderObject** pID)
	{return HQ_FAILED;}
	

	virtual HQReturnVal CreateProgram(HQShaderObject* vertexShaderID,
								HQShaderObject* pixelShaderID,
								HQShaderObject* geometryShaderID,
								HQShaderProgram **pID)
	{return HQ_FAILED;}

	virtual HQReturnVal RemoveProgram(HQShaderProgram* programID){return HQ_FAILED;}
	virtual HQReturnVal RemoveShader(HQShaderObject* shaderID) {return HQ_FAILED;}
	virtual void RemoveAllProgram() {}
	virtual void RemoveAllShader() {}
	virtual void RemoveAllResource() {}
	
	virtual hq_uint32 GetParameterIndex(HQShaderProgram* programID , 
											const char *parameterName)
	{return HQ_NOT_AVAIL_ID;}

	
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


	virtual HQReturnVal CreateUniformBuffer(void *initData, hq_uint32 size, bool isDynamic, HQUniformBuffer **pBufferIDOut)
	{
		return HQ_FAILED;
	}
	virtual HQReturnVal RemoveUniformBuffer(HQUniformBuffer* bufferID)
	{
		return HQ_FAILED;
	}
	virtual void RemoveAllUniformBuffers()  {
	}
	
	virtual HQReturnVal SetUniformBuffer(hq_uint32 slot ,  HQUniformBuffer* bufferID ) 
	{
		return HQ_FAILED;
	}

	virtual HQReturnVal SetUniformBuffer(HQShaderType stage, hq_uint32 slot, HQUniformBuffer* bufferID) {
		return HQ_FAILED;
	}

	virtual HQReturnVal CreateBufferUAV(void* initData, hquint32 elementSize, hquint32 numElements, HQBufferUAV** ppBufferOut) { return HQ_FAILED; }

	virtual HQReturnVal CreateComputeIndirectArgs(void* initData, hquint32 numElements, HQComputeIndirectArgsBuffer** ppBufferOut)  { return HQ_FAILED; }

	virtual HQReturnVal CreateDrawIndirectArgs(void* initData, hquint32 numElements, HQDrawIndirectArgsBuffer** ppBufferOut)  { return HQ_FAILED; }

	virtual HQReturnVal CreateDrawIndexedIndirectArgs(void* initData, hquint32 numElements, HQDrawIndexedIndirectArgsBuffer** ppBufferOut)  { return HQ_FAILED; }

	virtual HQReturnVal SetBufferUAVForGraphicsShader(hquint32 slot, HQBufferUAV * buffer, hquint32 firstElementIdx, hquint32 numElements)  { return HQ_FAILED; }
	virtual HQReturnVal SetBufferUAVForComputeShader(hquint32 slot, HQBufferUAV * buffer, hquint32 firstElementIdx, hquint32 numElements)  { return HQ_FAILED; }

	virtual HQReturnVal RemoveBufferUAV(HQBufferUAV * buffer) { return HQ_FAILED; }
	virtual void RemoveAllBufferUAVs() {}

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
