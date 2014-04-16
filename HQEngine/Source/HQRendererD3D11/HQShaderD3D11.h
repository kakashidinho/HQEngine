/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _SHADER_H_
#define _SHADER_H_
#include "../BaseImpl/HQClearViewportShaderCodeD3D1x.h"
#include "../HQShaderManager.h"
#include "HQLoggableObject.h"

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include "Cg/cg.h"
#include "Cg/cgD3D11.h"
#endif
#include <d3d11.h>

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#pragma comment( lib, "cg.lib" )		
#pragma comment( lib, "cgD3D11.lib" )
#endif

#include <map>
#include <list>
#include <string>

#define useV USEVSHADER
#define useG USEGSHADER
#define useP USEFSHADER
#define useVF (USEVSHADER | USEFSHADER)
#define useGF (USEGSHADER | USEFSHADER)
#define useVG (USEVSHADER | USEGSHADER)
#define useVGF (USEVSHADER | USEGSHADER | USEFSHADER)

#define HQ_DEFINE_SEMANTICS 0

struct HQShaderObjectD3D11 : public HQShaderObject, public HQBaseIDObject
{
	HQShaderObjectD3D11();
	~HQShaderObjectD3D11();

	virtual HQShaderType GetType() const { return type; }

	ID3D11DeviceChild* pD3DShader;
	ID3DBlob * pByteCodeInfo;
	HQShaderType type;
};

struct HQShaderConstBufferD3D11 : public HQUniformBuffer, public HQBaseIDObject
{
	HQShaderConstBufferD3D11(bool isDynamic, hq_uint32 size);
	HQShaderConstBufferD3D11(ID3D11Buffer *pD3DBuffer, bool isDynamic, hq_uint32 size);
	~HQShaderConstBufferD3D11();

	virtual hquint32 GetSize() const { return size; }
	virtual HQReturnVal Unmap();
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);

	ID3D11DeviceContext *pD3DContext;
	ID3D11Buffer *pD3DBuffer;
	HQLoggableObject *pLog;
	bool isDynamic;
	hq_uint32 size;
};

struct HQShaderProgramD3D11 : public HQShaderProgram, public HQBaseIDObject
{
	inline bool isUseVS() {return vertexShader != NULL;}
	inline bool isUseGS() {return geometryShader != NULL;}
	inline bool isUsePS() { return pixelShader != NULL; }

	virtual HQShaderObject * GetShader(HQShaderType type);

	HQSharedPtr<HQShaderObjectD3D11> vertexShader;
	HQSharedPtr<HQShaderObjectD3D11> geometryShader;
	HQSharedPtr<HQShaderObjectD3D11> pixelShader;
};

struct HQFixedFunctionShaderD3D11;

class HQShaderManagerD3D11 :public HQShaderManager, private HQIDItemManager<HQShaderProgramD3D11>, public HQLoggableObject
{
private:
	HQSharedPtr<HQShaderProgramD3D11> activeProgram;
	HQSharedPtr<HQShaderObjectD3D11> activeVShader,activeGShader,activePShader, activeCShader;
	
	HQSharedPtr<HQShaderConstBufferD3D11> uBufferSlots[4][D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	CGcontext cgContext;
	CGprofile cgVertexProfile,cgPixelProfile ,cgGeometryProfile;
#endif

	ID3D11Device* pD3DDevice;
	ID3D11DeviceContext* pD3DContext;
	HQIDItemManager<HQShaderObjectD3D11> shaderObjects;//danh sách shader object
	HQIDItemManager<HQShaderConstBufferD3D11> shaderConstBuffers;
	
	HQShaderObjectD3D11 clearVShader , clearGShader , clearPShader;//shaders for clearing viewport
#if !HQ_D3D_CLEAR_VP_USE_GS
	HQShaderConstBufferD3D11* clearShaderParameters;//uniform parameters for clear viewport shaders 
#endif

	D3D_FEATURE_LEVEL featureLevel;
	
	// for fixed function emulation 
	HQFixedFunctionShaderD3D11 * pFFEmu;

	void InitFFEmu();
	void ReleaseFFEmu();

	HQReturnVal ActiveFFEmu();
	HQReturnVal DeActiveFFEmu();
	bool IsFFEmuActive();
	HQReturnVal SetFFTransform(HQFFTransformMatrix type, const HQBaseMatrix4 *pMatrix);
	HQReturnVal SetFFRenderState(HQFFRenderState stateType, const void* pValue);

	//end for fixed function emulation 

	int GetNumMacros(const HQShaderMacro * pDefines);
	char ** GetPredefineMacroArgumentsCg(const HQShaderMacro * pDefines);//convert HQShaderMacro array to Cg compiler command arguments
	void DeAllocArgsCg(char **ppC);//delete arguments array
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	D3D10_SHADER_MACRO* GetPredefineMacroArgumentsHLSL(const HQShaderMacro * pDefines);//convert HQShaderMacro array to D3D macro array
	void DeAllocMacrosHLSL(D3D10_SHADER_MACRO *pM);//delete D3D macro array
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

	HQReturnVal CreateShaderFromStreamCg(HQShaderType type,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 bool isPreCompiled,
								 const char* entryFunctionName,
								 bool debugMode ,
								 HQShaderObject **pID);
	HQReturnVal CreateShaderFromMemoryCg(HQShaderType type,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 bool isPreCompiled,
								 const char* entryFunctionName,
								 bool debugMode ,
								 HQShaderObject **pID);
	HQReturnVal CreateShaderFromStreamHLSL(HQShaderType type,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 bool debugMode ,
								 HQShaderObject **pID);
	HQReturnVal CreateShaderFromMemoryHLSL(HQShaderType type,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 bool debugMode ,
								 HQShaderObject **pID);

	HQReturnVal CreateShader(HQShaderType type , ID3D10Blob *pBlob , HQShaderObjectD3D11 * sobject);

public:
	HQShaderManagerD3D11(ID3D11Device* g_pD3DDev , 
		ID3D11DeviceContext* pD3DContext,
		D3D_FEATURE_LEVEL featureLevel ,
		HQLogStream* logFileStream , 
		bool flushLog);
	~HQShaderManagerD3D11();
	
	bool IsUsingVShader(); //có đang dùng vertex shader không,hay đang dùng fixed function
	bool IsUsingGShader();//có đang dùng geometry shader không,hay đang dùng fixed function
	bool IsUsingPShader();//có đang dùng pixel/fragment shader không,hay đang dùng fixed function
	bool IsUsingShader() {return this->activeProgram != NULL;}
	
	ID3DBlob *GetCompiledVertexShader(HQShaderObject* vertexShaderID);

	HQReturnVal ActiveProgram(HQShaderProgram* programID);
	HQReturnVal ActiveComputeShader(HQShaderObject *shader);
	HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ);

	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject **pID);
	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject **pID);

	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 HQShaderObject **pID);

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 HQShaderObject **pID);

	HQReturnVal CreateShaderFromByteCodeStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 HQShaderObject **pID);

	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 HQShaderObject **pID);

	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 HQShaderObject **pID);

	HQReturnVal CreateProgram(HQShaderObject* vertexShaderID,
								HQShaderObject* pixelShaderID,
								HQShaderObject* geometryShaderID,
							  HQShaderProgram **pID);


	HQReturnVal DestroyProgram(HQShaderProgram* programID);
	void DestroyAllProgram();
	HQReturnVal DestroyShader(HQShaderObject* shaderID);
	void DestroyAllShader() ;
	void DestroyAllResource();

	
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

	hq_uint32 GetParameterIndex(HQShaderProgram* programID , 
											const char *parameterName);
	
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

	friend void cgErrorCallBack(void);

	HQReturnVal CreateUniformBuffer(hq_uint32 size, void *initData, bool isDynamic, HQUniformBuffer **pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(HQUniformBuffer* bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot, HQUniformBuffer* bufferID);

	/*----------------------------------*/
	ID3DBlob * GetCompiledClearVShader();
	void BeginClearViewport();
#if !HQ_D3D_CLEAR_VP_USE_GS
	void ChangeClearVPParams(HQColor clearColor, hqfloat32 clearDepth);
#endif
	void EndClearViewport();

	/*----------------------------------*/
	void NotifyFFRenderIfNeeded();// notify shader manager that the render device is going to draw something. Shader manager needs to update Fixed Function emulator if needed
	HQShaderObject* GetFFVertexShaderForInputLayoutCreation();//fixed function vertex shader

	bool IsFFShader(HQShaderObject* shader);
	bool IsFFProgram(HQShaderProgram* program);
	bool IsFFConstBuffer(HQUniformBuffer* buffer);
};

#endif
