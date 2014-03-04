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

struct HQShaderObjectD3D11
{
	HQShaderObjectD3D11();
	~HQShaderObjectD3D11();

	ID3D11DeviceChild* pD3DShader;
	ID3DBlob * pByteCodeInfo;
	HQShaderType type;
};

struct HQShaderConstBufferD3D11
{
	HQShaderConstBufferD3D11(bool isDynamic, hq_uint32 size);
	HQShaderConstBufferD3D11(ID3D11Buffer *pD3DBuffer, bool isDynamic, hq_uint32 size);
	~HQShaderConstBufferD3D11();

	ID3D11Buffer *pD3DBuffer;
	bool isDynamic;
	hq_uint32 size;
};

struct HQShaderProgramD3D11
{
	inline bool isUseVS() {return vertexShader != NULL;}
	inline bool isUseGS() {return geometryShader != NULL;}
	inline bool isUsePS() {return pixelShader != NULL;}

	HQSharedPtr<HQShaderObjectD3D11> vertexShader;
	HQSharedPtr<HQShaderObjectD3D11> geometryShader;
	HQSharedPtr<HQShaderObjectD3D11> pixelShader;

	hquint32 vertexShaderID;
	hquint32 geometryShaderID;
	hquint32 pixelShaderID;
};

struct HQFixedFunctionShaderD3D11;

class HQShaderManagerD3D11:public HQShaderManager,private HQItemManager<HQShaderProgramD3D11> , public HQLoggableObject
{
private:
	HQSharedPtr<HQShaderProgramD3D11> activeProgram;
	HQSharedPtr<HQShaderObjectD3D11> activeVShader,activeGShader,activePShader;
	
	HQSharedPtr<HQShaderConstBufferD3D11> uBufferSlots[3][D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	CGcontext cgContext;
	CGprofile cgVertexProfile,cgPixelProfile ,cgGeometryProfile;
#endif

	ID3D11Device* pD3DDevice;
	ID3D11DeviceContext* pD3DContext;
	HQItemManager<HQShaderObjectD3D11> shaderObjects;//danh sách shader object
	HQItemManager<HQShaderConstBufferD3D11> shaderConstBuffers;
	
	HQShaderObjectD3D11 clearVShader , clearGShader , clearPShader;//shaders for clearing viewport
#if !HQ_D3D_CLEAR_VP_USE_GS
	hquint32 clearShaderParameters;//uniform parameters for clear viewport shaders 
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

	HQReturnVal CreateShaderFromFileCg(HQShaderType type,
								 const char* fileName,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 bool isPreCompiled,
								 const char* entryFunctionName,
								 bool debugMode ,
								 hq_uint32 *pID);
	HQReturnVal CreateShaderFromMemoryCg(HQShaderType type,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 bool isPreCompiled,
								 const char* entryFunctionName,
								 bool debugMode ,
								 hq_uint32 *pID);
	HQReturnVal CreateShaderFromFileHLSL(HQShaderType type,
								 const char* fileName,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 bool debugMode ,
								 hq_uint32 *pID);
	HQReturnVal CreateShaderFromMemoryHLSL(HQShaderType type,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 bool debugMode ,
								 hq_uint32 *pID);

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
	
	ID3DBlob *GetCompiledVertexShader(hq_uint32 vertexShaderID);

	HQReturnVal ActiveProgram(hq_uint32 programID);

	HQReturnVal CreateShaderFromFile(HQShaderType type,
									 const char* fileName,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID);
	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID);

	HQReturnVal CreateShaderFromFile(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* fileName,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 hq_uint32 *pID);

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 hq_uint32 *pID);

	HQReturnVal CreateShaderFromByteCodeFile(HQShaderType type,
									 const char* file,
									 hq_uint32 *pID);

	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 hq_uint32 *pID);

	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 hq_uint32 *pID);

	HQReturnVal CreateProgram(hq_uint32 vertexShaderID,
							  hq_uint32 pixelShaderID,
							  hq_uint32 geometryShaderID,
							  const char** uniformParameterNames,
							  hq_uint32 *pID);


	HQReturnVal DestroyProgram(hq_uint32 programID);
	void DestroyAllProgram();
	HQReturnVal DestroyShader(hq_uint32 shaderID);
	void DestroyAllShader() ;
	void DestroyAllResource();

	hq_uint32 GetShader(hq_uint32 programID, HQShaderType shaderType);
	
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

	hq_uint32 GetParameterIndex(hq_uint32 programID , 
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

	HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(hq_uint32 bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID );
	HQReturnVal MapUniformBuffer(hq_uint32 bufferID , void **ppData);
	HQReturnVal UnmapUniformBuffer(hq_uint32 bufferID);
	HQReturnVal UpdateUniformBuffer(hq_uint32 bufferID, const void * pData);

	/*----------------------------------*/
	ID3DBlob * GetCompiledClearVShader();
	void BeginClearViewport();
#if !HQ_D3D_CLEAR_VP_USE_GS
	void ChangeClearVPParams(HQColor clearColor, hqfloat32 clearDepth);
#endif
	void EndClearViewport();

	/*----------------------------------*/
	void NotifyFFRenderIfNeeded();// notify shader manager that the render device is going to draw something. Shader manager needs to update Fixed Function emulator if needed
	hquint32 GetFFVertexShaderForInputLayoutCreation();//fixed function vertex shader

	bool IsFFShader(hquint32 shader);
	bool IsFFProgram(hquint32 program);
	bool IsFFConstBuffer(hquint32 buffer);
};

#endif
