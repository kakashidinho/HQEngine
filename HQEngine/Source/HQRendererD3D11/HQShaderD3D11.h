/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _SHADER_H_
#define _SHADER_H_
#include "../BaseImpl/HQClearViewportShaderCodeD3D1x.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "../HQShaderManager.h"
#include "HQCommonD3D11.h"
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

/*--------------------------------------------------------*/
#ifndef HQ_DEVICE_D3D11_CLASS_FORWARD_DECLARED
#define HQ_DEVICE_D3D11_CLASS_FORWARD_DECLARED
class HQDeviceD3D11;
#endif
struct HQFixedFunctionShaderD3D11;

/*-----------------HQShaderObjectD3D11-----------------------*/
struct HQShaderObjectD3D11 : public HQShaderObject, public HQBaseIDObject
{
	HQShaderObjectD3D11();
	~HQShaderObjectD3D11();

	virtual HQShaderType GetType() const { return type; }

	ID3D11DeviceChild* pD3DShader;
	ID3DBlob * pByteCodeInfo;
	HQShaderType type;
};

/*------------------HQShaderConstBufferD3D11------------------*/
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

/*----------------HQShaderProgramD3D11--------------------------*/
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


/*-----------------HQShaderIncludeHandlerD3D11------------------------*/
class HQShaderIncludeHandlerD3D11 : public ID3DInclude {
public:
	HQShaderIncludeHandlerD3D11();
	void SetFileManager(HQFileManager* includeFileManager) { this->includeFileManager = includeFileManager; }
	HQFileManager* GetFileManager() const { return includeFileManager; }

	STDMETHOD(Open)(THIS_ D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes);
	STDMETHOD(Close)(THIS_ LPCVOID pData);
private:
	HQFileManager* includeFileManager;
};

#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of HQBufferD3D11
#endif

/*-------------HQDrawIndirectBufferD3D11------------------*/
struct HQDrawIndirectBufferD3D11 : public HQGenericBufferD3D11 {
	HQDrawIndirectBufferD3D11(hquint32 size)
	: HQGenericBufferD3D11(HQ_DRAW_INDIRECT_BUFFER_D3D11, false, size, s_inputBoundSlotsMemManager)
	{
	}

	static HQSharedPtr<HQPoolMemoryManager> s_inputBoundSlotsMemManager;//important! must be created before any object's creation
};

/*-------------HQShaderUseOnlyBufferD3D11------------------*/
struct HQShaderUseOnlyBufferD3D11 : public HQGenericBufferD3D11 {
	HQShaderUseOnlyBufferD3D11(hquint32 size)
	: HQGenericBufferD3D11(HQ_SHADER_USE_ONLY_D3D11, false, size, s_inputBoundSlotsMemManager)
	{
	}

	static HQSharedPtr<HQPoolMemoryManager> s_inputBoundSlotsMemManager;//important! must be created before any object's creation
};

#ifdef WIN32
#	pragma warning( pop )
#endif


/*----------------HQShaderManagerD3D11------------------------------*/
class HQShaderManagerD3D11 :public HQShaderManager, private HQIDItemManager<HQShaderProgramD3D11>, public HQLoggableObject
{
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

	HQFileManager* GetIncludeFileManager() const { return this->includeHandler.GetFileManager(); }

	HQReturnVal SetIncludeFileManager(HQFileManager* fileManager) { this->includeHandler.SetFileManager(fileManager); return HQ_OK; }

	HQReturnVal ActiveProgram(HQShaderProgram* programID);
	HQReturnVal ActiveComputeShader(HQShaderObject *shader);

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


	HQReturnVal RemoveProgram(HQShaderProgram* programID);
	void RemoveAllProgram();
	HQReturnVal RemoveShader(HQShaderObject* shaderID);
	void RemoveAllShader() ;
	void RemoveAllResource();

	
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
	HQReturnVal RemoveUniformBuffer(HQUniformBuffer* bufferID);
	void RemoveAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot, HQUniformBuffer* bufferID);
	HQReturnVal SetUniformBuffer(HQShaderType stage, hq_uint32 slot, HQUniformBuffer* bufferID);

	HQReturnVal CreateBufferUAV(hquint32 numElements, hquint32 elementSize, void *initData, HQBufferUAV** ppBufferOut);

	HQReturnVal CreateComputeIndirectArgs(hquint32 numElements, void *initData, HQComputeIndirectArgsBuffer** ppBufferOut);

	HQReturnVal CreateDrawIndirectArgs(hquint32 numElements, void *initData, HQDrawIndirectArgsBuffer** ppBufferOut) ;

	HQReturnVal CreateDrawIndexedIndirectArgs(hquint32 numElements, void *initData, HQDrawIndexedIndirectArgsBuffer** ppBufferOut);

	HQReturnVal SetBufferUAVForComputeShader(hquint32 slot, HQBufferUAV * buffer, hquint32 firstElementIdx, hquint32 numElements);

	void OnTextureBindToComputeShaderUAVSlot(hquint32 slot);//unbind any buffer from compute shader's UAV slot {slot}
	
	void UnbindBufferFromAllUAVSlots(HQSharedPtr<HQGenericBufferD3D11>& pBuffer)//unbind buffer from every UAV slot
	{
		UnbindBufferFromAllUAVSlots(pBuffer.GetRawPointer());
	}
	void UnbindBufferFromAllUAVSlots(HQGenericBufferD3D11* pBuffer);//unbind buffer from every UAV slot


	HQReturnVal RemoveBufferUAV(HQBufferUAV * buffer);
	void RemoveAllBufferUAVs();

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

private:


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
		bool debugMode,
		HQShaderObject **pID);
	HQReturnVal CreateShaderFromMemoryCg(HQShaderType type,
		const char* pSourceData,
		const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
		bool isPreCompiled,
		const char* entryFunctionName,
		bool debugMode,
		HQShaderObject **pID);
	HQReturnVal CreateShaderFromStreamHLSL(HQShaderType type,
		HQDataReaderStream* dataStream,
		const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
		const char* entryFunctionName,
		bool debugMode,
		HQShaderObject **pID);
	HQReturnVal CreateShaderFromMemoryHLSL(HQShaderType type,
		const char* pSourceData,
		const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
		const char* entryFunctionName,
		bool debugMode,
		HQShaderObject **pID);

	HQReturnVal CreateShader(HQShaderType type, ID3D10Blob *pBlob, HQShaderObjectD3D11 * sobject);

	HQReturnVal CreateGenericDrawIndirectBuffer(hquint32 numElements, hquint32 elementSize, void *initData, HQBufferUAV** ppBufferOut);

	/*------------attributes--------------*/
	HQSharedPtr<HQShaderProgramD3D11> activeProgram;
	HQSharedPtr<HQShaderObjectD3D11> activeVShader, activeGShader, activePShader, activeCShader;

	HQSharedPtr<HQShaderConstBufferD3D11> uBufferSlots[4][D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
	HQGenericBufferD3D11::BufferSlot uavBufferSlots[2][D3D11_PS_CS_UAV_REGISTER_COUNT];

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	CGcontext cgContext;
	CGprofile cgVertexProfile, cgPixelProfile, cgGeometryProfile;
#endif

	HQDeviceD3D11 * pMasterDevice;

	ID3D11Device* pD3DDevice;
	ID3D11DeviceContext* pD3DContext;
	HQIDItemManager<HQShaderObjectD3D11> shaderObjects;//danh sách shader object
	HQIDItemManager<HQShaderConstBufferD3D11> shaderConstBuffers;
	HQIDItemManager<HQGenericBufferD3D11> uavBuffers;

	HQShaderObjectD3D11 clearVShader, clearGShader, clearPShader;//shaders for clearing viewport
#if !HQ_D3D_CLEAR_VP_USE_GS
	HQShaderConstBufferD3D11* clearShaderParameters;//uniform parameters for clear viewport shaders 
#endif

	D3D_FEATURE_LEVEL featureLevel;

	// for fixed function emulation 
	HQFixedFunctionShaderD3D11 * pFFEmu;

	HQShaderIncludeHandlerD3D11 includeHandler;
};

#endif
