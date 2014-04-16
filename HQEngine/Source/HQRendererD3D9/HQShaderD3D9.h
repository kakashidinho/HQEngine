/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _SHADER_H_
#define _SHADER_H_
#include "../HQShaderManager.h"
#include "../HQLoggableObject.h"
#if _MSC_VER < 1700
#include <hash_map>
#define hash_map_type stdext::hash_map
#else
#include <unordered_map>
#define hash_map_type std::unordered_map
#endif

#include "Cg/cg.h"
#include "Cg/cgD3D9.h"
#include <d3d9.h>

#pragma comment( lib, "cg.lib" )		
#pragma comment( lib, "cgD3D9.lib" )

#define HQ_TRANSLATE_CG_TO_HLSL 0


struct HQShaderVarParserInfoD3D9;

struct HQParameterD3D9
{
	UINT parameterReg[2];//parameter's register index both in vertex shader and pixel shader
};

#if HQ_TRANSLATE_CG_TO_HLSL
typedef ID3DXConstantTable HQConstantTableD3D9;
#else
struct HQConstantTableD3D9;
#endif

struct HQShaderObjectD3D9: public HQShaderObject, public HQBaseIDObject
{
	HQShaderObjectD3D9();
	~HQShaderObjectD3D9();

	virtual HQShaderType GetType() const {
		return type;
	}

	union {
		IDirect3DVertexShader9* vshader;
		IDirect3DPixelShader9* pshader;
	};

	HQConstantTableD3D9 *consTable;

	HQShaderType type;
};



struct HQShaderProgramD3D9 : public HQShaderProgram, public HQBaseIDObject
{
	virtual HQShaderObject * GetShader(HQShaderType type) {
		switch (type)
		{
		case HQ_VERTEX_SHADER:
			return vertexShader.GetRawPointer();
		case HQ_PIXEL_SHADER:
			return pixelShader.GetRawPointer();
		default:
			return NULL;
		}
	}

	HQSharedPtr<HQShaderObjectD3D9> vertexShader;
	HQSharedPtr<HQShaderObjectD3D9> pixelShader;
	HQClosedStringPrimeHashTable<hquint32> parameterIndexes;
	HQItemManager<HQParameterD3D9> parameters;
};

#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of HQSysMemBuffer
#endif

struct HQShaderConstBufferD3D9: public HQSysMemBuffer, public HQUniformBuffer, public HQBaseIDObject
{
	HQShaderConstBufferD3D9(HQSysMemBuffer::Listener* listener, bool isDynamic, hq_uint32 size);
	~HQShaderConstBufferD3D9();

	bool isDynamic;
	typedef HQLinkedList<hquint32, HQPoolMemoryManager> BufferSlotList;
	BufferSlotList boundSlots;//list of slots that this buffer bound to
};

#ifdef WIN32
#	pragma warning( pop )
#endif

class HQShaderManagerD3D9 :public HQShaderManager, private HQIDItemManager<HQShaderProgramD3D9>, public HQLoggableObject, public HQSysMemBuffer::Listener
{
public:
	HQShaderManagerD3D9(LPDIRECT3DDEVICE9 g_pD3DDev,HQLogStream* logFileStream ,bool flushLog);
	~HQShaderManagerD3D9();

	bool IsUsingVShader(); //có đang dùng vertex shader không,hay đang dùng fixed function
	bool IsUsingGShader();//có đang dùng geometry shader không,hay đang dùng fixed function
	bool IsUsingPShader();//có đang dùng pixel/fragment shader không,hay đang dùng fixed function
	bool IsUsingShader() {return this->activeProgram != NULL;}

	HQReturnVal ActiveProgram(HQShaderProgram* programID);
	HQReturnVal ActiveComputeShader(HQShaderObject *shader) { return HQ_FAILED; }

	HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ) { return HQ_FAILED; }

	void Commit();//this method should be called before draw any thing

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

	///
	///tạo shader từ mã đã compile
	///
	HQReturnVal CreateShaderFromByteCodeStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 HQShaderObject **pID)
	{
		return HQ_FAILED;
	}

	///
	///tạo shader từ mã đã compile
	///
	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 HQShaderObject **pID)
	{
		return HQ_FAILED;
	}

	HQReturnVal CreateProgram(HQShaderObject * vertexShaderID,
								HQShaderObject * pixelShaderID,
								HQShaderObject * geometryShaderID,
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

	/*------fixed function shader---------*/
	
	HQReturnVal SetRenderState(hq_uint32 stateType , const hq_int32 *value);

	/*------------------------------------*/

	friend void cgErrorCallBack(void);

	HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , HQUniformBuffer **pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(HQUniformBuffer* bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot ,  HQUniformBuffer* bufferID );

	void OnResetDevice();
private:
	struct BufferSlotInfo;

	HQSharedPtr<HQShaderProgramD3D9> activeProgram; 

	HQSharedPtr<HQShaderObjectD3D9> activeVShader,activePShader;

	CGcontext cgContext;
	CGprofile cgVertexProfile,cgPixelProfile;

	LPDIRECT3DDEVICE9 pD3DDevice;
	HQIDItemManager<HQShaderObjectD3D9> shaderObjects;//danh sách shader object

	HQIDItemManager<HQShaderConstBufferD3D9> shaderConstBuffers;//const buffer list

	BufferSlotInfo* vshaderConstSlots;
	BufferSlotInfo* pshaderConstSlots;
	
	//fixed function controlling values
	D3DTEXTUREOP firstStageOp[2];//color & alpha op

	char ** GetCompileArguments(const HQShaderMacro * pDefines, bool debug);//convert HQShaderMacro array to Cg compiler command arguments
	void DeAlloc(char **ppC);//delete arguments array

	HQReturnVal CreateShaderFromStreamEx(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const HQShaderMacro *pDefines,
									 const char **args,
									 bool debugMode,
									 HQShaderObject **pID
									 );
	HQReturnVal CreateShaderFromMemoryEx(HQShaderType type,
									 const char* pSourceData,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const HQShaderMacro *pDefines,
									 const char **args,
									 bool debugMode,
									 HQShaderObject **pID);

	HQSharedPtr<HQParameterD3D9> GetUniformParam(HQSharedPtr<HQShaderProgramD3D9>& pProgram,const char* parameterName);
	hq_uint32 GetParameterIndex(HQSharedPtr<HQShaderProgramD3D9>& pProgram, 
									const char *parameterName);

	template <size_t vecSize> void SetD3DVShaderConstantI(hquint32 startReg, 
													 const int* pValues,
													 hq_uint32 numElements);

	template <> void SetD3DVShaderConstantI<4>(hquint32 startReg, 
											const int* pValues,
											hq_uint32 numElements);
	
	
	template <size_t vecSize> void SetD3DVShaderConstantF(hquint32 startReg, 
													 const float* pValues,
													 hq_uint32 numElements);

	template <> void SetD3DVShaderConstantF<4>(hquint32 startReg, 
											 const float* pValues,
											 hq_uint32 numElements);

	template <size_t vecSize> void SetD3DPShaderConstantI(hquint32 startReg, 
													 const int* pValues,
													 hq_uint32 numElements);

	
	
	template <size_t vecSize> void SetD3DPShaderConstantF(hquint32 startReg, 
													 const float* pValues,
													 hq_uint32 numElements);


	hquint32 GetD3DConstantStartRegister(HQConstantTableD3D9* table, const char* name);

	virtual void BufferChangeEnded(HQSysMemBuffer* pConstBuffer);//implement HQSysMemBuffer::Listener

	void MarkBufferSlotDirty(hquint32 index);//mark the buffer slot dirty for all dependent shaders
	void MarkAllBufferSlotsDirtyForVShader(HQShaderObjectD3D9* shader);//mark all buffer slots associated with specified vertex shader dirty
	void MarkAllBufferSlotsDirtyForPShader(HQShaderObjectD3D9* shader);//mark all buffer slots associated with specified pixel shader dirty
};

#endif
