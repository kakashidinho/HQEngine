/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _SHADER_H_
#define _SHADER_H_
#include "../HQShaderManager.h"
#include "../BaseImpl/HQShaderParameterIndexTable.h"
#include "../HQLoggableObject.h"
#include "Cg/cg.h"
#include "Cg/cgD3D9.h"
#include <d3d9.h>

#pragma comment( lib, "cg.lib" )		
#pragma comment( lib, "cgD3D9.lib" )


struct HQParameterD3D9
{
	CGparameter parameter[2];//parameter both in vertex shader and pixel shader
	CGtype type;
};


struct HQShaderObjectD3D9
{
	HQShaderObjectD3D9()
	{
		program = NULL;
	}
	~HQShaderObjectD3D9()
	{
		if (program)
			cgDestroyProgram(program);
	}

	CGprogram program;
	HQShaderType type;
};



struct HQShaderProgramD3D9
{
	hquint32 vertexShaderID;
	hquint32 pixelShaderID;

	HQSharedPtr<HQShaderObjectD3D9> vertexShader;
	HQSharedPtr<HQShaderObjectD3D9> pixelShader;
	HQShaderParamIndexTable parameterIndexes;
	HQItemManager<HQParameterD3D9> parameters;
};

class HQShaderManagerD3D9:public HQShaderManager,private HQItemManager<HQShaderProgramD3D9> , public HQLoggableObject
{
private:

	HQSharedPtr<HQShaderProgramD3D9> activeProgram; 
#if defined _DEBUG || defined DEBUG
	hq_uint32 activeProgramID;
#endif
	HQSharedPtr<HQShaderObjectD3D9> activeVShader,activePShader;

	CGcontext cgContext;
	CGprofile cgVertexProfile,cgPixelProfile;

	LPDIRECT3DDEVICE9 pD3DDevice;
	HQItemManager<HQShaderObjectD3D9> shaderObjects;//danh sách shader object
	
	//fixed function controlling values
	D3DTEXTUREOP firstStageOp[2];//color & alpha op

	char ** GetPredefineMacroArguments(const HQShaderMacro * pDefines);//convert HQShaderMacro array to Cg compiler command arguments
	void DeAlloc(char **ppC);//delete arguments array

	HQReturnVal CreateShaderFromFileEx(HQShaderType type,
									 const char* fileName,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const char **args,
									 bool debugMode,
									 hq_uint32 *pID
									 );
	HQReturnVal CreateShaderFromMemoryEx(HQShaderType type,
									 const char* pSourceData,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const char **args,
									 bool debugMode,
									 hq_uint32 *pID);

	HQSharedPtr<HQParameterD3D9> GetUniformParam(HQSharedPtr<HQShaderProgramD3D9>& pProgram,const char* parameterName);
	hq_uint32 GetParameterIndex(HQSharedPtr<HQShaderProgramD3D9>& pProgram, 
									const char *parameterName);
public:
	HQShaderManagerD3D9(LPDIRECT3DDEVICE9 g_pD3DDev,HQLogStream* logFileStream ,bool flushLog);
	~HQShaderManagerD3D9();

	bool IsUsingVShader(); //có đang dùng vertex shader không,hay đang dùng fixed function
	bool IsUsingGShader();//có đang dùng geometry shader không,hay đang dùng fixed function
	bool IsUsingPShader();//có đang dùng pixel/fragment shader không,hay đang dùng fixed function
	bool IsUsingShader() {return this->activeProgram != NULL;}

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

	///
	///tạo shader từ mã đã compile
	///
	HQReturnVal CreateShaderFromByteCodeFile(HQShaderType type,
									 const char* file,
									 hq_uint32 *pID)
	{
		return HQ_FAILED;
	}

	///
	///tạo shader từ mã đã compile
	///
	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 hq_uint32 *pID)
	{
		return HQ_FAILED;
	}

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

	/*------fixed function shader---------*/
	
	HQReturnVal SetRenderState(hq_uint32 stateType , const hq_int32 *value);

	/*------------------------------------*/

	friend void cgErrorCallBack(void);

	HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(hq_uint32 bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID );
	HQReturnVal MapUniformBuffer(hq_uint32 bufferID , void **ppData);
	HQReturnVal UnmapUniformBuffer(hq_uint32 bufferID);
	HQReturnVal UpdateUniformBuffer(hq_uint32 bufferID, const void * pData);

	void OnResetDevice();
};

#endif
