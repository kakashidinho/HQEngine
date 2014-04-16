/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_SHADER_MANANGER_H_
#define _HQ_SHADER_MANANGER_H_

#include "HQRendererCoreType.h"
#include "HQRendererPlatformDef.h"
#include "HQReturnVal.h"
#include "HQ3DMathBasics.h"

/*---------------------------------------*/
/*
Shader manager - chỉ chấp nhận 
-Nvidia CG shading language 
-directX 10 high level shading language , 
-HQEngine extended OpenGL and OpenGL ES shading language (phiên bản mở rộng của HQEngine dành cho GLSL và GLSL ES)
*/
class HQShaderManager
{
protected:
	virtual ~HQShaderManager(){};
public:
	HQShaderManager(){};
	
	///có đang dùng vertex shader không,hay đang dùng fixed function
	virtual bool IsUsingVShader()=0;
	///có đang dùng geometry shader không,hay đang dùng fixed function
	virtual bool IsUsingGShader()=0;
	///có đang dùng pixel/fragment shader không,hay đang dùng fixed function
	virtual bool IsUsingPShader()=0;
	
	///
	///active shader program với id là {programID} ,
	///nếu programID là NULL => không dùng shader 
	///
	virtual HQReturnVal ActiveProgram(HQShaderProgram* programID) = 0;

	///
	///active compute shader. If shader is NULL => deactive compute shader
	///
	virtual HQReturnVal ActiveComputeShader(HQShaderObject *shader) = 0;

	///
	///dispatch commands from compute shader
	///
	virtual HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ) = 0;
	
	//4 method sau chỉ dùng để tạo shader từ mã nguồn ngôn ngữ Cg (ngoại trừ openGL ES device hoặc openGL device với option "GLSL-only" là ngôn ngữ HQEngine extended HQ_OPENGLES )
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject **pID) = 0;
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,//null terminated string
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject **pID) = 0;

	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject **pID)
	{
		return CreateShaderFromStream(type,
									 dataStream,
									 NULL,//ko dùng predefined macro
									 isPreCompiled,
									 entryFunctionName,
									 pID);
	}

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,//null terminated string
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObject **pID)
	{
		return CreateShaderFromMemory(type,
									 pSourceData,
									 NULL,//ko dùng predefined macro
									 isPreCompiled,
									 entryFunctionName,
									 pID);
	}
	
	//4 method sau có thể dùng để tạo shader từ mã nguồn tùy theo <compileMode> .
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 HQShaderObject **pID) = 0;
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,//nul terminated string
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 HQShaderObject **pID) = 0;
	HQReturnVal CreateShaderFromStream(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 HQDataReaderStream* dataStream,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 HQShaderObject **pID)
	{
		return CreateShaderFromStream(type,
									compileMode,
									 dataStream,
									 NULL,//ko dùng predefined macro
									 entryFunctionName,
									 pID);
	}

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,//nul terminated string
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 HQShaderObject **pID)
	{
		return CreateShaderFromMemory(type,
									compileMode,
									 pSourceData,
									 NULL,//ko dùng predefined macro
									 entryFunctionName,
									 pID);
	}

	///
	///tạo shader từ mã đã compile
	///
	virtual HQReturnVal CreateShaderFromByteCodeStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 HQShaderObject **pID) = 0;

	///
	///tạo shader từ mã đã compile
	///
	virtual HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 HQShaderObject **pID) = 0;

	///
	///tạo shader program bằng cách kết hợp các shader object:
	///-{vertexShaderID} - id của vertexshader object ,có thể là NULL nếu không dùng vertex shader. 
	///-{pixelShaderID} - id của pixelshader/fragment object ,có thể là NULL nếu không dùng pixel/fragment shader. 
	///-{geometryShaderID} - id của geometryshader object ,có thể là NULL nếu không dùng geometry shader. 
	///Ít nhất 1 trong 3 id phải là id của 1 shader object ,không phải NULL. 
	///Để tránh không tương thích, tốt nhất mỗi shader program phải có ít nhất vertexshader object và pixelshader object.
	///Các shader object có thể không tương thích nếu tạo từ các mã nguồn ngôn ngữ khác nhau ví dụ vertex shader tạo từ Cg, pixel shader tạo từ Glsl ,v.v.v.v.. 
	///
	virtual HQReturnVal CreateProgram(HQShaderObject* vertexShaderID,
								HQShaderObject* pixelShaderID,
								HQShaderObject* geometryShaderID,
								HQShaderProgram **pID)=0;

	virtual HQReturnVal DestroyProgram(HQShaderProgram* programID) = 0;
	virtual HQReturnVal DestroyShader(HQShaderObject* shaderID) = 0;
	virtual void DestroyAllProgram()=0;
	virtual void DestroyAllShader() = 0;
	virtual void DestroyAllResource()=0;//destroy both programs & shader objects


	///
	///return HQ_NOT_AVAIL_ID if parameter doesn't exist
	///
	virtual hq_uint32 GetParameterIndex(HQShaderProgram* programID,
											const char *parameterName)=0; 

	
	virtual HQReturnVal SetUniformInt(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform2Int(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform3Int(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform4Int(const char* parameterName,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	
	virtual HQReturnVal SetUniformFloat(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform2Float(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform3Float(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform4Float(const char* parameterName,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;


	virtual HQReturnVal SetUniformMatrix(const char* parameterName,
								   const HQBaseMatrix4* pMatrices,
								   hq_uint32 numMatrices=1)=0;
	virtual HQReturnVal SetUniformMatrix(const char* parameterName,
								   const HQBaseMatrix3x4* pMatrices,
								   hq_uint32 numMatrices=1)=0;
	

	inline HQReturnVal SetUniformInt(const char * parameterName,
								hq_int32 intValue)
	{
		return SetUniformInt(parameterName,&intValue,1);
	};
	inline HQReturnVal SetUniformFloat(const char * parameterName,
								hq_float32 floatValue)
	{
		return SetUniformFloat(parameterName,&floatValue,1);
	};
	inline HQReturnVal SetUniformMatrix(const char * parameterName,
								const HQBaseMatrix4& matrix)
	{
		return SetUniformMatrix(parameterName,&matrix,1);
	};

	inline HQReturnVal SetUniformMatrix(const char * parameterName,
								const HQBaseMatrix3x4& matrix)
	{
		return SetUniformMatrix(parameterName,&matrix,1);
	};

	/*Set Uniform by parameter index version*/
	virtual HQReturnVal SetUniformInt(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform2Int(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform3Int(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform4Int(hq_uint32 parameterIndex,
								 const hq_int32* pValues,
								 hq_uint32 numElements=1)=0;
	
	virtual HQReturnVal SetUniformFloat(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform2Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform3Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;
	virtual HQReturnVal SetUniform4Float(hq_uint32 parameterIndex,
								 const hq_float32* pValues,
								 hq_uint32 numElements=1)=0;


	virtual HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix4* pMatrices,
								   hq_uint32 numMatrices=1)=0;
	
	virtual HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix3x4* pMatrices,
								   hq_uint32 numMatrices=1)=0;

	inline HQReturnVal SetUniformInt(hq_uint32 parameterIndex,
								hq_int32 intValue)
	{
		return SetUniformInt(parameterIndex,&intValue,1);
	};
	inline HQReturnVal SetUniformFloat(hq_uint32 parameterIndex,
								hq_float32 floatValue)
	{
		return SetUniformFloat(parameterIndex,&floatValue,1);
	};
	inline HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								const HQBaseMatrix4& matrix)
	{
		return SetUniformMatrix(parameterIndex,&matrix,1);
	};
	inline HQReturnVal SetUniformMatrix(hq_uint32 parameterIndex,
								const HQBaseMatrix3x4& matrix)
	{
		return SetUniformMatrix(parameterIndex,&matrix,1);
	};
	
#if 1
#	define HQ_RENDER_UNIFORM_BUFFER_SUPPORT 1
	///
	///Dynamic buffer can be updated by calling Map and Unmap methods
	///
	virtual HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , HQUniformBuffer **pBufferIDOut) = 0;
	virtual HQReturnVal DestroyUniformBuffer(HQUniformBuffer* bufferID) = 0;
	virtual void DestroyAllUniformBuffers() = 0;
	///
	///Direct3d : {slot} = {buffer slot} bitwise OR với enum HQShaderType để chỉ  {buffer slot} thuộc shader stage nào. 
	///			Ví dụ muốn gắn uniform buffer vào buffer slot 3 của vertex shader , ta truyền tham số {slot} = (3 | HQ_VERTEX_SHADER).
	///
	virtual HQReturnVal SetUniformBuffer(hq_uint32 slot, HQUniformBuffer* bufferID) = 0;
	
#endif
};

#endif
