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
#include "HQFileManager.h"

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
	///set file manager for loading files in include directives during compilation
	///
	virtual HQReturnVal SetIncludeFileManager(HQFileManager* fileManager) = 0;

	
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

	virtual HQReturnVal RemoveProgram(HQShaderProgram* programID) = 0;
	virtual HQReturnVal RemoveShader(HQShaderObject* shaderID) = 0;
	virtual void RemoveAllProgram()=0;
	virtual void RemoveAllShader() = 0;
	virtual void RemoveAllResource()=0;//destroy both programs & shader objects


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
	virtual HQReturnVal RemoveUniformBuffer(HQUniformBuffer* bufferID) = 0;
	virtual void RemoveAllUniformBuffers() = 0;
	///
	///Direct3d : {slot} = {buffer slot} bitwise OR với enum HQShaderType để chỉ  {buffer slot} thuộc shader stage nào. 
	///			Ví dụ muốn gắn uniform buffer vào buffer slot 3 của vertex shader , ta truyền tham số {slot} = (3 | HQ_VERTEX_SHADER).
	///
	virtual HQReturnVal SetUniformBuffer(hq_uint32 slot, HQUniformBuffer* bufferID) = 0;
	
#endif

	///
	///create a buffer containing indirect compute arguments. 
	///buffer contains an array of {numElements} elements, each element is in form {uint numGroupX, uint numGroupY, uint numGroupZ}
	///
	virtual HQReturnVal CreateComputeIndirectArgs(hquint32 numElements, void *initData, HQComputeIndirectArgsBuffer** ppBufferOut) = 0;

	///
	///create a buffer containing indirect draw instance arguments. 
	///buffer contains an array of {numElements} elements, each element is in form 
	///{uint number_of_vertices_per_instance, uint number_of_instances, uint first_vertex, uint first_instance}
	///
	virtual HQReturnVal CreateDrawIndirectArgs(hquint32 numElements, void *initData, HQDrawIndirectArgsBuffer** ppBufferOut) = 0;

	///
	///create a buffer containing indirect draw indexed instance arguments . 
	///buffer contains an array of {numElements} elements, each element is in form 
	///{uint number_of_indices_per_instance, uint number_of_instances, uint first_index, uint/int first_vertex, uint first_instance}
	///
	virtual HQReturnVal CreateDrawIndexedIndirectArgs(hquint32 numElements, void *initData, HQDrawIndexedIndirectArgsBuffer** ppBufferOut) = 0;

	///
	///Set UAV buffer to be read and written by compute shader. 
	///Direct3D 11: {slot} is UAV slot. max number of slots is 64. 
	///			Note: setting this will unset previously bound UAV texture from the same slot. 
	///				  indirect compute/draw buffer is viewed as RWBuffer<uint> in shader, while vertex/index is viewed as RWByteAddressBuffer
	///OpenGL: {slot} is shader storage slot. max number of slots is obtained by HQRenderDevice::GetMaxShaderBufferUAVs(). 
	///			Note: buffer is viewed as storage block in shader
	///
	virtual HQReturnVal SetBufferUAVForComputeShader(hquint32 slot, HQBufferUAV * buffer, hquint32 firstElementIdx, hquint32 numElements) = 0;

	virtual HQReturnVal RemoveBufferUAV(HQBufferUAV * buffer) = 0;
	virtual void RemoveAllBufferUAVs() = 0;
};

#endif
