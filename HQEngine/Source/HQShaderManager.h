/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_SHADER_MANANGER_H_
#define _HQ_SHADER_MANANGER_H_

#include "HQRendererCoreType.h"
#include "HQRendererPlatformDef.h"
#include "HQReturnVal.h"
#include "HQUtilMath.h"

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
	///nếu programID là NOT_USE_SHADER => không dùng shader 
	///
	virtual HQReturnVal ActiveProgram(hq_uint32 programID)=0;
	
	//4 method sau chỉ dùng để tạo shader từ mã nguồn ngôn ngữ Cg (ngoại trừ openGL ES device hoặc openGL device với option "GLSL-only" là ngôn ngữ HQEngine extended GLES )
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromFile(HQShaderType type,
									 const char* fileName,
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID)=0;
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,//null terminated string
									 const HQShaderMacro * pDefines,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID)=0;

	HQReturnVal CreateShaderFromFile(HQShaderType type,
									 const char* fileName,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID)
	{
		return CreateShaderFromFile(type,
									 fileName,
									 NULL,//ko dùng predefined macro
									 isPreCompiled,
									 entryFunctionName,
									 pID);
	}

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 const char* pSourceData,//null terminated string
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 hq_uint32 *pID)
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
	virtual HQReturnVal CreateShaderFromFile(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* fileName,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 hq_uint32 *pID)=0;
	
	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,//nul terminated string
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 hq_uint32 *pID)=0;
	HQReturnVal CreateShaderFromFile(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* fileName,
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 hq_uint32 *pID)
	{
		return CreateShaderFromFile(type,
									compileMode,
									 fileName,
									 NULL,//ko dùng predefined macro
									 entryFunctionName,
									 pID);
	}

	HQReturnVal CreateShaderFromMemory(HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char* pSourceData,//nul terminated string
									 const char* entryFunctionName,//should be "main" if language is GLSL
									 hq_uint32 *pID)
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
	virtual HQReturnVal CreateShaderFromByteCodeFile(HQShaderType type,
									 const char* file,
									 hq_uint32 *pID)=0;

	///
	///tạo shader từ mã đã compile
	///
	virtual HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 hq_uint32 *pID) =0;

	///
	///tạo shader program bằng cách kết hợp các shader object:
	///-{vertexShaderID} - id của vertexshader object ,có thể là NOT_USE_VSHADER nếu không dùng vertex shader. 
	///-{pixelShaderID} - id của pixelshader/fragment object ,có thể là NOT_USE_PSHADER nếu không dùng pixel/fragment shader. 
	///-{geometryShaderID} - id của geometryshader object ,có thể là NOT_USE_GSHADER nếu không dùng geometry shader. 
	///Ít nhất 1 trong 3 id phải là id của 1 shader object ,không phải NOT_USE_*SHADER. 
	///Để tránh không tương thích, tốt nhất mỗi shader program phải có ít nhất vertexshader object và pixelshader object.
	///Các shader object có thể không tương thích nếu tạo từ các mã nguồn ngôn ngữ khác nhau ví dụ vertex shader tạo từ Cg, pixel shader tạo từ Glsl ,v.v.v.v.. 
	///-{unifromParameterNames} là con trỏ đến danh sách các chuỗi tên của các biến uniform trong shader program,
	///danh sách kết thúc bằng phần tử NULL,con trỏ danh sách này có thể NULL (coi như rỗng). 
	///Danh sách này nhằm khởi tạo danh sách các biến uniform dc dùng trong shader program. Nếu trong danh sách này không chứa
	///biến mà sau này dùng đến trong các method SetUniform* , shader manager sẽ cố thử thêm biến vào danh sách biến uniform của shader program
	///nếu biến này thật sự tồn tại.
	///
	virtual HQReturnVal CreateProgram(hq_uint32 vertexShaderID,
							  hq_uint32 pixelShaderID,
							  hq_uint32 geometryShaderID,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)=0;

	virtual HQReturnVal DestroyProgram(hq_uint32 programID)=0;
	virtual HQReturnVal DestroyShader(hq_uint32 shaderID) = 0;
	virtual void DestroyAllProgram()=0;
	virtual void DestroyAllShader() = 0;
	virtual void DestroyAllResource()=0;//destroy both programs & shader objects
	
	///
	///return ID of shader used to create program. 
	///Be careful, if shader object is destroyed, returned shader ID is invalid
	///
	virtual hq_uint32 GetShader(hq_uint32 programID, HQShaderType shaderType) = 0;

	///
	///return NOT_AVAIL_ID if parameter doesn't exist
	///
	virtual hq_uint32 GetParameterIndex(hq_uint32 programID , 
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
	
#ifndef GLES
#	define HQ_RENDER_UNIFORM_BUFFER_SUPPORT 1
	///
	///Dynamic buffer can be updated by calling Map and Unmap methods
	///
	virtual HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut) = 0;
	virtual HQReturnVal DestroyUniformBuffer(hq_uint32 bufferID) = 0;
	virtual void DestroyAllUniformBuffers() = 0;
	///
	///Direct3d : {slot} = {buffer slot} bitwise OR với enum HQShaderType để chỉ  {buffer slot} thuộc shader stage nào. 
	///			Ví dụ muốn gắn uniform buffer vào buffer slot 3 của vertex shader , ta truyền tham số {slot} = (3 | HQ_VERTEX_SHADER).
	///
	virtual HQReturnVal SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID ) = 0;
	///direct3d 10/11 : chỉ có thể dùng với dynamic buffer
	virtual HQReturnVal MapUniformBuffer(hq_uint32 bufferID , void **ppData) = 0;
	///direct3d 10/11 : chỉ có thể dùng với dynamic buffer
	virtual HQReturnVal UnmapUniformBuffer(hq_uint32 bufferID) = 0;

	///
	///Copy {pData} vào uniform buffer. Toàn bộ buffer sẽ được update. Lưu ý không nên update trên dynamic buffer, nên dùng map và unmap trên dynamic buffer
	///
	virtual HQReturnVal UpdateUniformBuffer(hq_uint32 bufferID, const void * pData)= 0;
#endif
};

#endif
