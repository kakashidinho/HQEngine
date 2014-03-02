/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_COMMON_H
#define HQ_SHADER_GL_COMMON_H

#include "../HQRenderDevice.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "../BaseImpl/HQShaderParameterIndexTable.h"
#include "HQShaderGL_GLSL_VarParser.h"

#include "../HQLinkedList.h"
#include <string>

#include "glHeaders.h"
#ifndef GLES

/*---------------------Cg API--------------------*/
//#define CG_IMPLICIT_LINK

#ifdef CG_IMPLICIT_LINK

#include "Cg/cg.h"
#include "Cg/cgGL.h"

#ifdef WIN32
#	pragma comment( lib, "cg.lib" )
#	pragma comment( lib, "cgGL.lib" )
#endif

#else

#include "cgFunctionPointers.h"



#define HQ_DECL_CG_FUNC_PTR(name) pf##name name

extern HQ_DECL_CG_FUNC_PTR( cgGetError ) ;
extern HQ_DECL_CG_FUNC_PTR( cgSetErrorCallback );
extern HQ_DECL_CG_FUNC_PTR( cgGetErrorString ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGetNamedParameter ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGetParameterType ) ;
extern HQ_DECL_CG_FUNC_PTR( cgCreateContext ) ;
extern HQ_DECL_CG_FUNC_PTR( cgDestroyContext ) ;
extern HQ_DECL_CG_FUNC_PTR( cgCreateProgramFromFile ) ;
extern HQ_DECL_CG_FUNC_PTR( cgCreateProgram ) ;
extern HQ_DECL_CG_FUNC_PTR( cgCombinePrograms2 ) ;
extern HQ_DECL_CG_FUNC_PTR( cgCombinePrograms3 ) ;
extern HQ_DECL_CG_FUNC_PTR( cgDestroyProgram ) ;
extern HQ_DECL_CG_FUNC_PTR( cgSetParameterValueir ) ;
extern HQ_DECL_CG_FUNC_PTR( cgSetParameterValuefr ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLGetLatestProfile ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLIsProfileSupported ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLSetOptimalOptions ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLSetDebugMode ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLLoadProgram ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLUnbindProgram ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLDisableProfile ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGLBindProgram ) ;
extern HQ_DECL_CG_FUNC_PTR( cgGetArrayTotalSize);
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter1iv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter2iv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter3iv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter4iv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter1fv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter2fv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter3fv );
extern HQ_DECL_CG_FUNC_PTR( cgSetParameter4fv );
extern HQ_DECL_CG_FUNC_PTR( cgGLEnableProgramProfiles );
extern HQ_DECL_CG_FUNC_PTR( cgGetString );
extern HQ_DECL_CG_FUNC_PTR( cgGLSetContextGLSLVersion );
extern HQ_DECL_CG_FUNC_PTR( cgGLGetGLSLVersion );
extern HQ_DECL_CG_FUNC_PTR( cgGLGetContextGLSLVersion );

/*-----------end CG API function pointers definition------------*/

#endif//ifdef CG_IMPLICIT_LINK

#endif


#define useV USEVSHADER
#define useG USEGSHADER
#define useF USEFSHADER
#define useVF (USEVSHADER | USEFSHADER)
#define useGF (USEGSHADER | USEFSHADER)
#define useVG (USEVSHADER | USEGSHADER)
#define useVGF (USEVSHADER | USEGSHADER | USEFSHADER)



/*--------------HQShaderParameterGL-------------------*/
struct HQShaderParameterGL
{
	union{
#ifndef GLES
		struct//for shader created from cg language
		{
			CGparameter parameter;
			CGtype type;
		};
#endif
        struct
        {
            GLint location;//for shader created from glsl
            GLint texUnit;//for sampler type parameter
        };
	};
};

/*----------HQShaderObjectGL----------------------*/
struct HQShaderObjectGL
{
	HQShaderObjectGL();
	~HQShaderObjectGL();

	union
	{
#ifndef GLES
		CGprogram program;//for shader created from cg language
#endif
		GLuint shader;//for shader created from glsl
	};
	HQShaderType type;
	HQLinkedList<HQShaderAttrib> * pAttribList;//for HQEngine extended GLSL
	HQLinkedList<HQUniformSamplerGL> * pUniformSamplerList;//for HQEngine extended GLSL

	bool isGLSL;//compiled from openGL shading laguage or Cg language
};

/*---------HQBaseShaderProgramGL----------------------*/
struct HQBaseShaderProgramGL
{
	HQBaseShaderProgramGL();
	~HQBaseShaderProgramGL();

	virtual HQShaderParameterGL* TryCreateParameterObject(const char *parameterName) = 0;//just create paremeter object (if it exists in shader), doesn't add to paremeter list
	hq_uint32 TryCreateParamObjAndAddToParamsList(const char *parameterName);//create paremeter object (if it exists in shader), return parameter index
	//get pointer to parameter object , if it doesn't exist in parameter list ,
	//this method will try to create parameter object(if it exists in shader)
	//and add to parameter list
	const HQShaderParameterGL* GetParameter(const char *parameterName);
	//get index of parameter object in parameter list,
	//if it doesn't exist in parameter list ,
	//this method will try to create parameter object(if it exists in shader)
	//and add to parameter list
	hq_uint32 GetParameterIndex(const char *parameterName);

	/*----------attributes-----------------*/
	union{
#ifndef GLES
		CGprogram program;//for program created from cg language
#endif
		GLuint programGLHandle;//for program created from glsl
	};


	HQShaderParamIndexTable parameterIndexes;
	HQItemManager<HQShaderParameterGL> parameters;

	hquint32 vertexShaderID;
	hquint32 geometryShaderID;
	hquint32 pixelShaderID;
	bool isGLSL;//created from openGL shading laguage or Cg language
};


HQ_FORCE_INLINE const HQShaderParameterGL* HQBaseShaderProgramGL::GetParameter(const char *parameterName)
{
	hq_uint32 * pIndex = this->parameterIndexes.GetItemPointer(parameterName);
	if (pIndex == NULL)//chưa có
	{
		hq_uint32 index = this->TryCreateParamObjAndAddToParamsList(parameterName);
		if (index != HQ_NOT_AVAIL_ID)
			return this->parameters.GetItemRawPointerNonCheck(index);
	}
	else
	{
		return this->parameters.GetItemRawPointerNonCheck(*pIndex);
	}

	return NULL;
}

HQ_FORCE_INLINE hq_uint32 HQBaseShaderProgramGL::GetParameterIndex(const char *parameterName)
{
	hq_uint32 * pIndex = this->parameterIndexes.GetItemPointer(parameterName);
	if (pIndex == NULL)//chưa có
	{
		return this->TryCreateParamObjAndAddToParamsList(parameterName);
	}

	return * pIndex;
}

/*-------HQBaseShaderManagerGL - base class for all type of shader manager class------------*/
class HQBaseShaderManagerGL : public HQShaderManager, public HQResetable
{
public:
	~HQBaseShaderManagerGL() {}

	virtual void NotifyFFRenderIfNeeded() {}
	virtual void OnLost() {}
	virtual void OnReset() {}

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
};

/*-------------HQBaseCommonShaderManagerGL----------------*/
class HQBaseCommonShaderManagerGL:
	public HQBaseShaderManagerGL,
	public HQItemManager<HQBaseShaderProgramGL> ,
	public HQLoggableObject
{
protected:
	hq_uint32 activeProgram;
	HQItemManager<HQShaderObjectGL> shaderObjects;//danh sách shader object

	HQSharedPtr<HQShaderParameterGL> GetParameterInline(HQBaseShaderProgramGL* pProgramRawPtr ,const char *parameterName);
public:

	HQBaseCommonShaderManagerGL(HQLogStream* logFileStream , const char * logPrefix , bool flushLog);
	~HQBaseCommonShaderManagerGL();

	bool IsUsingVShader(); //có đang dùng vertex shader không,hay đang dùng fixed function
	bool IsUsingGShader();//có đang dùng geometry shader không,hay đang dùng fixed function
	bool IsUsingPShader();//có đang dùng pixel/fragment shader không,hay đang dùng fixed function
	bool IsUsingShader() {return this->activeProgram != HQ_NOT_USE_SHADER;}

	hq_uint32 GetActiveProgram() {return activeProgram;}

	HQReturnVal DestroyProgram(hq_uint32 programID);
	void DestroyAllProgram();
	HQReturnVal DestroyShader(hq_uint32 shaderID);
	void DestroyAllShader() ;
	void DestroyAllResource();

	hq_uint32 GetShader(hq_uint32 programID, HQShaderType shaderType);

	hq_uint32 GetParameterIndex(hq_uint32 programID ,const char *parameterName);

#ifndef GLES
	HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(hq_uint32 bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID );
	HQReturnVal MapUniformBuffer(hq_uint32 bufferID , void **ppData);
	HQReturnVal UnmapUniformBuffer(hq_uint32 bufferID);
	HQReturnVal UpdateUniformBuffer(hq_uint32 bufferID, const void * pData);
#endif
};

extern HQBaseCommonShaderManagerGL* g_pShaderMan;

#endif
