/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_COMMON_H
#define HQ_SHADER_GL_COMMON_H

#include "../HQRenderDevice.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "HQShaderGL_GLSL_VarParser.h"

#include "../HQLinkedList.h"
#include <string>

#include "glHeaders.h"


#ifdef GL_PROGRAM_PIPELINE_BINDING
extern GLuint ge_shader_pipeline;
#	define HQ_GLSL_SHADER_PIPELINE_DEFINED
#	define HQ_GLSL_SHADER_PIPELINE_ID ge_shader_pipeline
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
	GLint location;//for shader created from glsl
    GLint texUnit;//for sampler type parameter
};

/*----------HQShaderObjectGL----------------------*/
struct HQShaderObjectGL: public HQShaderObject, public HQBaseIDObject
{
	HQShaderObjectGL();
	~HQShaderObjectGL();

	virtual HQShaderType GetType() const { return type; }

	GLuint shader;//for shader created from glsl
	HQShaderType type;
	HQLinkedList<HQUniformBlockInfoGL>* pUniformBlocks;//for extended version of GLSL
	HQLinkedList<HQShaderAttrib> * pAttribList;//for HQEngine extended GLSL
	HQLinkedList<HQUniformSamplerGL> * pUniformSamplerList;//for HQEngine extended GLSL

	bool isGLSL;//compiled from openGL shading laguage or Cg language
	bool selfContainProgram;
};

/*---------HQBaseShaderProgramGL----------------------*/
struct HQBaseShaderProgramGL : public HQShaderProgram, public HQBaseIDObject
{
	HQBaseShaderProgramGL();
	virtual ~HQBaseShaderProgramGL();

	virtual HQShaderObject * GetShader(HQShaderType type);//implement HQShaderProgram

	HQShaderParameterGL* TryCreateParameterObject(const char *parameterName);//just create paremeter object (if it exists in shader), doesn't add to paremeter list
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
	GLuint programGLHandle;//for program created from glsl


	HQClosedStringPrimeHashTable<hquint32> parameterIndexes;
	HQItemManager<HQShaderParameterGL> parameters;

	HQShaderObjectGL* vertexShader;
	HQShaderObjectGL* geometryShader;
	HQShaderObjectGL* pixelShader;
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

	virtual void OnDraw() {}//this is called before drawing
	virtual void OnLost() {}
	virtual void OnReset() {}

	///
	///tạo shader từ mã đã compile
	///
	HQReturnVal CreateShaderFromByteCodeStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 HQShaderObject** pID)
	{
		return HQ_FAILED;
	}

	///
	///tạo shader từ mã đã compile
	///
	HQReturnVal CreateShaderFromByteCode(HQShaderType type,
									 const hqubyte8* byteCodeData,
									 hq_uint32 byteCodeLength,
									 HQShaderObject** pID)
	{
		return HQ_FAILED;
	}
protected:
};

/*-------------HQBaseCommonShaderManagerGL----------------*/
class HQBaseCommonShaderManagerGL:
	public HQBaseShaderManagerGL,
	public HQIDItemManager<HQBaseShaderProgramGL> ,
	public HQLoggableObject
{
public:

	HQBaseCommonShaderManagerGL(HQLogStream* logFileStream , const char * logPrefix , bool flushLog);
	~HQBaseCommonShaderManagerGL();

	bool IsUsingVShader(); //có đang dùng vertex shader không,hay đang dùng fixed function
	bool IsUsingGShader();//có đang dùng geometry shader không,hay đang dùng fixed function
	bool IsUsingPShader();//có đang dùng pixel/fragment shader không,hay đang dùng fixed function
	bool IsUsingShader() {return this->activeProgram != NULL;}

	HQFileManager* GetIncludeFileManager() const { return this->includeFileManager; }

	HQReturnVal SetIncludeFileManager(HQFileManager* fileManager) { this->includeFileManager = fileManager; return HQ_OK; }

	HQSharedPtr<HQBaseShaderProgramGL> GetActiveProgram() { return activeProgram; }

	HQReturnVal RemoveProgram(HQShaderProgram* programID);
	void RemoveAllProgram();
	HQReturnVal RemoveShader(HQShaderObject* shaderID);
	void RemoveAllShader() ;
	void RemoveAllResource();

	hq_uint32 GetParameterIndex(HQShaderProgram* programID ,const char *parameterName);

protected:
	//these virtual functions can be inlined
	virtual HQBaseShaderProgramGL * CreateNewProgramObject() = 0;
	virtual void OnProgramCreated(HQBaseShaderProgramGL *program) = 0;
	virtual void OnProgramActivated() = 0;//a handler method to notify the parent class that a program has been activated
	/*-------------------------------------*/

	HQSharedPtr<HQBaseShaderProgramGL> activeProgram;
	HQSharedPtr<HQShaderObjectGL> activeCShader;//active compute shader
	HQIDItemManager<HQShaderObjectGL> shaderObjects;//danh sách shader object

	HQFileManager* includeFileManager;

	HQSharedPtr<HQShaderParameterGL> GetParameterInline(HQBaseShaderProgramGL* pProgramRawPtr, const char *parameterName);
};

extern HQBaseCommonShaderManagerGL* g_pShaderMan;

#endif
