/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_UBO_H
#define HQ_SHADER_GL_UBO_H

#include "HQShaderGL_Common.h"

#ifdef GL_UNIFORM_BUFFER
#	define HQ_GL_UNIFORM_BUFFER_DEFINED
#endif

#ifdef HQ_GL_UNIFORM_BUFFER_DEFINED

#define MAX_UNIFORM_BUFFER_SLOTS 36

class HQBaseShaderManagerGL_UBO;

struct HQUniformBufferGL: public HQUniformBuffer, public HQBaseIDObject
{
	HQUniformBufferGL(HQBaseShaderManagerGL_UBO *manager , hq_uint32 size ,GLenum usage) ;
	~HQUniformBufferGL();
	
	virtual hquint32 GetSize() const { return size; }///mappable size
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);

	GLuint buffer;
	HQBaseShaderManagerGL_UBO *manager;
	hq_uint32 size;
	GLenum usage;
};

//for openGL 3.1 and later or ARB_uniform_buffer_object extension
class HQBaseShaderManagerGL_UBO : public HQBaseCommonShaderManagerGL
{

public:
	HQBaseShaderManagerGL_UBO(HQLogStream* logFileStream , const char * logPrefix , bool flushLog);
	
	GLuint GetCurrentBoundUBuffer() {return this->currentBoundUBuffer;}
	void InvalidateCurrentBoundUBuffer() {this->currentBoundUBuffer = 0;}

	HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , HQUniformBuffer **pBufferIDOut);
	HQReturnVal RemoveUniformBuffer(HQUniformBuffer* bufferID);
	void RemoveAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot, HQUniformBuffer* bufferID) __FINAL__;
	HQReturnVal SetUniformBuffer(HQShaderType stage, hq_uint32 slot, HQUniformBuffer* bufferID){
		return this->HQBaseShaderManagerGL_UBO::SetUniformBuffer(slot, bufferID);//shader stage is ignored since opengl used same set of buffer slots for all shader stages
	}


	inline void BindUniformBuffer(GLuint buffer)
	{
		if (currentBoundUBuffer != buffer)
		{
			glBindBuffer(GL_UNIFORM_BUFFER, buffer);
			currentBoundUBuffer = buffer;
		}
	}

	/*-----unsupported features---------*/
	HQReturnVal CreateBufferUAV(hquint32 numElements, hquint32 elementSize, void *initData, HQBufferUAV** ppBufferOut) { return HQ_FAILED; }

	HQReturnVal CreateComputeIndirectArgs(hquint32 numElements, void *initData, HQComputeIndirectArgsBuffer** ppBufferOut)  { return HQ_FAILED; }

	HQReturnVal CreateDrawIndirectArgs(hquint32 numElements, void *initData, HQDrawIndirectArgsBuffer** ppBufferOut)  { return HQ_FAILED; }

	HQReturnVal CreateDrawIndexedIndirectArgs(hquint32 numElements, void *initData, HQDrawIndexedIndirectArgsBuffer** ppBufferOut)  { return HQ_FAILED; }

	HQReturnVal SetBufferUAVForComputeShader(hquint32 slot, HQBufferUAV * buffer, hquint32 firstElementIdx, hquint32 numElements)  { return HQ_FAILED; }

	HQReturnVal RemoveBufferUAV(HQBufferUAV * buffer) { return HQ_FAILED; }
	void RemoveAllBufferUAVs() {}

protected:
	//implement HQBaseCommonShaderManagerGL
	virtual HQBaseShaderProgramGL * CreateNewProgramObject() { return HQ_NEW HQBaseShaderProgramGL(); }
	virtual inline void OnProgramCreated(HQBaseShaderProgramGL *program) {}//do nothing
	virtual inline void OnProgramActivated() {}//do nothing

	HQIDItemManager<HQUniformBufferGL> uniformBuffers;

	GLuint currentBoundUBuffer;
	HQSharedPtr<HQUniformBufferGL> uBufferSlots[MAX_UNIFORM_BUFFER_SLOTS];
};


#endif//ifdef HQ_GL_UNIFORM_BUFFER_DEFINED

#endif
