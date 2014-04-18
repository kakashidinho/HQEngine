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
	HQReturnVal DestroyUniformBuffer(HQUniformBuffer* bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot, HQUniformBuffer* bufferID);

	inline void BindUniformBuffer(GLuint buffer)
	{
		if (currentBoundUBuffer != buffer)
		{
			glBindBuffer(GL_UNIFORM_BUFFER, buffer);
			currentBoundUBuffer = buffer;
		}
	}

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
