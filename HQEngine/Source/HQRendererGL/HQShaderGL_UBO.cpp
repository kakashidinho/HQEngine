/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"

#include "glHeaders.h"
#include "HQShaderGL_UBO.h"

#ifdef HQ_GL_UNIFORM_BUFFER_DEFINED

/*---------HQUniformBufferGL--------------*/
HQUniformBufferGL::HQUniformBufferGL(HQBaseShaderManagerGL_UBO *manager , hq_uint32 size ,GLenum usage) 
{
	this->manager = manager;
	this->buffer = 0;
	this->size = size;
	this->usage = usage;
}
HQUniformBufferGL::~HQUniformBufferGL()
{
	if (buffer)
	{
		if (this->manager->GetCurrentBoundUBuffer() == this->buffer)
			this->manager->InvalidateCurrentBoundUBuffer();
		glDeleteBuffers(1 , &buffer);
	}
}


HQReturnVal HQUniformBufferGL::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
#if defined _DEBUG || defined DEBUG
	if (pData == NULL)
		return HQ_FAILED;
	if (offset != 0 || (size != this->size && size != 0))
	{
		manager->Log("Error : uniform buffer can't be updated partially!");
		return HQ_FAILED;
	}
#endif

	manager->BindUniformBuffer(this->buffer);

	glBufferSubData(GL_UNIFORM_BUFFER, 0, this->size, pData);//copy data to entire buffer

	return HQ_OK;
}
HQReturnVal HQUniformBufferGL::Unmap()
{
	manager->BindUniformBuffer(this->buffer);

	glUnmapBuffer(GL_UNIFORM_BUFFER);

	return HQ_OK;
}

HQReturnVal HQUniformBufferGL::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
	if (offset != 0 || (size != this->size && size != 0))
	{
		manager->Log("Error : uniform buffer can't be updated partially!");
		return HQ_FAILED;
	}
#endif

	manager->BindUniformBuffer(this->buffer);

	*ppData = glMapBufferRange(GL_UNIFORM_BUFFER, 0, this->size, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

	return HQ_OK;
}

/*--------------HQBaseShaderManagerGL_UBO-----------*/
HQBaseShaderManagerGL_UBO::HQBaseShaderManagerGL_UBO(HQLogStream* logFileStream , const char * logPrefix , bool flushLog)
:HQBaseCommonShaderManagerGL(logFileStream , logPrefix , flushLog)
{
	this->currentBoundUBuffer = 0;
}

HQReturnVal HQBaseShaderManagerGL_UBO::CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , HQUniformBuffer **pBufferIDOut)
{
	HQUniformBufferGL *newBuffer = new HQUniformBufferGL(this , size , _GL_DRAW_BUFFER_USAGE(isDynamic));
	if (newBuffer == NULL)
		return HQ_FAILED_MEM_ALLOC;
	glGenBuffers(1 , &newBuffer->buffer);
	
	this->BindUniformBuffer(newBuffer->buffer);
	
	glBufferData(GL_UNIFORM_BUFFER , size , initData , newBuffer->usage);
	
	if (newBuffer->buffer == 0 || GL_OUT_OF_MEMORY == glGetError() || 
		!this->uniformBuffers.AddItem(newBuffer , pBufferIDOut))
	{
		delete newBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
HQReturnVal HQBaseShaderManagerGL_UBO::RemoveUniformBuffer(HQUniformBuffer* bufferID)
{
	return (HQReturnVal)this->uniformBuffers.Remove(bufferID);
}
void HQBaseShaderManagerGL_UBO::RemoveAllUniformBuffers()
{
	this->uniformBuffers.RemoveAll();
}
HQReturnVal HQBaseShaderManagerGL_UBO::SetUniformBuffer(hq_uint32 slot ,  HQUniformBuffer* bufferID )
{
	if (slot >= MAX_UNIFORM_BUFFER_SLOTS)
		return HQ_FAILED;
	HQSharedPtr<HQUniformBufferGL> buffer = this->uniformBuffers.GetItemPointer(bufferID);
	if (this->uBufferSlots[slot] != buffer)
	{
		if (buffer != NULL)
		{
			glBindBufferBase(GL_UNIFORM_BUFFER , slot , buffer->buffer);
			this->currentBoundUBuffer = buffer->buffer;
		}
		this->uBufferSlots[slot] = buffer;
	}

	return HQ_OK;
}

#endif//ifdef HQ_GL_UNIFORM_BUFFER_DEFINED
