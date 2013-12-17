/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#ifndef GLES

#include "glHeaders.h"
#include "HQShaderGL_UBO.h"

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

/*--------------HQBaseShaderManagerGL_UBO-----------*/
HQBaseShaderManagerGL_UBO::HQBaseShaderManagerGL_UBO(HQLogStream* logFileStream , const char * logPrefix , bool flushLog)
:HQBaseCommonShaderManagerGL(logFileStream , logPrefix , flushLog)
{
	this->currentBoundUBuffer = 0;
}

HQReturnVal HQBaseShaderManagerGL_UBO::CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut)
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

	return HQ_FAILED;
}
HQReturnVal HQBaseShaderManagerGL_UBO::DestroyUniformBuffer(hq_uint32 bufferID)
{
	return (HQReturnVal)this->uniformBuffers.Remove(bufferID);
}
void HQBaseShaderManagerGL_UBO::DestroyAllUniformBuffers()
{
	this->uniformBuffers.RemoveAll();
}
HQReturnVal HQBaseShaderManagerGL_UBO::SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID )
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

	return HQ_FAILED;
}
HQReturnVal HQBaseShaderManagerGL_UBO::MapUniformBuffer(hq_uint32 bufferID , void **ppData)
{
	HQUniformBufferGL* buffer = this->uniformBuffers.GetItemRawPointer(bufferID);
#if defined DEBUG || defined _DEBUG
	if (buffer == NULL)
		return HQ_FAILED_INVALID_ID;
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	this->BindUniformBuffer(buffer->buffer);
		
	glBufferData(GL_UNIFORM_BUFFER , buffer->size , NULL , buffer->usage);//discard old data
	
	*ppData = glMapBuffer( GL_UNIFORM_BUFFER , GL_WRITE_ONLY);

	return HQ_OK;
}
HQReturnVal HQBaseShaderManagerGL_UBO::UnmapUniformBuffer(hq_uint32 bufferID)
{
	HQUniformBufferGL* buffer = this->uniformBuffers.GetItemRawPointer(bufferID);
#if defined DEBUG || defined _DEBUG	
	if (buffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	this->BindUniformBuffer(buffer->buffer);
	glUnmapBuffer( GL_UNIFORM_BUFFER);
	return HQ_OK;
}

HQReturnVal HQBaseShaderManagerGL_UBO::UpdateUniformBuffer(hq_uint32 bufferID, const void * pData)
{
	HQUniformBufferGL* buffer = this->uniformBuffers.GetItemRawPointer(bufferID);
#if defined _DEBUG || defined DEBUG	
	if (buffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	//update entire buffer
	hq_uint32 size = buffer->size;

	this->BindUniformBuffer(buffer->buffer);
	glBufferSubData(GL_UNIFORM_BUFFER , 0 , size , pData); 
	return HQ_OK;
}

#endif//ifndef GLES
