/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"

#include "HQDeviceGL.h"


/*--------------HQBufferGL-----------------*/
HQBufferGL::HQBufferGL(hq_uint32 size, GLenum targetName, GLenum usage)
{
	this->targetName = targetName;
	this->bufferName = 0;
	this->usage = usage;
	this->size = size;

	/*---------info for shader storage----------*/
	this->totalElements = 0;
	this->elementSize = 0;
}

HQBufferGL::~HQBufferGL()
{
	if (this->bufferName != 0)
	{
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())
#endif
			glDeleteBuffers(1, &this->bufferName);
	}
}

HQReturnVal CopyGLBufferContent(void *dest, size_t size, GLuint buffer, GLenum targetName, GLenum targetBindingPoint)
{
	//save current bound buffer
	GLint oldBoundBuffer;
	glGetIntegerv(targetBindingPoint, &oldBoundBuffer);

	//bind buffer
	glBindBuffer(targetName, buffer);

	void *mappedData = glMapBuffer(targetName, GL_READ_ONLY);

	HQReturnVal re = HQ_FAILED;
#ifdef HQ_OPENGLES
	if (mappedData != NULL)
	{
		memcpy(dest, mappedData, size);
		re = HQ_OK;
	}
#else
	glGetBufferSubData(targetName, 0, size, dest);
	re = HQ_OK;
#endif
	//restore old bound buffer
	glBindBuffer(targetName, oldBoundBuffer);


	return re;
}
