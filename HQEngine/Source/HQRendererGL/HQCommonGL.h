/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_COMMON_GL_H
#define HQ_COMMON_GL_H


/*-------------generic buffer------------------------------*/
struct HQBufferGL : public HQGraphicsBufferRawRetrievable, public HQBaseIDObject
{
	HQBufferGL(hq_uint32 size, GLenum targetName, GLenum usage);
	virtual ~HQBufferGL();

	virtual void Init(const void *initData) {};

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle() { return (void*)bufferName; }

	GLenum targetName;
	GLenum usage;//GL_STATIC_DRAW or GL_DYNAMIC_DRAW
	hq_uint32 size;
	GLuint bufferName;

	/*---------info for shader storage----------*/
	hquint32 totalElements;
	hquint32 elementSize;
};
/*-------------------------------*/
inline HQBufferGL::HQBufferGL(hq_uint32 size, GLenum targetName, GLenum usage)
{
	this->targetName = targetName;
	this->bufferName = 0;
	this->usage = usage;
	this->size = size;

	/*---------info for shader storage----------*/
	this->totalElements = 0;
	this->elementSize = 0;
}

inline HQBufferGL::~HQBufferGL()
{
	if (this->bufferName != 0)
	{
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())
#endif
			glDeleteBuffers(1, &this->bufferName);
	}
}

#endif