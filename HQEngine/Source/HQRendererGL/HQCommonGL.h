/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_COMMON_GL_H
#define HQ_COMMON_GL_H


//copy buffer content to cpu. Note: buffer must be mappable
HQReturnVal CopyGLBufferContent(void *dest, size_t size, GLuint buffer, GLenum targetName, GLenum targetBindingPoint);

/*-------------generic buffer------------------------------*/
struct HQBufferGL : public HQGraphicsBufferRawRetrievable, public HQBaseIDObject
{
	HQBufferGL(hq_uint32 size, GLenum targetName, GLenum usage);
	virtual ~HQBufferGL();

	virtual void Init(const void *initData) {};

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle() { return (void*)bufferName; }

	//implement HQGraphicsBufferRawRetrievable
	virtual HQReturnVal TransferData(HQGraphicsBufferRawRetrievable* src, hquint32 destOffset, hquint32 srcOffset, hquint32 size)
	{
		return HQ_FAILED;
	}

	GLenum targetName;
	GLenum usage;//GL_STATIC_DRAW or GL_DYNAMIC_DRAW
	hq_uint32 size;
	GLuint bufferName;

	/*---------info for shader storage----------*/
	hquint32 totalElements;
	hquint32 elementSize;
};

#endif