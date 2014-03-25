/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_FAKE_UBO_H
#define HQ_SHADER_GL_FAKE_UBO_H

#include "HQShaderGL_Common.h"


struct HQFakeUniformBufferGL
{
	HQFakeUniformBufferGL(hq_uint32 size, bool isDynamic);
	~HQFakeUniformBufferGL();

	void * pRawBuffer;
	hq_uint32 size;
	bool isDynamic;

	typedef HQLinkedList<hquint32, HQPoolMemoryManager> BufferSlotList;
	BufferSlotList boundSlots;//list of slots that this buffer bound to
};

//for opengl version that doesn't support UBO
class HQBaseShaderManagerGL_FakeUBO : public HQBaseCommonShaderManagerGL
{

public:
	HQBaseShaderManagerGL_FakeUBO(HQLogStream* logFileStream, const char * logPrefix, bool flushLog);
	~HQBaseShaderManagerGL_FakeUBO();

	HQReturnVal CreateUniformBuffer(hq_uint32 size, void *initData, bool isDynamic, hq_uint32 *pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(hq_uint32 bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot, hq_uint32 bufferID);
	HQReturnVal MapUniformBuffer(hq_uint32 bufferID, void **ppData);
	HQReturnVal UnmapUniformBuffer(hq_uint32 bufferID);
	HQReturnVal UpdateUniformBuffer(hq_uint32 bufferID, const void * pData);

protected:
	//implement HQBaseCommonShaderManagerGL
	virtual HQBaseShaderProgramGL * CreateNewProgramObject();
	virtual void OnProgramCreated(HQBaseShaderProgramGL *program);
	virtual void OnProgramActivated(HQBaseShaderProgramGL* program);
	//implement HQBaseShaderManagerGL
	virtual void Commit();//this is called before drawing

	HQItemManager<HQFakeUniformBufferGL> uniformBuffers;

	struct BufferSlotInfo;
	BufferSlotInfo* uBufferSlots;
};


#endif
