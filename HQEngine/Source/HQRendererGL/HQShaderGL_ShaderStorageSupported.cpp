/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"

#include "HQDeviceGL.h"
#include "HQShaderGL_ShaderStorageSupported.h"

#ifdef HQ_GL_SHADER_STORAGE_BUFFER_DEFINED

/*----------HQShaderStorageBufferGL--------------------*/
HQShaderStorageBufferGL::HQShaderStorageBufferGL(hq_uint32 elemSize, hquint32 numElems, GLenum target)
:HQBufferGL(elemSize * numElems, target, GL_STATIC_DRAW)
{
	this->pMasterDevice = g_pOGLDev;

	this->elementSize = elemSize;
	this->totalElements = numElems;
}
HQShaderStorageBufferGL::~HQShaderStorageBufferGL()
{
}

void HQShaderStorageBufferGL::Init(const void *initData)
{
	glGenBuffers(1, &this->bufferName);

	this->BindBuffer();

	glBufferData(this->targetName, this->size, initData, this->usage);
}

hquint32 HQShaderStorageBufferGL::GetSize() const///mappable size
{
	return size;
}
HQReturnVal HQShaderStorageBufferGL::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	hq_uint32 i = offset + size;
	if (i > this->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = this->size;

	this->BindBuffer();
	glBufferSubData(this->targetName, offset, size, pData);
	return HQ_OK;
}
HQReturnVal HQShaderStorageBufferGL::Unmap()
{
	this->BindBuffer();
	glUnmapBuffer(this->targetName);
	return HQ_OK;
}
HQReturnVal HQShaderStorageBufferGL::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
	this->BindBuffer();
	GLbitfield access = GL_MAP_WRITE_BIT;
	if (mapType == HQ_MAP_DISCARD)
	{
		access |= GL_MAP_INVALIDATE_BUFFER_BIT;
	}

	if (size == 0)
		size = this->size - offset;

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = (hqubyte8*)glMapBufferRange(this->targetName, offset, size, access);

	if (*ppData == NULL)
		return HQ_FAILED;

	return HQ_OK;
}

/*--------------HQDrawIndirectBufferGL-------------------*/
HQDrawIndirectBufferGL::HQDrawIndirectBufferGL(hq_uint32 elemSize, hquint32 numElems)
: HQShaderStorageBufferGL(elemSize, numElems, GL_DRAW_INDIRECT_BUFFER)
{

}

HQDrawIndirectBufferGL::~HQDrawIndirectBufferGL(){
	if (pMasterDevice->GetBoundDrawIndirectBuffer() == this->bufferName)
		pMasterDevice->BindDrawIndirectBuffer(0);
}

void HQDrawIndirectBufferGL::BindBuffer()
{
	pMasterDevice->BindDrawIndirectBuffer(this->bufferName);
}

/*--------------HQDispatchIndirectBufferGL-------------------*/
HQDispatchIndirectBufferGL::HQDispatchIndirectBufferGL(hq_uint32 elemSize, hquint32 numElems)
: HQShaderStorageBufferGL(elemSize, numElems, GL_DISPATCH_INDIRECT_BUFFER)
{

}

HQDispatchIndirectBufferGL::~HQDispatchIndirectBufferGL(){
	if (pMasterDevice->GetBoundDispatchIndirectBuffer() == this->bufferName)
		pMasterDevice->BindDispatchIndirectBuffer(0);
}

void HQDispatchIndirectBufferGL::BindBuffer()
{
	pMasterDevice->BindDispatchIndirectBuffer(this->bufferName);
}

/*--------------HQGeneralPurposeStorageBufferGL-------------------*/
HQGeneralPurposeStorageBufferGL::HQGeneralPurposeStorageBufferGL(hq_uint32 elemSize, hquint32 numElems)
: HQShaderStorageBufferGL(elemSize, numElems, GL_SHADER_STORAGE_BUFFER)
{

}

HQGeneralPurposeStorageBufferGL::~HQGeneralPurposeStorageBufferGL(){
	HQBaseShaderManagerGL_StorageBlockSupprted* pManager =
		static_cast <HQBaseShaderManagerGL_StorageBlockSupprted*> (pMasterDevice->GetShaderManager());

	if (pManager->GetBoundShaderStorageBuffer() == this->bufferName)
		pManager->BindShaderStorageBuffer(0);
}

void HQGeneralPurposeStorageBufferGL::BindBuffer()
{
	HQBaseShaderManagerGL_StorageBlockSupprted* pManager =
		static_cast <HQBaseShaderManagerGL_StorageBlockSupprted*> (pMasterDevice->GetShaderManager());

	pManager->BindShaderStorageBuffer(this->bufferName);
}

/*----------------HQBaseShaderManagerGL_StorageBlockSupprted------------------------*/
HQBaseShaderManagerGL_StorageBlockSupprted::HQBaseShaderManagerGL_StorageBlockSupprted(HQLogStream* logFileStream, const char * logPrefix, bool flushLog)
: HQBaseShaderManagerGL_UBO(logFileStream, logPrefix, flushLog),
boundStorageBuffer(0)
{
	this->pMasterDevice = g_pOGLDev;
	this->shaderStorageBufferSlots = HQ_NEW HQSharedPtr<HQBufferGL>[this->pMasterDevice->GetDeviceCaps().nShaderStorageBlocks];
}

HQBaseShaderManagerGL_StorageBlockSupprted::~HQBaseShaderManagerGL_StorageBlockSupprted(){
	delete[] this->shaderStorageBufferSlots;
}


HQReturnVal HQBaseShaderManagerGL_StorageBlockSupprted::RemoveBufferUAV(HQBufferUAV * buffer)
{
	return (HQReturnVal)shaderStorageBuffers.Remove(buffer);
}
void HQBaseShaderManagerGL_StorageBlockSupprted::RemoveAllBufferUAVs()
{
	shaderStorageBuffers.RemoveAll();
}


HQReturnVal HQBaseShaderManagerGL_StorageBlockSupprted::CreateBufferUAV(void* initData, hquint32 elementSize, hquint32 numElements, HQBufferUAV** ppBufferOut)
{
	HQGeneralPurposeStorageBufferGL* newBuffer = HQ_NEW HQGeneralPurposeStorageBufferGL(elementSize, numElements);

	newBuffer->Init(initData);

	if (glGetError() == GL_OUT_OF_MEMORY || !this->shaderStorageBuffers.AddItem(newBuffer, ppBufferOut))
	{
		delete newBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQBaseShaderManagerGL_StorageBlockSupprted::CreateComputeIndirectArgs(void* initData, hquint32 numElements, HQComputeIndirectArgsBuffer** ppBufferOut)
{
	//element structure 
	struct Element {
		hquint32 groupX;
		hquint32 groupY;
		hquint32 groupZ;
	};

	HQDispatchIndirectBufferGL* newBuffer = HQ_NEW HQDispatchIndirectBufferGL(sizeof(Element), numElements);
	
	newBuffer->Init(initData);

	if (glGetError() == GL_OUT_OF_MEMORY || !this->shaderStorageBuffers.AddItem(newBuffer, ppBufferOut))
	{
		delete newBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQBaseShaderManagerGL_StorageBlockSupprted::CreateDrawIndirectArgs(void* initData, hquint32 numElements, HQDrawIndirectArgsBuffer** ppBufferOut)
{
	//element structure 
	struct Element {
		hquint32 number_of_vertices_per_instance;
		hquint32 number_of_instances;
		hquint32 first_vertex;
		hquint32 first_instance;
	};

	HQDrawIndirectBufferGL* newBuffer = HQ_NEW HQDrawIndirectBufferGL(sizeof(Element), numElements);

	newBuffer->Init(initData);

	if (glGetError() == GL_OUT_OF_MEMORY || !this->shaderStorageBuffers.AddItem(newBuffer, ppBufferOut))
	{
		delete newBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQBaseShaderManagerGL_StorageBlockSupprted::CreateDrawIndexedIndirectArgs(void* initData, hquint32 numElements, HQDrawIndexedIndirectArgsBuffer** ppBufferOut)
{
	//element structure 
	struct Element {
		hquint32 number_of_indices_per_instance;
		hquint32 number_of_instances;
		hquint32 first_index;
		hqint32 first_vertex;
		hquint32 first_instance;
	};

	HQDrawIndirectBufferGL* newBuffer = HQ_NEW HQDrawIndirectBufferGL(sizeof(Element), numElements);

	newBuffer->Init(initData);

	if (glGetError() == GL_OUT_OF_MEMORY || !this->shaderStorageBuffers.AddItem(newBuffer, ppBufferOut))
	{
		delete newBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQBaseShaderManagerGL_StorageBlockSupprted::SetBufferUAVForComputeShader(hquint32 uavSlot, HQBufferUAV * buffer, hquint32 firstElementIdx, hquint32 numElements)
{

#if defined _DEBUG || defined DEBUG
	if (uavSlot >= this->pMasterDevice->GetDeviceCaps().nShaderStorageBlocks)
	{
		Log("SetBufferUAVForComputeShader() Error : slot=%u is out of range!", uavSlot);
		return HQ_FAILED;
	}
#endif

	HQBufferGL* pGenericGLBuffer = static_cast<HQBufferGL*> (buffer);
	HQVertexStreamManagerGL* pVStreamMan = static_cast<HQVertexStreamManagerGL*>(this->pMasterDevice->GetVertexStreamManager());
	HQSharedPtr<HQBufferGL> pBuffer;
	if (pGenericGLBuffer != NULL)
	{
		switch (pGenericGLBuffer->targetName)
		{
		case GL_ARRAY_BUFFER://need to retrieve shared pointer from vertex stream manager
			pBuffer = pVStreamMan->GetVertexBufferSharedPtr(pGenericGLBuffer).UpCast<HQBufferGL>();
			break;
		case GL_ELEMENT_ARRAY_BUFFER://need to retrieve shared pointer from vertex stream manager
			pBuffer = pVStreamMan->GetIndexBufferSharedPtr(pGenericGLBuffer).UpCast<HQBufferGL>();
			break;
		case GL_DISPATCH_INDIRECT_BUFFER: case GL_DRAW_INDIRECT_BUFFER: case GL_SHADER_STORAGE_BUFFER://we own this buffer
			pBuffer = this->shaderStorageBuffers.GetItemPointer(pGenericGLBuffer).UpCast<HQBufferGL>();
			break;
		}
	}

	if (numElements == 0)
		numElements = pBuffer->totalElements - firstElementIdx;

	HQSharedPtr< HQBufferGL>* pCurrentBufferSlot;

	pCurrentBufferSlot = this->shaderStorageBufferSlots + uavSlot;
	if (*pCurrentBufferSlot != pBuffer)
		*pCurrentBufferSlot = pBuffer;//hold reference to buffer

	//now bind buffer
	if (pBuffer == NULL)
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, uavSlot, 0, 0, 0);
	else
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 
						uavSlot, 
						pBuffer->bufferName, 
						pBuffer->elementSize * firstElementIdx, 
						pBuffer->elementSize * numElements);



	return HQ_OK;
}

#endif//ifdef HQ_GL_SHADER_STORAGE_BUFFER_DEFINED