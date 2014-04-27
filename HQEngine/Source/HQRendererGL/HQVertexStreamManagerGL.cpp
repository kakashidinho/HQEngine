/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQDeviceGL.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "HQCommonVertexStreamManGL_inline.h"



//vertex buffer----------------------------------------------------
HQVertexBufferGL::~HQVertexBufferGL()
{
	if (this->manager->GetCurrentBoundVBuffer() == this->bufferName)
	{
		this->manager->InvalidateCurrentBoundVBuffer();
	}
}

void HQVertexBufferGL::Init(const void *initData)
{
	glGenBuffers(1 , &this->bufferName);

	manager->BindVertexBuffer( this->bufferName);
	
	glBufferData(GL_ARRAY_BUFFER , this->size , initData, this->usage);
}

//mappable vertex buffer------------------------------------------------------------
HQReturnVal HQMappableVertexBufferGL::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	hq_uint32 i = offset + size;
	if (i > this->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = this->size;

	manager->BindVertexBuffer(this->bufferName);
	glBufferSubData(GL_ARRAY_BUFFER, offset, size, pData);
	return HQ_OK;
}
HQReturnVal HQMappableVertexBufferGL::Unmap()
{
	manager->BindVertexBuffer(this->bufferName);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	return HQ_OK;
}
HQReturnVal HQMappableVertexBufferGL::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
	manager->BindVertexBuffer(this->bufferName);
	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ARRAY_BUFFER, this->size, NULL, this->usage);
	}

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = (hqubyte8*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY) + offset;

	return HQ_OK;
}

HQReturnVal HQMappableVertexBufferGL::CopyContent(void *dest)
{
	return CopyGLBufferContent(dest, this->size, this->bufferName, GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING);
}


//unmappable vertexbuffer---------------------------------------------------
HQUnmappableVertexBufferGL::HQUnmappableVertexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage)
: HQVertexBufferGL(manager, size, usage), HQSysMemBuffer()
{
	if (this->usage == GL_DYNAMIC_DRAW)
		this->AllocRawBuffer(size);
}

void HQUnmappableVertexBufferGL::Init(const void *initData)
{
	HQVertexBufferGL::Init(initData);
	if (initData != NULL && this->GetRawBuffer() != NULL)
		this->Update(0, this->HQBufferGL::size, initData);//copy to system memory
}

HQReturnVal HQUnmappableVertexBufferGL::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	if (this->usage != GL_DYNAMIC_DRAW)
	{
		this->manager->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
	HQReturnVal re = HQSysMemBuffer::Update(offset, size, pData);

	if (re == HQ_OK)
	{
		manager->BindVertexBuffer(this->bufferName);
		glBufferSubData(GL_ARRAY_BUFFER, offset, size == 0 ? this->HQBufferGL::size : size, pData);
	}

	return re;
}
HQReturnVal HQUnmappableVertexBufferGL::Unmap()
{
	if (this->usage != GL_DYNAMIC_DRAW)
	{
		this->manager->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}

	HQReturnVal re = HQSysMemBuffer::Unmap();

	if (re == HQ_OK)
	{
		//now copy from sytem memory to GL buffer
		manager->BindVertexBuffer(this->bufferName);
		glBufferSubData(GL_ARRAY_BUFFER, 0, this->HQBufferGL::size, this->GetRawBuffer());
	}

	return re;
}
HQReturnVal HQUnmappableVertexBufferGL::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
	if (this->usage != GL_DYNAMIC_DRAW)
	{
		this->manager->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}

	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ARRAY_BUFFER, this->HQBufferGL::size, NULL, this->usage);
	}

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	
	return HQSysMemBuffer::Map(ppData, mapType, offset, size);
}

//index buffer---------------------------------------------------------------
void HQIndexBufferGL::Init(const void *initData)
{
	GLuint currentIBO;
	glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, (GLint*)&currentIBO);

	glGenBuffers(1 , &this->bufferName);


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER , this->bufferName);
	
	glBufferData(GL_ELEMENT_ARRAY_BUFFER , this->size , initData, this->usage);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, currentIBO);
}


//mappable index buffer------------------------------------------------------
HQReturnVal HQMappableIndexBufferGL::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	hq_uint32 i = offset + size;
	if (i > this->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = this->size;

	HQIndexBufferGL *activeIndexBuffer = manager->GetActiveIndexBuffer().GetRawPointer();

	//avoid affecting active index buffer
	if (this != activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->bufferName);

	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, offset, size, pData);

	//avoid affecting active index buffer
	if (this != activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeIndexBuffer != NULL ? activeIndexBuffer->bufferName : 0);

	return HQ_OK;
}
HQReturnVal HQMappableIndexBufferGL::Unmap()
{
	HQIndexBufferGL *activeIndexBuffer = manager->GetActiveIndexBuffer().GetRawPointer();
	//avoid affecting active index buffer
	if (this != activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->bufferName);

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

	//avoid affecting active index buffer
	if (this != activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeIndexBuffer != NULL ? activeIndexBuffer->bufferName : 0);
	return HQ_OK;
}
HQReturnVal HQMappableIndexBufferGL::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
	HQIndexBufferGL *activeIndexBuffer = manager->GetActiveIndexBuffer().GetRawPointer();
	//avoid affecting active index buffer
	if (this != activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->bufferName);

	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->size, NULL, this->usage);
	}
#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = (hqubyte8*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY) + offset;

	//avoid affecting active index buffer
	if (this != activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeIndexBuffer != NULL ? activeIndexBuffer->bufferName : 0);

	return HQ_OK;
}

HQReturnVal HQMappableIndexBufferGL::CopyContent(void *dest)
{
	return CopyGLBufferContent(dest, this->size, this->bufferName, GL_ELEMENT_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER_BINDING);
}

//unmappable index buffer---------------------------------------------------
HQUnmappableIndexBufferGL::HQUnmappableIndexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage, HQIndexDataType dataType)
: HQIndexBufferGL(manager, size, usage, dataType), HQSysMemBuffer()
{
	if (this->usage == GL_DYNAMIC_DRAW)
		this->AllocRawBuffer(size);
}

void HQUnmappableIndexBufferGL::Init(const void *initData)
{
	HQIndexBufferGL::Init(initData);
	if (initData != NULL && this->GetRawBuffer() != NULL)
		this->Update(0, this->HQBufferGL::size, initData);//copy to system memory
}

HQReturnVal HQUnmappableIndexBufferGL::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	if (this->usage != GL_DYNAMIC_DRAW)
	{
		this->manager->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
	HQReturnVal re = HQSysMemBuffer::Update(offset, size, pData);

	if (re == HQ_OK)
	{
		HQIndexBufferGL *activeIndexBuffer = manager->GetActiveIndexBuffer().GetRawPointer();
		//avoid affecting active index buffer
		if (this != activeIndexBuffer)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->bufferName);

		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, offset, size == 0 ? this->HQBufferGL::size : size, pData);

		//avoid affecting active index buffer
		if (this != activeIndexBuffer)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeIndexBuffer != NULL ? activeIndexBuffer->bufferName : 0);
	}

	return re;
}
HQReturnVal HQUnmappableIndexBufferGL::Unmap()
{
	if (this->usage != GL_DYNAMIC_DRAW)
	{
		this->manager->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}

	HQReturnVal re = HQSysMemBuffer::Unmap();

	if (re == HQ_OK)
	{
		//now copy from sytem memory to GL buffer
		HQIndexBufferGL *activeIndexBuffer = manager->GetActiveIndexBuffer().GetRawPointer();
		//avoid affecting active index buffer
		if (this != activeIndexBuffer)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->bufferName);

		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, this->HQBufferGL::size, this->GetRawBuffer());

		//avoid affecting active index buffer
		if (this != activeIndexBuffer)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeIndexBuffer != NULL ? activeIndexBuffer->bufferName : 0);
	}

	return re;
}
HQReturnVal HQUnmappableIndexBufferGL::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
	if (this->usage != GL_DYNAMIC_DRAW)
	{
		this->manager->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}

	if (mapType == HQ_MAP_DISCARD)
	{
		HQIndexBufferGL *activeIndexBuffer = manager->GetActiveIndexBuffer().GetRawPointer();
		//avoid affecting active index buffer
		if (this != activeIndexBuffer)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->bufferName);

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->HQBufferGL::size, NULL, this->usage);

		//avoid affecting active index buffer
		if (this != activeIndexBuffer)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeIndexBuffer != NULL ? activeIndexBuffer->bufferName : 0);
	}

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif

	return HQSysMemBuffer::Map(ppData, mapType, offset, size);
}

/*---------vertex format---------------*/
HQVertexInputLayoutGL::HQVertexInputLayoutGL()
{
	attribs = NULL;
	numAttribs = 0;
	flags = 0;
}
HQVertexInputLayoutGL::~HQVertexInputLayoutGL()
{
	SafeDeleteArray(attribs);
}
/*---------vertex stream manager-------*/

/*--------for HQVertexStreamManDelegateGL---------*/

inline void HQVertexStreamManagerGL::EnableVertexAttribArray(GLuint index)
{
	glEnableVertexAttribArray(index);
}
inline void HQVertexStreamManagerGL::DisableVertexAttribArray(GLuint index)
{
	glDisableVertexAttribArray(index);
}
inline void HQVertexStreamManagerGL::SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer)
{
	glVertexAttribPointer(vAttribInfo.attribIndex , 
						  vAttribInfo.size,
						  vAttribInfo.dataType,
						  vAttribInfo.normalized,
						  stride,
						  vAttribInfo.offset);
}
	/*------*/

HQVertexStreamManagerGL::HQVertexStreamManagerGL(const char *logPrefix, hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog)
:HQLoggableObject(logFileStream , logPrefix ,flushLog)
{
	this->currentBoundVBuffer = 0;
	this->maxVertexAttribs = maxVertexAttribs;
	this->streams = new HQVertexStreamGL[maxVertexAttribs];
	this->vAttribInfoNodeCache = new HQVertexAttribInfoNodeGL[maxVertexAttribs];
	for (hq_uint32 i = 0; i < maxVertexAttribs ; ++i)
	{
		this->vAttribInfoNodeCache[i].pAttribInfo = NULL;
		this->vAttribInfoNodeCache[i].pNext = NULL;
	}

	this->indexDataType = GL_UNSIGNED_SHORT;
	this->indexShiftFactor = sizeof(unsigned short) >> 1;//1
	this->indexStartAddress = NULL;

#if defined HQ_OPENGLES
	if (!GLEW_OES_mapbuffer)
		Log("Init done! No buffer mapping supported.");
	else
#endif
		Log("Init done!");
}

HQVertexStreamManagerGL::HQVertexStreamManagerGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream, bool flushLog)
:HQLoggableObject(logFileStream , "GL Vertex Stream Manager :" ,flushLog)
{
	this->currentBoundVBuffer = 0;
	this->maxVertexAttribs = maxVertexAttribs;
	this->streams = new HQVertexStreamGL[maxVertexAttribs];
	this->vAttribInfoNodeCache = new HQVertexAttribInfoNodeGL[maxVertexAttribs];
	for (hq_uint32 i = 0; i < maxVertexAttribs ; ++i)
	{
		this->vAttribInfoNodeCache[i].pAttribInfo = NULL;
		this->vAttribInfoNodeCache[i].pNext = NULL;
	}

	this->indexDataType = GL_UNSIGNED_SHORT;
	this->indexShiftFactor = sizeof(unsigned short) >> 1;//1
	this->indexStartAddress = NULL;

#if defined HQ_OPENGLES
	if (!GLEW_OES_mapbuffer)
		Log("Init done! No buffer mapping supported.");
	else
#endif
		Log("Init done!");
}

HQVertexStreamManagerGL::~HQVertexStreamManagerGL()
{
	SafeDeleteArray(this->streams);
	SafeDeleteArray(this->vAttribInfoNodeCache);
	Log("Released!");
}

HQVertexBufferGL * HQVertexStreamManagerGL::CreateNewVertexBufferObj(hq_uint32 size, GLenum usage)
{
	bool mappingSupported = true;
#if defined HQ_OPENGLES
	if (!GLEW_OES_mapbuffer)
		mappingSupported = false;
#endif

	if (mappingSupported)
		return HQ_NEW HQMappableVertexBufferGL(this, size, usage);
	else
		return HQ_NEW HQUnmappableVertexBufferGL(this, size, usage);
}

HQIndexBufferGL * HQVertexStreamManagerGL::CreateNewIndexBufferObj(hq_uint32 size, GLenum usage, HQIndexDataType dataType)
{
	bool mappingSupported = true;
#if defined HQ_OPENGLES
	if (!GLEW_OES_mapbuffer)
		mappingSupported = false;
#endif

	if (mappingSupported)
		return HQ_NEW HQMappableIndexBufferGL(this, size, usage, dataType);
	else
		return HQ_NEW HQUnmappableIndexBufferGL(this, size, usage, dataType);
}

HQReturnVal HQVertexStreamManagerGL::CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,HQVertexBuffer **pID)
{
	HQVertexBufferGL* newVBuffer;
	try{
		newVBuffer = this->CreateNewVertexBufferObj(size, _GL_DRAW_BUFFER_USAGE(dynamic));

	}
	catch (std::bad_alloc e)
	{
		return HQ_FAILED_MEM_ALLOC;
	}

	newVBuffer->Init(initData);
	
	if (glGetError() == GL_OUT_OF_MEMORY || !this->vertexBuffers.AddItem(newVBuffer , pID))
	{
		delete newVBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerGL::CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , HQIndexBuffer **pID)
{
	if (!g_pOGLDev->IsIndexDataTypeSupported(dataType))
	{
		if (dataType == HQ_IDT_UINT)
			this->Log("Error : index data type HQ_IDT_UINT is not supported!");
		return HQ_FAILED;
	}
	HQIndexBufferGL* newIBuffer;
	try{
		newIBuffer = this->CreateNewIndexBufferObj(size , _GL_DRAW_BUFFER_USAGE( dynamic ) , dataType);
	}
	catch (std::bad_alloc e)
	{
		return HQ_FAILED_MEM_ALLOC;
	}
	
	newIBuffer->Init( initData);

	if (glGetError() == GL_OUT_OF_MEMORY || !this->indexBuffers.AddItem(newIBuffer , pID))
	{
		delete newIBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerGL::CreateVertexBufferUAV(const void *initData,
	hq_uint32 elementSize,
	hq_uint32 numElements,
	HQVertexBufferUAV **ppVertexBufferOut)
{
	if (!GLEW_VERSION_4_3)
	{
		this->Log("Error : UAV buffer is not supported!");
		return HQ_FAILED;
	}

	HQVertexBuffer* pVNewBuf;

	hquint32 size = elementSize * numElements;

	HQReturnVal re = this->CreateVertexBuffer(initData, size, false, false, &pVNewBuf);

	if (re == HQ_OK)
	{
		HQBufferGL * pNewBuf = static_cast<HQBufferGL*>(pVNewBuf);
		pNewBuf->elementSize = elementSize;
		pNewBuf->totalElements = numElements;

		if (ppVertexBufferOut != NULL)
			*ppVertexBufferOut = pNewBuf;
	}
	return re;
}

HQReturnVal HQVertexStreamManagerGL::CreateIndexBufferUAV(const void *initData,
	hq_uint32 numElements,
	HQIndexDataType indexDataType,
	HQVertexBufferUAV **ppIndexBufferOut)
{
	if (!GLEW_VERSION_4_3)
	{
		this->Log("Error : UAV buffer is not supported!");
		return HQ_FAILED;
	}

	HQIndexBuffer* pINewBuf;
	hquint32 elementSize = 0;
	switch (indexDataType)
	{
	case HQ_IDT_UINT:
		elementSize = 4;
		break;
	case HQ_IDT_USHORT:
		elementSize = 2;
		break;
	}
	hquint32 size = elementSize * numElements;

	HQReturnVal re = this->CreateIndexBuffer(initData, size, false, indexDataType, &pINewBuf);

	if (re == HQ_OK)
	{
		HQBufferGL * pNewBuf = static_cast<HQBufferGL*>( pINewBuf);
		pNewBuf->elementSize = elementSize;
		pNewBuf->totalElements = numElements;

		if (ppIndexBufferOut != NULL)
			*ppIndexBufferOut = pNewBuf;
	}
	return re;
}

HQReturnVal HQVertexStreamManagerGL::CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												HQShaderObject* vertexShaderID , 
												HQVertexLayout **pID)
{
	if (vAttribDesc == NULL)
		return HQ_FAILED;
	if (numAttrib >= this->maxVertexAttribs)
		return HQ_FAILED_TOO_MANY_ATTRIBUTES;
	//check if any vertex attribute' desc is invalid
	for (hq_uint32 i = 0 ; i < numAttrib ; ++i)
	{
		if (!this->IsVertexAttribValid(vAttribDesc[i]))
			return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	HQVertexInputLayoutGL *vLayout = NULL;
	try{
		vLayout = HQ_NEW HQVertexInputLayoutGL();
		vLayout->numAttribs = numAttrib;

		vLayout->attribs = HQ_NEW HQVertexAttribInfoGL[numAttrib];
	}
	catch (std::bad_alloc e)
	{
		delete vLayout;
		return HQ_FAILED_MEM_ALLOC;
	}
	

	for (hq_uint32 i = 0 ; i < numAttrib ; ++i)
	{
		this->ConvertToVertexAttribInfo(vAttribDesc[i] , vLayout->attribs[i]);

		vLayout->flags |= (0x1 << vLayout->attribs[i].attribIndex);
		
	}

	/*-----selection sort theo thứ tự stream index---------*/
	HQVertexAttribInfoGL temp;
	hq_uint32 minIndex;

	for (hq_uint32 i = 0 ; i < numAttrib ; ++i)
	{
		minIndex = i;
		for (hq_uint32 j = i; j < numAttrib ; ++j)
		{
			if (vLayout->attribs[j].streamIndex < vLayout->attribs[minIndex].streamIndex)
			{
				minIndex = j;
			}
		}

		if (minIndex != i)//swap
		{
			memcpy(&temp , &vLayout->attribs[i] , sizeof (HQVertexAttribInfoGL));
			memcpy(&vLayout->attribs[i] , &vLayout->attribs[minIndex] , sizeof (HQVertexAttribInfoGL));
			memcpy(&vLayout->attribs[minIndex] , &temp , sizeof (HQVertexAttribInfoGL));
		}
	}
	
	/*------------------------*/

	if (!this->inputLayouts.AddItem(vLayout , pID))
	{
		delete vLayout;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

void HQVertexStreamManagerGL::ConvertToVertexAttribInfo(const HQVertexAttribDesc &vAttribDesc, HQVertexAttribInfoGL &vAttribInfo)
{
	vAttribInfo.streamIndex = vAttribDesc.stream;
	vAttribInfo.attribIndex = vAttribDesc.usage;
	vAttribInfo.offset = (void*)vAttribDesc.offset;
	switch (vAttribDesc.dataType)
	{
	case HQ_VADT_FLOAT :
		vAttribInfo.dataType = GL_FLOAT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 1;
		break;
	case HQ_VADT_FLOAT2 :
		vAttribInfo.dataType = GL_FLOAT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 2;
		break;
	case HQ_VADT_FLOAT3 :
		vAttribInfo.dataType = GL_FLOAT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 3;
		break;
	case HQ_VADT_FLOAT4 :
		vAttribInfo.dataType = GL_FLOAT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_UBYTE4 :
		vAttribInfo.dataType = GL_UNSIGNED_BYTE;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_SHORT :
		vAttribInfo.dataType = GL_SHORT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 1;
		break;
	case HQ_VADT_SHORT2 :
		vAttribInfo.dataType = GL_SHORT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 2;
		break;
	case HQ_VADT_SHORT4  :
		vAttribInfo.dataType = GL_SHORT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_USHORT :
		vAttribInfo.dataType = GL_UNSIGNED_SHORT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 1;
		break;
	case HQ_VADT_USHORT2 :
		vAttribInfo.dataType = GL_UNSIGNED_SHORT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 2;
		break;
	case HQ_VADT_USHORT4  :
		vAttribInfo.dataType = GL_UNSIGNED_SHORT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_USHORT2N :
		vAttribInfo.dataType = GL_UNSIGNED_SHORT;
		vAttribInfo.normalized = GL_TRUE;
		vAttribInfo.size = 2;
		break;
	case HQ_VADT_USHORT4N :
		vAttribInfo.dataType = GL_UNSIGNED_SHORT;
		vAttribInfo.normalized = GL_TRUE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_INT :
		vAttribInfo.dataType = GL_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 1;
		break;
	case HQ_VADT_INT2 :
		vAttribInfo.dataType = GL_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 2;
		break;
	case HQ_VADT_INT3 :
		vAttribInfo.dataType = GL_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 3;
		break;
	case HQ_VADT_INT4  :
		vAttribInfo.dataType = GL_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_UINT :
		vAttribInfo.dataType = GL_UNSIGNED_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 1;
		break;
	case HQ_VADT_UINT2 :
		vAttribInfo.dataType = GL_UNSIGNED_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 2;
		break;
	case HQ_VADT_UINT3 :
		vAttribInfo.dataType = GL_UNSIGNED_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 3;
		break;
	case HQ_VADT_UINT4  :
		vAttribInfo.dataType = GL_UNSIGNED_INT;
		vAttribInfo.normalized = GL_FALSE;
		vAttribInfo.size = 4;
		break;
	case HQ_VADT_UBYTE4N:
		vAttribInfo.dataType = GL_UNSIGNED_BYTE;
		vAttribInfo.normalized = GL_TRUE;
		vAttribInfo.size = 4;
		break;
	}
}

HQReturnVal HQVertexStreamManagerGL::SetVertexBuffer(HQVertexBuffer* vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride)
{
	if (streamIndex >= this->maxVertexAttribs)
		return HQ_FAILED;
	HQSharedPtr<HQBufferGL> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID).UpCast<HQBufferGL>();

	return HQVertexStreamManDelegateGL<HQVertexStreamManagerGL>::SetVertexBuffer(this , vBuffer , this->streams[streamIndex], stride);
}

HQReturnVal  HQVertexStreamManagerGL::SetIndexBuffer(HQIndexBuffer* indexBufferID )
{
	HQSharedPtr<HQIndexBufferGL> iBuffer = this->indexBuffers.GetItemPointer(indexBufferID);
	if (this->activeIndexBuffer != iBuffer)
	{
		if (iBuffer == NULL)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER , 0);
		else
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iBuffer->bufferName);

			this->indexDataType = iBuffer->dataType;
#if 0
			switch (this->indexDataType)
			{
			case GL_UNSIGNED_INT://0x1405
				this->indexShiftFactor = sizeof(unsigned int) >> 1;//2
				break;
			case GL_UNSIGNED_SHORT://0x1403
				this->indexShiftFactor = sizeof(unsigned short) >> 1;//1
			}
#else//optimized version
			this->indexShiftFactor = (this->indexDataType & 0x0000000f) >> 1;//ex : (GL_UNSIGNED_INT & 0xf) >> 1 = (0x1405 & 0xf) >> 1 = 2  
#endif	
		}

		this->activeIndexBuffer = iBuffer;
	}

	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerGL::SetVertexInputLayout(HQVertexLayout* inputLayoutID) 
{
	//set vertex layout 
	HQSharedPtr<HQVertexInputLayoutGL> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	HQReturnVal hr = HQVertexStreamManDelegateGL<HQVertexStreamManagerGL>::SetVertexInputLayout(
							this,
							pVLayout,
							this->activeInputLayout,
							this->streams,
							this->maxVertexAttribs,
							this->vAttribInfoNodeCache);


	return hr;

}



HQReturnVal HQVertexStreamManagerGL::RemoveVertexBuffer(HQVertexBuffer* ID) 
{
	return (HQReturnVal)this->vertexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerGL::RemoveIndexBuffer(HQIndexBuffer* ID) 
{
	return (HQReturnVal)this->indexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerGL::RemoveVertexInputLayout(HQVertexLayout* ID) 
{
	return (HQReturnVal)this->inputLayouts.Remove(ID);
}
void HQVertexStreamManagerGL::RemoveAllVertexBuffer() 
{
	this->vertexBuffers.RemoveAll();
}
void HQVertexStreamManagerGL::RemoveAllIndexBuffer() 
{
	this->indexBuffers.RemoveAll();
}
void HQVertexStreamManagerGL::RemoveAllVertexInputLayout()
{
	this->inputLayouts.RemoveAll();
}

#ifdef DEVICE_LOST_POSSIBLE
void HQVertexStreamManagerGL::OnLost()
{
	//invalidate current state
	this->currentBoundVBuffer = 0;
}

void HQVertexStreamManagerGL::OnReset()
{
	//recreate buffers
	HQItemManager<HQVertexBufferGL>::Iterator itev;
	HQItemManager<HQIndexBufferGL>::Iterator itei;
	HQItemManager<HQVertexInputLayoutGL>::Iterator itel;

	this->vertexBuffers.GetIterator(itev);

	while (!itev.IsAtEnd())
	{
		itev->Init(NULL);
		++itev;
	}


	this->indexBuffers.GetIterator(itei);

	while (!itei.IsAtEnd())
	{
		itei->Init(NULL);
		++itei;
	}

	/*--------------reset vertex stream, index buffer & input layout------------*/
	
	//reset index buffer
	if (this->activeIndexBuffer != NULL)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER , this->activeIndexBuffer->bufferName);

	//reset input layout
	HQSharedPtr<HQVertexInputLayoutGL> currentInputLayout = this->activeInputLayout;
	this->activeInputLayout = HQSharedPtr<HQVertexInputLayoutGL> :: null;
	this->inputLayouts.GetIterator(itel);
	while (!itel.IsAtEnd())
	{
		if (itel.GetItemPointer() == currentInputLayout)
		{
			this->SetVertexInputLayout(itel.GetItemPointer().GetRawPointer());
			break;
		}
		++itel;
	}


	//reset vertex stream
	for (hquint32 i = 0; i < this->maxVertexAttribs; ++i)
	{
		if (this->streams[i].vertexBuffer != NULL)
		{
			HQSharedPtr<HQBufferGL> currentBuffer = this->streams[i].vertexBuffer;
			this->streams[i].vertexBuffer = HQSharedPtr<HQBufferGL>::null;
			this->vertexBuffers.GetIterator(itev);
			while (!itev.IsAtEnd())
			{
				if (itev.GetItemPointer().GetRawPointer() == currentBuffer.GetRawPointer())
				{
					this->SetVertexBuffer(itev.GetItemPointer().GetRawPointer(), i, this->streams[i].stride);
					break;
				}
				++itev;
			}
		}
	}
}
#endif
