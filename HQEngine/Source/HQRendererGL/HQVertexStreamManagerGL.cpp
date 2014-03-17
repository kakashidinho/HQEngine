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

/*---------buffer----------------------*/


HQBufferGL::HQBufferGL(hq_uint32 size , GLenum usage)
{
	this->bufferName = 0;
#if defined HQ_OPENGLES
	if ((!GLEW_OES_mapbuffer && usage == GL_DYNAMIC_DRAW) || !GLEW_VERSION_1_1)
	{
		//we need cache data for mapping buffer if mapbuffer is not supported, or vbo is not supported
		this->cacheData = malloc(size);
		if (this->cacheData == NULL)
			throw std::bad_alloc();
	}
	else
		this->cacheData = NULL;
#endif
	this->usage = usage;
	this->size = size;
}
HQBufferGL::~HQBufferGL()
{
	if (this->bufferName != 0)
	{
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())
#endif
			glDeleteBuffers( 1 , &this->bufferName);
	}
#ifdef HQ_OPENGLES
	if (cacheData != NULL)
		free(cacheData);
#endif
}

//vertex buffer
struct HQVertexBufferGL : public HQBufferGL
{
	HQVertexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size , GLenum usage) 
		: HQBufferGL(size , usage) 
	{ this->manager = manager ;}
	~HQVertexBufferGL();

	void OnCreated(HQVertexStreamManagerGL *manager, const void *initData);

	HQVertexStreamManagerGL *manager;
};


HQVertexBufferGL::~HQVertexBufferGL()
{
	if (this->manager->GetCurrentBoundVBuffer() == this->bufferName)
	{
		this->manager->InvalidateCurrentBoundVBuffer();
	}
}

void HQVertexBufferGL::OnCreated(HQVertexStreamManagerGL *manager, const void *initData)
{
	glGenBuffers(1 , &this->bufferName);

	manager->BindVertexBuffer( this->bufferName);
	
	glBufferData(GL_ARRAY_BUFFER , this->size , initData, this->usage);
}

//index buffer
void HQIndexBufferGL::OnCreated(HQVertexStreamManagerGL *manager, const void *initData)
{
	glGenBuffers(1 , &this->bufferName);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER , this->bufferName);
	
	glBufferData(GL_ELEMENT_ARRAY_BUFFER , this->size , initData, this->usage);
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

	Log("Init done!");
}

HQVertexStreamManagerGL::~HQVertexStreamManagerGL()
{
	SafeDeleteArray(this->streams);
	SafeDeleteArray(this->vAttribInfoNodeCache);
	Log("Released!");
}

HQReturnVal HQVertexStreamManagerGL::CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,hq_uint32 *pID)
{
	HQBufferGL* newVBuffer;
	try{
		newVBuffer = new HQVertexBufferGL(this , size ,_GL_DRAW_BUFFER_USAGE( dynamic ));
#ifdef HQ_OPENGLES
		if (!GLEW_OES_mapbuffer &&  initData != NULL && dynamic)//map buffer is not supported
			memcpy(newVBuffer->cacheData , initData, size);
#endif

	}
	catch (std::bad_alloc e)
	{
		return HQ_FAILED_MEM_ALLOC;
	}

	newVBuffer->OnCreated(this, initData);
	
	if (glGetError() == GL_OUT_OF_MEMORY || !this->vertexBuffers.AddItem(newVBuffer , pID))
	{
		delete newVBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerGL::CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , hq_uint32 *pID)
{
	if (!g_pOGLDev->IsIndexDataTypeSupported(dataType))
	{
		if (dataType == HQ_IDT_UINT)
			this->Log("Error : index data type HQ_IDT_UINT is not supported!");
		return HQ_FAILED;
	}
	HQIndexBufferGL* newIBuffer;
	try{
		newIBuffer = new HQIndexBufferGL(size , _GL_DRAW_BUFFER_USAGE( dynamic ) , dataType);
#ifdef HQ_OPENGLES
		if (!GLEW_OES_mapbuffer &&  initData != NULL && dynamic)//map buffer is not supported
			memcpy(newIBuffer->cacheData , initData, size);
#endif
	}
	catch (std::bad_alloc e)
	{
		return HQ_FAILED_MEM_ALLOC;
	}
	
	newIBuffer->OnCreated(this, initData);

	if (glGetError() == GL_OUT_OF_MEMORY || !this->indexBuffers.AddItem(newIBuffer , pID))
	{
		delete newIBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerGL::CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												hq_uint32 vertexShaderID , 
												hq_uint32 *pID)
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
		vLayout = new HQVertexInputLayoutGL();
		vLayout->numAttribs = numAttrib;

		vLayout->attribs = new HQVertexAttribInfoGL[numAttrib];
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

void HQVertexStreamManagerGL::ConvertToVertexAttribInfo(const HQVertexAttribDesc &vAttribDesc ,HQVertexAttribInfoGL &vAttribInfo)
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

HQReturnVal HQVertexStreamManagerGL::SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride)
{
	if (streamIndex >= this->maxVertexAttribs)
		return HQ_FAILED;
	HQSharedPtr<HQBufferGL> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID);

	return HQVertexStreamManDelegateGL<HQVertexStreamManagerGL>::SetVertexBuffer(this , vBuffer , this->streams[streamIndex], stride);
}

HQReturnVal  HQVertexStreamManagerGL::SetIndexBuffer(hq_uint32 indexBufferID )
{
	HQSharedPtr<HQIndexBufferGL> iBuffer = this->indexBuffers.GetItemPointer(indexBufferID);
	if (this->activeIndexBuffer != iBuffer)
	{
		if (iBuffer == NULL)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER , 0);
		else
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER , iBuffer->bufferName);
		
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

		this->activeIndexBuffer = iBuffer;
	}

	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerGL::SetVertexInputLayout(hq_uint32 inputLayoutID) 
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

HQReturnVal HQVertexStreamManagerGL::MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) 
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif

	this->BindVertexBuffer(vBuffer->bufferName);
	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ARRAY_BUFFER , vBuffer->size , NULL, vBuffer->usage);
	}

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = glMapBuffer(GL_ARRAY_BUFFER , GL_WRITE_ONLY);
	
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerGL::UnmapVertexBuffer(hq_uint32 vertexBufferID) 
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	this->BindVertexBuffer(vBuffer->bufferName);
	glUnmapBuffer(GL_ARRAY_BUFFER );
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerGL::MapIndexBuffer(HQMapType mapType , void **ppData) 
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
#endif
	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ELEMENT_ARRAY_BUFFER , this->activeIndexBuffer->size , NULL,  this->activeIndexBuffer->usage);
	}
#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = glMapBuffer(GL_ELEMENT_ARRAY_BUFFER , GL_WRITE_ONLY);
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerGL::UnmapIndexBuffer() 
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
#endif

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER );
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerGL::UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	
	hq_uint32 i = offset + size;
	if (i > vBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = vBuffer->size;

	this->BindVertexBuffer(vBuffer->bufferName);
	glBufferSubData(GL_ARRAY_BUFFER , offset , size , pData); 
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerGL::UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData)
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
#endif
	hq_uint32 i = offset + size;
	if (i > this->activeIndexBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = this->activeIndexBuffer->size;

	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER , offset , size , pData); 

	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerGL::UpdateIndexBuffer(hquint32 bufferID, hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQSharedPtr<HQIndexBufferGL> pBuffer = this->indexBuffers.GetItemPointer(bufferID);
#if defined _DEBUG || defined DEBUG
	if (pBuffer == NULL)
		return HQ_FAILED;
#endif
	hq_uint32 i = offset + size;
	if (i > pBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = pBuffer->size;

	if (pBuffer != this->activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pBuffer->bufferName);

	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER , offset , size , pData); 
	
	if (pBuffer != this->activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->activeIndexBuffer->bufferName);

	return HQ_OK;
}


HQReturnVal HQVertexStreamManagerGL::RemoveVertexBuffer(hq_uint32 ID) 
{
	return (HQReturnVal)this->vertexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerGL::RemoveIndexBuffer(hq_uint32 ID) 
{
	return (HQReturnVal)this->indexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerGL::RemoveVertexInputLayout(hq_uint32 ID) 
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
	HQItemManager<HQBufferGL>::Iterator itev;
	HQItemManager<HQIndexBufferGL>::Iterator itei;
	HQItemManager<HQVertexInputLayoutGL>::Iterator itel;

	this->vertexBuffers.GetIterator(itev);

	while (!itev.IsAtEnd())
	{
		itev->OnCreated(this, NULL);
		++itev;
	}


	this->indexBuffers.GetIterator(itei);

	while (!itei.IsAtEnd())
	{
		itei->OnCreated(this, NULL);
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
			this->SetVertexInputLayout(itel.GetID());
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
				if (itev.GetItemPointer() == currentBuffer)
				{
					this->SetVertexBuffer(itev.GetID(), i, this->streams[i].stride);
					break;
				}
				++itev;
			}
		}
	}
}
#endif

#ifdef HQ_OPENGLES
/*--------HQVertexStreamManagerNoMapGL------------------*/

HQReturnVal HQVertexStreamManagerNoMapGL::MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) 
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
	if (vBuffer->usage != GL_DYNAMIC_DRAW)
	{
		this->Log("Error : static buffer can't be mapped!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

	this->BindVertexBuffer(vBuffer->bufferName);
	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ARRAY_BUFFER , vBuffer->size , NULL, vBuffer->usage);
	}

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = vBuffer->cacheData;
	
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerNoMapGL::UnmapVertexBuffer(hq_uint32 vertexBufferID) 
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	this->BindVertexBuffer(vBuffer->bufferName);
	glBufferData(GL_ARRAY_BUFFER , vBuffer->size , vBuffer->cacheData, vBuffer->usage);
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerNoMapGL::MapIndexBuffer(HQMapType mapType , void **ppData) 
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
	if (this->activeIndexBuffer->usage != GL_DYNAMIC_DRAW)
	{
		this->Log("Error : static buffer can't be mapped!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif
	if (mapType == HQ_MAP_DISCARD)
	{
		glBufferData(GL_ELEMENT_ARRAY_BUFFER , this->activeIndexBuffer->size , NULL,  this->activeIndexBuffer->usage);
	}
#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = this->activeIndexBuffer->cacheData;
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerNoMapGL::UnmapIndexBuffer() 
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
#endif

	glBufferData(GL_ELEMENT_ARRAY_BUFFER , this->activeIndexBuffer->size , this->activeIndexBuffer->cacheData,  this->activeIndexBuffer->usage);
	return HQ_OK;
}



HQReturnVal HQVertexStreamManagerNoMapGL::UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	
	hq_uint32 i = offset + size;
	if (i > vBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = vBuffer->size;

	this->BindVertexBuffer(vBuffer->bufferName);
	glBufferSubData(GL_ARRAY_BUFFER , offset , size , pData); 
	if (vBuffer->usage == GL_DYNAMIC_DRAW)
		memcpy((hqubyte*)vBuffer->cacheData + offset , pData , size);//update cache data
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerNoMapGL::UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData)
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
	
#endif
	hq_uint32 i = offset + size;
	if (i > this->activeIndexBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = this->activeIndexBuffer->size;

	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER , offset , size , pData); 
	
	if (this->activeIndexBuffer->usage == GL_DYNAMIC_DRAW)
		memcpy((hqubyte*)this->activeIndexBuffer->cacheData + offset , pData , size);//update cache data

	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerNoMapGL::UpdateIndexBuffer(hquint32 bufferID, hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQSharedPtr<HQIndexBufferGL> pBuffer = this->indexBuffers.GetItemPointer(bufferID);

#if defined _DEBUG || defined DEBUG
	if (pBuffer == NULL)
		return HQ_FAILED;
	
#endif
	hq_uint32 i = offset + size;
	if (i > pBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = pBuffer->size;
	
	if (pBuffer != this->activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pBuffer->bufferName);

	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER , offset , size , pData); 
	
	if (pBuffer != this->activeIndexBuffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->activeIndexBuffer->bufferName);

	if (pBuffer->usage == GL_DYNAMIC_DRAW)
		memcpy((hqubyte*)pBuffer->cacheData + offset , pData , size);//update cache data

	return HQ_OK;
}

#endif
