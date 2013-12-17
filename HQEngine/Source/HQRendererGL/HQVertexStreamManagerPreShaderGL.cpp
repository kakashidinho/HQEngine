/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQCommonVertexStreamManGL_inline.h"
#include "HQVertexStreamManagerPreShaderGL.h"
#include "HQDeviceGL.h"

GLenum g_clientStateType[4] = 
{
	GL_VERTEX_ARRAY,
	GL_COLOR_ARRAY,
	GL_NORMAL_ARRAY,
	GL_TEXTURE_COORD_ARRAY
};

inline void Common_SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride , void* start)
{
	switch (vAttribInfo.attribIndex)
	{
	case 0://position
		glVertexPointer(vAttribInfo.size,vAttribInfo.dataType , stride ,(hqubyte8*)start + (hquint32)vAttribInfo.offset);
		break;
	case 1://color
		glColorPointer(vAttribInfo.size,vAttribInfo.dataType , stride ,(hqubyte*)start + (hquint32)vAttribInfo.offset);
		break;
	case 2://normal
		glNormalPointer(vAttribInfo.dataType , stride , (hqubyte*)start + (hquint32)vAttribInfo.offset);
		break;
	case 3://texcoord
		glTexCoordPointer(vAttribInfo.size,vAttribInfo.dataType , stride , (hqubyte*)start + (hquint32)vAttribInfo.offset);
		break;
	}
}

/*-----------------------HQVertexStreamManagerNoShaderGL--------------------------------------*/
/*--------for HQVertexStreamManDelegateGL---------*/
inline void HQVertexStreamManagerNoShaderGL::EnableVertexAttribArray(GLuint index)
{
	glEnableClientState(g_clientStateType[index]);
}
inline void HQVertexStreamManagerNoShaderGL::DisableVertexAttribArray(GLuint index)
{
	glDisableClientState(g_clientStateType[index]);
}
inline void HQVertexStreamManagerNoShaderGL::SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer)
{
	Common_SetVertexAttribPointer(vAttribInfo , stride , NULL);
}

/*-------override base class-----------*/
bool HQVertexStreamManagerNoShaderGL::IsVertexAttribValid(const HQVertexAttribDesc &vAttribDesc)
{
	switch(vAttribDesc.usage)
	{
	case HQ_VAU_POSITION:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT2 :
		case HQ_VADT_FLOAT3 :
		case HQ_VADT_FLOAT4 :
		case HQ_VADT_SHORT2 :
		case HQ_VADT_SHORT4 :
			return true;
		default:
			this->Log("Error : fixed function vertex position only accepts FLOAT2 , FLOAT3 , FLOAT4 , SHORT2 , SHORT4 data types!");
			return false;
		}
	case HQ_VAU_COLOR:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT4 :
		case HQ_VADT_UBYTE4N:
			return true;
		default:
			this->Log("Error : fixed function vertex color only accepts FLOAT4 , UBYTE4N data types!");
			return false;
		}
	case HQ_VAU_NORMAL:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT3 :
			return true;
		default:
			this->Log("Error : fixed function vertex normal only accepts FLOAT3 data type!");
			return false;
		}
	case HQ_VAU_TEXCOORD0:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT2 :
		case HQ_VADT_FLOAT3 :
		case HQ_VADT_FLOAT4 :
		case HQ_VADT_SHORT2 :
		case HQ_VADT_SHORT4  :
			return true;
		default:
			this->Log("Error : fixed function vertex texcoord only accepts FLOAT2 , FLOAT3 , FLOAT4 , SHORT2 , SHORT4 data types!");
			return false;
		}
	}
	this->Log("Error : vertex attribute's usage is not supported in fixed function mode!");
	return false;
}

HQReturnVal HQVertexStreamManagerNoShaderGL::SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) 
{
	if (streamIndex >= this->maxVertexAttribs)
		return HQ_FAILED;
	HQSharedPtr<HQBufferGL> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID);

	return HQVertexStreamManDelegateGL<HQVertexStreamManagerNoShaderGL>::SetVertexBuffer(this , vBuffer , this->streams[streamIndex], stride);
}

HQReturnVal HQVertexStreamManagerNoShaderGL::SetVertexInputLayout(hq_uint32 inputLayoutID)
{
	HQSharedPtr<HQVertexInputLayoutGL> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	return HQVertexStreamManDelegateGL<HQVertexStreamManagerNoShaderGL>::SetVertexInputLayout(
							this,
							pVLayout,
							this->activeInputLayout,
							this->streams,
							this->maxVertexAttribs,
							this->vAttribInfoNodeCache);
}


/*---------------------HQVertexStreamManagerNoShaderNoMapGL------------------------------------------*/
/*--------for HQVertexStreamManDelegateGL---------*/
inline void HQVertexStreamManagerNoShaderNoMapGL::EnableVertexAttribArray(GLuint index)
{
	glEnableClientState(g_clientStateType[index]);
}
inline void HQVertexStreamManagerNoShaderNoMapGL::DisableVertexAttribArray(GLuint index)
{
	glDisableClientState(g_clientStateType[index]);
}
inline void HQVertexStreamManagerNoShaderNoMapGL::SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer)
{
	Common_SetVertexAttribPointer(vAttribInfo , stride , NULL);
}
/*-------override base class-----------*/
HQReturnVal HQVertexStreamManagerNoShaderNoMapGL::SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) 
{
	if (streamIndex >= this->maxVertexAttribs)
		return HQ_FAILED;
	HQSharedPtr<HQBufferGL> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID);

	return HQVertexStreamManDelegateGL<HQVertexStreamManagerNoShaderNoMapGL>::SetVertexBuffer(this , vBuffer , this->streams[streamIndex], stride);
}

HQReturnVal HQVertexStreamManagerNoShaderNoMapGL::SetVertexInputLayout(hq_uint32 inputLayoutID)
{
	HQSharedPtr<HQVertexInputLayoutGL> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	return HQVertexStreamManDelegateGL<HQVertexStreamManagerNoShaderNoMapGL>::SetVertexInputLayout(
							this,
							pVLayout,
							this->activeInputLayout,
							this->streams,
							this->maxVertexAttribs,
							this->vAttribInfoNodeCache);
}

/*----------------------HQVertexStreamManagerArrayGL------------------------------------------*/
/*--------for HQVertexStreamManDelegateGL---------*/
inline void HQVertexStreamManagerArrayGL::SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer)
{
	Common_SetVertexAttribPointer(vAttribInfo , stride , vbuffer->cacheData);
}

/*-------override base class-----------*/
HQReturnVal HQVertexStreamManagerArrayGL::CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,hq_uint32 *pID)
{
	HQBufferGL* newVBuffer;
	try{
		newVBuffer = new HQBufferGL(size ,_GL_DRAW_BUFFER_USAGE( dynamic ));
	
		if (initData != NULL)
			memcpy(newVBuffer->cacheData , initData, size);

	}
	catch (std::bad_alloc e)
	{
		return HQ_FAILED_MEM_ALLOC;
	}

	if (!this->vertexBuffers.AddItem(newVBuffer , pID))
	{
		delete newVBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerArrayGL::CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , hq_uint32 *pID)
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

		if (initData != NULL)
			memcpy(newIBuffer->cacheData , initData, size);

	}
	catch (std::bad_alloc e)
	{
		return HQ_FAILED_MEM_ALLOC;
	}

	if (!this->indexBuffers.AddItem(newIBuffer , pID))
	{
		delete newIBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerArrayGL::SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride)
{
	if (streamIndex >= this->maxVertexAttribs)
		return HQ_FAILED;
	HQSharedPtr<HQBufferGL> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID);

	return HQVertexStreamManDelegateGL<HQVertexStreamManagerArrayGL>::SetVertexBuffer(this , vBuffer , this->streams[streamIndex], stride);
}

HQReturnVal  HQVertexStreamManagerArrayGL::SetIndexBuffer(hq_uint32 indexBufferID )
{
	HQSharedPtr<HQIndexBufferGL> iBuffer = this->indexBuffers.GetItemPointer(indexBufferID);
	if (this->activeIndexBuffer != iBuffer)
	{
		
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
		this->indexStartAddress = iBuffer->cacheData;
		this->activeIndexBuffer = iBuffer;
	}

	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerArrayGL::SetVertexInputLayout(hq_uint32 inputLayoutID) 
{
	HQSharedPtr<HQVertexInputLayoutGL> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	return HQVertexStreamManDelegateGL<HQVertexStreamManagerArrayGL>::SetVertexInputLayout(
							this,
							pVLayout,
							this->activeInputLayout,
							this->streams,
							this->maxVertexAttribs,
							this->vAttribInfoNodeCache);

}

HQReturnVal HQVertexStreamManagerArrayGL::MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) 
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

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = vBuffer->cacheData;
	
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerArrayGL::UnmapVertexBuffer(hq_uint32 vertexBufferID) 
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerArrayGL::MapIndexBuffer(HQMapType mapType , void **ppData) 
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
#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	*ppData = this->activeIndexBuffer->cacheData;
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerArrayGL::UnmapIndexBuffer() 
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
#endif

	return HQ_OK;
}



HQReturnVal HQVertexStreamManagerArrayGL::UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQBufferGL* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
	if (vBuffer->usage != GL_DYNAMIC_DRAW)
	{
		this->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif
	
	hq_uint32 i = offset + size;
	if (i > vBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = vBuffer->size;

	memcpy((hqubyte*)vBuffer->cacheData + offset , pData , size);//update cache data
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerArrayGL::UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData)
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
	if (this->activeIndexBuffer->usage != GL_DYNAMIC_DRAW)
	{
		this->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif
	hq_uint32 i = offset + size;
	if (i > this->activeIndexBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = this->activeIndexBuffer->size;

	memcpy((hqubyte*)this->activeIndexBuffer->cacheData + offset , pData , size);//update cache data

	return HQ_OK;
}


HQReturnVal HQVertexStreamManagerArrayGL::UpdateIndexBuffer(hquint32 bufferID, hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQSharedPtr<HQIndexBufferGL> pBuffer = this->indexBuffers.GetItemPointer(bufferID);

#if defined _DEBUG || defined DEBUG
	if (pBuffer == NULL)
		return HQ_FAILED;
	if (pBuffer->usage != GL_DYNAMIC_DRAW)
	{
		this->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif
	hq_uint32 i = offset + size;
	if (i > pBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	if (i == 0)//update toàn bộ buffer
		size = pBuffer->size;

	memcpy((hqubyte*)pBuffer->cacheData + offset , pData , size);//update cache data

	return HQ_OK;
}
