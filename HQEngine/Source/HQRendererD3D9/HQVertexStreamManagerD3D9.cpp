/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQVertexStreamManagerD3D9.h"

/*---------buffer----------------------*/
//--------------HQVertexBufferD3D9----------------------
void HQVertexBufferD3D9::OnResetDevice()
{
	DWORD usage = D3DUSAGE_WRITEONLY;
	if (isDynamic)
		usage |= D3DUSAGE_DYNAMIC;
	if (isForPointSprites)
		usage |= D3DUSAGE_POINTS;
	if(FAILED(pD3DDevice->CreateVertexBuffer(this->size,usage,
											 0,D3DPOOL_DEFAULT,&this->pD3DBuffer,
											 NULL)))
	{
		SafeRelease(pD3DBuffer);
	}
}

HQReturnVal HQVertexBufferD3D9::Unmap()
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif

	this->pD3DBuffer->Unlock();

	return HQ_OK;
}

HQReturnVal HQVertexBufferD3D9::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif
	hq_uint32 i = offset + size;
	if (i > this->size)
		return HQ_FAILED_INVALID_SIZE;

	void *l_pData;
	if (FAILED(this->pD3DBuffer->Lock(offset, size, (void**)&l_pData, 0)))
		return HQ_FAILED;
	if (i == 0)
		memcpy(l_pData, pData, this->size);//update toàn bộ buffer
	else
		memcpy(l_pData, pData, size);

	this->pD3DBuffer->Unlock();
	return HQ_OK;
}
HQReturnVal HQVertexBufferD3D9::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif

	if (size == 0 && offset != 0)
		size = this->size - offset;

	DWORD lockFlags = 0;
	switch (mapType)
	{
	case HQ_MAP_DISCARD:
		lockFlags = D3DLOCK_DISCARD;
		break;
	case HQ_MAP_NOOVERWRITE:
		lockFlags = D3DLOCK_NOOVERWRITE;
		break;
	}

	if (FAILED(this->pD3DBuffer->Lock(offset, size, ppData, lockFlags)))
		return HQ_FAILED;

	return HQ_OK;
}

HQReturnVal HQVertexBufferD3D9::CopyContent(void * dest)
{
	void *lockedData;
	if (FAILED(this->pD3DBuffer->Lock(0, 0, &lockedData, D3DLOCK_READONLY)))
		return HQ_FAILED;

	memcpy(dest, lockedData, this->size);

	this->pD3DBuffer->Unlock();

	return HQ_OK;
}

//-------------------HQIndexBufferD3D9-------------------------
void HQIndexBufferD3D9::OnResetDevice()
{
	DWORD usage = D3DUSAGE_WRITEONLY;
	if (isDynamic)
		usage |= D3DUSAGE_DYNAMIC;
	
	D3DFORMAT format;
	switch (this->dataType)
	{
	case HQ_IDT_USHORT:
		format = D3DFMT_INDEX16;
		break;
	case HQ_IDT_UINT:
		format = D3DFMT_INDEX32;
		break;
	}

	if(FAILED(pD3DDevice->CreateIndexBuffer(this->size,
											usage,
											format,
											D3DPOOL_DEFAULT,
											&this->pD3DBuffer,
											NULL)))
	{
		SafeRelease(pD3DBuffer);
	}
}

HQReturnVal HQIndexBufferD3D9::Unmap()
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif

	this->pD3DBuffer->Unlock();

	return HQ_OK;
}

HQReturnVal HQIndexBufferD3D9::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif
	hq_uint32 i = offset + size;
	if (i > this->size)
		return HQ_FAILED_INVALID_SIZE;

	void *l_pData;
	if (FAILED(this->pD3DBuffer->Lock(offset, size, (void**)&l_pData, 0)))
		return HQ_FAILED;
	if (i == 0)
		memcpy(l_pData, pData, this->size);//update toàn bộ buffer
	else
		memcpy(l_pData, pData, size);

	this->pD3DBuffer->Unlock();
	return HQ_OK;
}
HQReturnVal HQIndexBufferD3D9::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif
	if (size == 0 && offset != 0)
		size = this->size - offset;

	DWORD lockFlags = 0;
	switch (mapType)
	{
	case HQ_MAP_DISCARD:
		lockFlags = D3DLOCK_DISCARD;
		break;
	case HQ_MAP_NOOVERWRITE:
		lockFlags = D3DLOCK_NOOVERWRITE;
		break;
	}

	if (FAILED(this->pD3DBuffer->Lock(offset, size, ppData, lockFlags)))
		return HQ_FAILED;

	return HQ_OK;
}

HQReturnVal HQIndexBufferD3D9::CopyContent(void * dest)
{
	void *lockedData;
	if (FAILED(this->pD3DBuffer->Lock(0, 0, &lockedData, D3DLOCK_READONLY)))
		return HQ_FAILED;

	memcpy(dest, lockedData, this->size);

	this->pD3DBuffer->Unlock();

	return HQ_OK;
}

/*---------vertex input layout---------*/
void HQVertexInputLayoutD3D9::OnResetDevice()
{
	if (FAILED(pD3DDevice->CreateVertexDeclaration(this->elements , &this->pD3DDecl)))
		SafeRelease(pD3DDecl);
}

/*---------vertex stream manager-------*/

HQVertexStreamManagerD3D9::HQVertexStreamManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice , HQLogStream *logFileStream, bool flushLog)
:HQLoggableObject(logFileStream , "D3D9 Vertex Stream Manager :" ,flushLog)
{
	this->pD3DDevice = pD3DDevice;

	Log("Init done!");
}

HQVertexStreamManagerD3D9::~HQVertexStreamManagerD3D9()
{
	Log("Released!");
}

HQReturnVal HQVertexStreamManagerD3D9::CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,HQVertexBuffer **pID)
{
	HQVertexBufferD3D9* newVBuffer = new HQVertexBufferD3D9(this->pD3DDevice , size , dynamic , isForPointSprites);

	if (newVBuffer->pD3DBuffer == NULL || !this->vertexBuffers.AddItem(newVBuffer , pID))
	{
		delete newVBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	
	if (initData)
	{
		void *pData;
		
		newVBuffer->pD3DBuffer->Lock(0 , 0 , &pData , 0);
		memcpy(pData , initData , size);
		newVBuffer->pD3DBuffer->Unlock();
	}
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerD3D9::CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , HQIndexBuffer **pID)
{
	HQIndexBufferD3D9* newIBuffer = new HQIndexBufferD3D9(this->pD3DDevice , size , dynamic , dataType);

	if (newIBuffer->pD3DBuffer == NULL || !this->indexBuffers.AddItem(newIBuffer , pID))
	{
		delete newIBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	if(initData)
	{
		void *pData;

		newIBuffer->pD3DBuffer->Lock(0 , 0 , &pData , 0);
		memcpy(pData , initData , size);
		newIBuffer->pD3DBuffer->Unlock();
	}
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerD3D9::CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												HQShaderObject* vertexShaderID , 
												HQVertexLayout **pID)
{
	if (vAttribDesc == NULL)
		return HQ_FAILED;
	if (numAttrib >= 16)
		return HQ_FAILED_TOO_MANY_ATTRIBUTES;
	D3DVERTEXELEMENT9 *elements = new D3DVERTEXELEMENT9[numAttrib + 1];
	if (elements == NULL)
		return HQ_FAILED_MEM_ALLOC;
	
	for (hq_uint32 i = 0 ; i < numAttrib ; ++i)
		this->ConvertToVertexElement(vAttribDesc[i] , elements[i]);
	D3DVERTEXELEMENT9 end = D3DDECL_END();
	elements[numAttrib] = end;

	HQVertexInputLayoutD3D9 *vLayout = new HQVertexInputLayoutD3D9(this->pD3DDevice , elements);
	if (vLayout == NULL)
	{
		delete elements;
		return HQ_FAILED_MEM_ALLOC;
	}

	if (!this->inputLayouts.AddItem(vLayout , pID))
	{
		delete vLayout;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

void HQVertexStreamManagerD3D9::ConvertToVertexElement(const HQVertexAttribDesc &vAttribDesc ,D3DVERTEXELEMENT9 &vElement)
{
	vElement.Stream = vAttribDesc.stream;
	vElement.Offset = vAttribDesc.offset;
	vElement.Method = D3DDECLMETHOD_DEFAULT;
	
	switch (vAttribDesc.usage)
	{
	case  HQ_VAU_POSITION:
		vElement.Usage = D3DDECLUSAGE_POSITION;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_COLOR:
		vElement.Usage = D3DDECLUSAGE_COLOR;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_NORMAL	:
		vElement.Usage = D3DDECLUSAGE_NORMAL;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_TEXCOORD0	:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_TEXCOORD1	:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 1;
		break;
	case HQ_VAU_TEXCOORD2	:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 2;
		break;
	case HQ_VAU_TEXCOORD3:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 3;
		break;
	case HQ_VAU_TEXCOORD4:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 4;
		break;
	case HQ_VAU_TEXCOORD5	:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 5;
		break;
	case HQ_VAU_TEXCOORD6:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 6;
		break;
	case HQ_VAU_TEXCOORD7	:
		vElement.Usage = D3DDECLUSAGE_TEXCOORD;
		vElement.UsageIndex = 7;
		break;
	case HQ_VAU_TANGENT	:
		vElement.Usage = D3DDECLUSAGE_TANGENT;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_BINORMAL:
		vElement.Usage = D3DDECLUSAGE_BINORMAL;
		vElement.UsageIndex = 0;
		break;
	case  HQ_VAU_BLENDWEIGHT:
		vElement.Usage = D3DDECLUSAGE_BLENDWEIGHT;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_BLENDINDICES:
		vElement.Usage = D3DDECLUSAGE_BLENDINDICES;
		vElement.UsageIndex = 0;
		break;
	case HQ_VAU_PSIZE	:
		vElement.Usage = D3DDECLUSAGE_PSIZE;
		vElement.UsageIndex = 0;
		break;
	}

	switch (vAttribDesc.dataType)
	{
	case HQ_VADT_FLOAT :
		vElement.Type = D3DDECLTYPE_FLOAT1;
		break;
	case HQ_VADT_FLOAT2 :
		vElement.Type = D3DDECLTYPE_FLOAT2;
		break;
	case HQ_VADT_FLOAT3 :
		vElement.Type = D3DDECLTYPE_FLOAT3;
		break;
	case HQ_VADT_FLOAT4 :
		vElement.Type = D3DDECLTYPE_FLOAT4;
		break;
	case HQ_VADT_UBYTE4 :
		vElement.Type = D3DDECLTYPE_UBYTE4;
		break;
	case HQ_VADT_SHORT2 :
		vElement.Type = D3DDECLTYPE_SHORT2;
		break;
	case HQ_VADT_SHORT4 :
		vElement.Type = D3DDECLTYPE_SHORT4;
		break;
	case HQ_VADT_USHORT2N :
		vElement.Type = D3DDECLTYPE_USHORT2N;
		break;
	case HQ_VADT_USHORT4N :
		vElement.Type = D3DDECLTYPE_USHORT4N;
		break;
	case HQ_VADT_UBYTE4N:
		vElement.Type = D3DDECLTYPE_D3DCOLOR;
		break;
	}
}

HQReturnVal HQVertexStreamManagerD3D9::SetVertexBuffer(HQVertexBuffer* vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride)
{
#if defined _DEBUG || defined DEBUG
	if (streamIndex >= 16)
		return HQ_FAILED;
#endif
	HQSharedPtr<HQVertexBufferD3D9> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID);
	if (vBuffer != this->streams[streamIndex].vertexBuffer || stride != this->streams[streamIndex].stride)
	{
		if (vBuffer == NULL)
			pD3DDevice->SetStreamSource(streamIndex , NULL , 0 , 0);
		else
			pD3DDevice->SetStreamSource(streamIndex , vBuffer->pD3DBuffer , 0 , stride);

		this->streams[streamIndex].vertexBuffer = vBuffer;
		this->streams[streamIndex].stride = stride;
	}
	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerD3D9::SetIndexBuffer(HQIndexBuffer* indexBufferID)
{
	HQSharedPtr<HQIndexBufferD3D9> iBuffer = this->indexBuffers.GetItemPointer(indexBufferID);
	if (this->activeIndexBuffer != iBuffer)
	{
		if (iBuffer == NULL )
			pD3DDevice->SetIndices(NULL);
		else
			pD3DDevice->SetIndices(iBuffer->pD3DBuffer);
		this->activeIndexBuffer = iBuffer;
	}

	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerD3D9::SetVertexInputLayout(HQVertexLayout* inputLayoutID)
{
	HQSharedPtr<HQVertexInputLayoutD3D9> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	if (this->activeInputLayout != pVLayout)
	{
		if (pVLayout == NULL )
			pD3DDevice->SetVertexDeclaration(NULL);
		else
			pD3DDevice->SetVertexDeclaration(pVLayout->pD3DDecl);
		this->activeInputLayout = pVLayout;
	}
	

	return HQ_OK;

}


HQReturnVal HQVertexStreamManagerD3D9::RemoveVertexBuffer(HQVertexBuffer* ID)
{
	return (HQReturnVal)this->vertexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerD3D9::RemoveIndexBuffer(HQIndexBuffer* ID)
{
	return (HQReturnVal)this->indexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerD3D9::RemoveVertexInputLayout(HQVertexLayout* ID)
{
	return (HQReturnVal)this->inputLayouts.Remove(ID);
}
void HQVertexStreamManagerD3D9::RemoveAllVertexBuffer() 
{
	this->vertexBuffers.RemoveAll();
}
void HQVertexStreamManagerD3D9::RemoveAllIndexBuffer() 
{
	this->indexBuffers.RemoveAll();
}
void HQVertexStreamManagerD3D9::RemoveAllVertexInputLayout()
{
	this->inputLayouts.RemoveAll();
}

void HQVertexStreamManagerD3D9::OnResetDevice()
{
	HQItemManager<HQVertexBufferD3D9>::Iterator itev;
	HQItemManager<HQIndexBufferD3D9>::Iterator itei;
	HQItemManager<HQVertexInputLayoutD3D9>::Iterator itel;
	this->vertexBuffers.GetIterator(itev);
	this->indexBuffers.GetIterator(itei);
	this->inputLayouts.GetIterator(itel);

	while (!itev.IsAtEnd())
	{
		itev->OnResetDevice();
		++itev;
	}
	while (!itei.IsAtEnd())
	{
		itei->OnResetDevice();
		++itei;
	}
	while (!itel.IsAtEnd())
	{
		itel->OnResetDevice();
		++itel;
	}

	/*-------reset buffers---------*/
	if (this->activeIndexBuffer != NULL)
		pD3DDevice->SetIndices(activeIndexBuffer->pD3DBuffer);
	else
		pD3DDevice->SetIndices(NULL);

	if (this->activeInputLayout != NULL)
		pD3DDevice->SetVertexDeclaration(activeInputLayout->pD3DDecl);
	else
		pD3DDevice->SetVertexDeclaration(NULL);

	for (hq_uint32 i = 0 ; i < 16 ; ++i)
	{
		if (this->streams[i].vertexBuffer != NULL)
			pD3DDevice->SetStreamSource(i , streams[i].vertexBuffer->pD3DBuffer , 0 , streams[i].stride);
	}
}


void HQVertexStreamManagerD3D9::OnLostDevice()
{
	HQItemManager<HQVertexBufferD3D9>::Iterator itev;
	HQItemManager<HQIndexBufferD3D9>::Iterator itei;
	HQItemManager<HQVertexInputLayoutD3D9>::Iterator itel;
	this->vertexBuffers.GetIterator(itev);
	this->indexBuffers.GetIterator(itei);
	this->inputLayouts.GetIterator(itel);

	while (!itev.IsAtEnd())
	{
		itev->OnLostDevice();
		++itev;
	}
	while (!itei.IsAtEnd())
	{
		itei->OnLostDevice();
		++itei;
	}
	while (!itel.IsAtEnd())
	{
		itel->OnLostDevice();
		++itel;
	}
}
