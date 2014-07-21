/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQCommonD3D11.h"

/*----------HQGenericBufferD3D11---------------*/
HQSharedPtr<HQPoolMemoryManager> HQGenericBufferD3D11::s_uavBoundSlotsMemManager;

/*-------------------HQBufferD3D11-------------------------*/
HQReturnVal HQBufferD3D11::Unmap()
{
	pD3DContext->Unmap(this->pD3DBuffer, 0);
	return HQ_OK;
}

HQReturnVal HQBufferD3D11::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	hq_uint32 i = offset + size;
	if (i > this->size)
		return HQ_FAILED_INVALID_SIZE;
	D3D11_MAP mapType = D3D11_MAP_WRITE_NO_OVERWRITE;
	if (i == 0)//update toàn bộ buffer
	{
		size = this->size;
		mapType = D3D11_MAP_WRITE_DISCARD;
	}
	else if (offset == 0 && size == this->size)
		mapType = D3D11_MAP_WRITE_DISCARD;

	if (this->isDynamic == true)
	{
		D3D11_MAPPED_SUBRESOURCE mapped;

		if (SUCCEEDED(pD3DContext->Map(this->pD3DBuffer, 0, mapType, 0, &mapped)))
		{
			memcpy((hq_ubyte8*)mapped.pData + offset, pData, size);

			pD3DContext->Unmap(this->pD3DBuffer, 0);
		}
	}
	else{
		D3D11_BOX box;
		box.top = 0;
		box.bottom = 1;
		box.front = 0;
		box.back = 1;
		box.left = offset;
		box.right = offset + size;

		pD3DContext->UpdateSubresource(this->pD3DBuffer, 0, &box, pData, 0, 0);
	}

	return HQ_OK;
}
HQReturnVal HQBufferD3D11::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
#if defined _DEBUG || defined DEBUG	
	if (this->pD3DBuffer == NULL)
		return HQ_FAILED;
	if (this->isDynamic == false)
	{
		this->pLog->Log("Error : static buffer can't be mapped!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

	D3D11_MAP d3dmapType = D3D11_MAP_WRITE;
	switch (mapType)
	{
	case HQ_MAP_DISCARD:
		d3dmapType = D3D11_MAP_WRITE_DISCARD;
		break;
	case HQ_MAP_NOOVERWRITE:
		d3dmapType = D3D11_MAP_WRITE_NO_OVERWRITE;
		break;
	default:
		d3dmapType = D3D11_MAP_WRITE_DISCARD;
	}

#if defined _DEBUG || defined DEBUG
	if (ppData == NULL)
		return HQ_FAILED;
#endif
	D3D11_MAPPED_SUBRESOURCE mappedSubResource;

	if (FAILED(pD3DContext->Map(this->pD3DBuffer, 0, d3dmapType, 0, &mappedSubResource)))
		return HQ_FAILED;

	*ppData = (hqubyte8*)mappedSubResource.pData + offset;

	return HQ_OK;
}

HQReturnVal HQBufferD3D11::CopyContent(void *dest)
{
	if (this->pD3DStagingBuffer == NULL)
	{
		//create staging buffer  to copy the content to
		D3D11_BUFFER_DESC vbd;
		this->pD3DBuffer->GetDesc(&vbd);//copy buffer size from {resource}

		vbd.Usage = D3D11_USAGE_STAGING;
		vbd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		vbd.BindFlags = 0;
		vbd.MiscFlags = 0;
		vbd.StructureByteStride = 0;

		ID3D11Device *pD3DDevice;
		this->pD3DBuffer->GetDevice(&pD3DDevice);

		if (FAILED(pD3DDevice->CreateBuffer(&vbd, NULL, &this->pD3DStagingBuffer)))
		{
			pD3DDevice->Release();
			return HQ_FAILED_MEM_ALLOC;
		}

		pD3DDevice->Release();
	}
	//copy content
	this->pD3DContext->CopyResource(this->pD3DStagingBuffer, this->pD3DBuffer);

	D3D11_MAPPED_SUBRESOURCE mappedSubResource;
	HQReturnVal re = HQ_FAILED;
	//now copy from staging buffer to {dest}
	if (SUCCEEDED(this->pD3DContext->Map(this->pD3DStagingBuffer, 0, D3D11_MAP_READ, 0, &mappedSubResource)))
	{
		memcpy(dest, mappedSubResource.pData, this->size);
		this->pD3DContext->Unmap(this->pD3DStagingBuffer, 0);

		re = HQ_OK;
	}

	return re;
}

HQReturnVal HQBufferD3D11::TransferData(HQGraphicsBufferRawRetrievable* src, hquint32 destOffset, hquint32 srcOffset, hquint32 size)
{
	HQBufferD3D11 * pSrcBufferD3D11 = static_cast<HQBufferD3D11*>(src);

	if (destOffset + srcOffset + size == 0)//entire buffer
	{
		pD3DContext->CopyResource(this->pD3DBuffer, pSrcBufferD3D11->pD3DBuffer);
	}
	else{
		D3D11_BOX src_box;
		src_box.top = 0;
		src_box.bottom = 1;
		src_box.front = 0;
		src_box.back = 1;
		src_box.left = srcOffset;
		src_box.right = srcOffset + size;

		pD3DContext->CopySubresourceRegion(this->pD3DBuffer, 0, destOffset, 0, 0, pSrcBufferD3D11->pD3DBuffer, 0, &src_box);
	}

	return HQ_OK;
}


HQReturnVal CopyD3D11BufferContent(void *dest, ID3D11Buffer * resource)
{
	//create temp readable buffer to copy the content to
	D3D11_BUFFER_DESC vbd;
	resource->GetDesc(&vbd);//copy buffer size from {resource}

	vbd.Usage = D3D11_USAGE_STAGING;
	vbd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	vbd.BindFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	ID3D11Buffer *tempBuffer = NULL;
	ID3D11Device *pD3DDevice;
	ID3D11DeviceContext *pD3DContext;
	resource->GetDevice(&pD3DDevice);
	pD3DDevice->GetImmediateContext(&pD3DContext);

	if (FAILED(pD3DDevice->CreateBuffer(&vbd, NULL, &tempBuffer)))
	{
		pD3DContext->Release();
		pD3DDevice->Release();
		return HQ_FAILED_MEM_ALLOC;
	}

	//copy content
	pD3DContext->CopyResource(tempBuffer, resource);

	D3D11_MAPPED_SUBRESOURCE mappedSubResource;
	HQReturnVal re = HQ_FAILED;
	//now copy from temp buffer to {dest}
	if (SUCCEEDED(pD3DContext->Map(tempBuffer, 0, D3D11_MAP_READ, 0, &mappedSubResource)))
	{
		memcpy(dest, mappedSubResource.pData, vbd.ByteWidth);
		pD3DContext->Unmap(tempBuffer, 0);

		re = HQ_OK;
	}
	

	pD3DContext->Release();
	pD3DDevice->Release();
	tempBuffer->Release();

	return re;
}


HQReturnVal CopyD3D11Texture2DContent(void *dest, ID3D11Texture2D * resource, size_t sizeToCopy)
{
	//create temp readable texture  to copy the content to
	D3D11_TEXTURE2D_DESC desc;
	resource->GetDesc(&desc);//copy texture format and size from {resource}

	desc.Usage = D3D11_USAGE_STAGING;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.BindFlags = 0;
	desc.MiscFlags = 0;

	ID3D11Texture2D *tempBuffer = NULL;
	ID3D11Device *pD3DDevice;
	ID3D11DeviceContext *pD3DContext;
	resource->GetDevice(&pD3DDevice);
	pD3DDevice->GetImmediateContext(&pD3DContext);

	if (FAILED(pD3DDevice->CreateTexture2D(&desc, NULL, &tempBuffer)))
	{
		pD3DContext->Release();
		pD3DDevice->Release();
		return HQ_FAILED_MEM_ALLOC;
	}

	//copy content
	pD3DContext->CopyResource(tempBuffer, resource);

	D3D11_MAPPED_SUBRESOURCE mappedSubResource;
	HQReturnVal re = HQ_OK;
	//now copy from temp buffer to {dest}
	UINT oneArraySliceSize = sizeToCopy / desc.ArraySize;
	for (UINT i = 0; i < desc.ArraySize; ++i)
	{
		//copy each array slice turn by turn
		UINT subResource = D3D11CalcSubresource(0, i, desc.MipLevels);
		if (SUCCEEDED(pD3DContext->Map(tempBuffer, subResource, D3D11_MAP_READ, 0, &mappedSubResource)))
		{
			memcpy(((hqubyte8*)dest) + oneArraySliceSize * i, mappedSubResource.pData, oneArraySliceSize);
			pD3DContext->Unmap(tempBuffer, subResource);

		}
		else {
			re = HQ_FAILED;
		}
	}


	pD3DContext->Release();
	pD3DDevice->Release();
	tempBuffer->Release();

	return re;
}