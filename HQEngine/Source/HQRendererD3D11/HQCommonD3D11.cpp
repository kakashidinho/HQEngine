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
#if defined _DEBUG || defined DEBUG	
	if (this->isDynamic == false)
	{
		this->pLog->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

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

	D3D11_MAPPED_SUBRESOURCE mapped;

	if (SUCCEEDED(pD3DContext->Map(this->pD3DBuffer, 0, mapType, 0, &mapped)))
	{
		memcpy((hq_ubyte8*)mapped.pData + offset, pData, size);

		pD3DContext->Unmap(this->pD3DBuffer, 0);
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