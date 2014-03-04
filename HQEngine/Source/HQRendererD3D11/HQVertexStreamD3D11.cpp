/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQVertexStreamD3D11.h"

#if HQ_D3D_CLEAR_VP_USE_GS
struct ClearBufferVertex
{
	HQColorui color;
	hq_float32 depth;
};
#else//#if !HQ_D3D_CLEAR_VP_USE_GS
struct ClearBufferVertex
{
	hqfloat32 position[2];
};
#endif//#if HQ_D3D_CLEAR_VP_USE_GS

const hq_uint32 g_clearVBStride = sizeof(ClearBufferVertex);
const hq_uint32 g_offset = 0;
ID3D11Buffer * const g_nullBuffer = NULL;

const char *semanticName[] =
{
#if HQ_DEFINE_SEMANTICS
	"POSITION",
	"COLOR",
	"NORMAL",
	"TANGENT",
	"BINORMAL",
	"BLENDWEIGHT",
	"BLENDINDICES",
	"TEXCOORD",
	"TEXCOORD",
	"TEXCOORD",
	"TEXCOORD",
	"TEXCOORD",
	"TEXCOORD",
	"TEXCOORD",
	"TEXCOORD",
	"PSIZE",
#else
	"VPOSITION",
	"VCOLOR",
	"VNORMAL",
	"VTANGENT",
	"VBINORMAL",
	"VBLENDWEIGHT",
	"VBLENDINDICES",
	"VTEXCOORD",
	"VTEXCOORD",
	"VTEXCOORD",
	"VTEXCOORD",
	"VTEXCOORD",
	"VTEXCOORD",
	"VTEXCOORD",
	"VTEXCOORD",
	"VPSIZE",
#endif//if HQ_DEFINE_SEMANTICS
};

/*---------vertex stream manager-------*/

HQVertexStreamManagerD3D11::HQVertexStreamManagerD3D11(ID3D11Device* pD3DDevice , 
													ID3D11DeviceContext *pD3DContext, 
													HQShaderManagerD3D11 *pShaderMan,
													HQLogStream* logFileStream , bool flushLog)
:HQLoggableObject(logFileStream , "D3D11 Vertex Stream Manager :" ,flushLog)
{
	this->pD3DDevice = pD3DDevice;
	this->pD3DContext = pD3DContext;
	this->pShaderMan = pShaderMan;
	
	/*------create vertex buffer and input layout for clearing viewport---------*/
	D3D11_BUFFER_DESC vbd;
#if HQ_D3D_CLEAR_VP_USE_GS
	vbd.Usage = D3D11_USAGE_DYNAMIC;
	vbd.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
	vbd.ByteWidth = g_clearVBStride;
#else
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.CPUAccessFlags = 0;
	vbd.ByteWidth = 4 * g_clearVBStride;
#endif
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

#if HQ_D3D_CLEAR_VP_USE_GS
	pD3DDevice->CreateBuffer(&vbd , NULL ,&this->pCleaVBuffer);

#else //#if !HQ_D3D_CLEAR_VP_USE_GS
	{
		ClearBufferVertex pV[4];
		D3D11_SUBRESOURCE_DATA subResource;
		//create full screen quad
		pV[0].position[0] = -1.0f; pV[0].position[1] = -1.0f;
		pV[1].position[0] = -1.0f; pV[1].position[1] = 1.0f;
		pV[2].position[0] = 1.0f; pV[2].position[1] = -1.0f;
		pV[3].position[0] = 1.0f; pV[3].position[1] = 1.0f;
		
		subResource.pSysMem = pV;

		pD3DDevice->CreateBuffer(&vbd , &subResource ,&this->pCleaVBuffer);
	}
#endif

#if HQ_D3D_CLEAR_VP_USE_GS
	const UINT numElements = 2;
	D3D11_INPUT_ELEMENT_DESC id[numElements];
	id[0].SemanticName = "COLOR";
	id[0].SemanticIndex = 0;
	id[0].Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	id[0].InputSlot = 0;
	id[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	id[0].InstanceDataStepRate = 0;
	id[0].AlignedByteOffset = 0;

	id[1].SemanticName = "DEPTH";
	id[1].SemanticIndex = 0;
	id[1].Format = DXGI_FORMAT_R32_FLOAT;
	id[1].InputSlot = 0;
	id[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	id[1].InstanceDataStepRate = 0;
	id[1].AlignedByteOffset = sizeof(HQColorui);
#else//#if !HQ_D3D_CLEAR_VP_USE_GS
	const UINT numElements = 1;
	D3D11_INPUT_ELEMENT_DESC id[numElements];
	id[0].SemanticName = "POSITION";
	id[0].SemanticIndex = 0;
	id[0].Format = DXGI_FORMAT_R32G32_FLOAT;
	id[0].InputSlot = 0;
	id[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	id[0].InstanceDataStepRate = 0;
	id[0].AlignedByteOffset = 0;
#endif//#if HQ_D3D_CLEAR_VP_USE_GS

	ID3DBlob *pBlob = pShaderMan->GetCompiledClearVShader();

	pD3DDevice->CreateInputLayout(id , numElements ,pBlob->GetBufferPointer() , pBlob->GetBufferSize(), &this->pClearInputLayout);

	Log("Init done!");
}

HQVertexStreamManagerD3D11::~HQVertexStreamManagerD3D11()
{
	SafeRelease(this->pCleaVBuffer);
	SafeRelease(this->pClearInputLayout);
	Log("Released!");
}

HQReturnVal HQVertexStreamManagerD3D11::CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,hq_uint32 *pID)
{
	HQBufferD3D11* newVBuffer = HQ_NEW HQBufferD3D11(dynamic , size);
	
	//tạo vertex buffer
	D3D11_BUFFER_DESC vbd;
	if (dynamic)
	{
		vbd.Usage = D3D11_USAGE_DYNAMIC;
		vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	}
	else
	{
		vbd.Usage = D3D11_USAGE_IMMUTABLE;
		vbd.CPUAccessFlags = 0;
	}
	vbd.ByteWidth = size;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = initData;

	D3D11_SUBRESOURCE_DATA *pInitiData = (initData == NULL) ? NULL : &vinitData;

	if(FAILED(pD3DDevice->CreateBuffer(&vbd , pInitiData , &newVBuffer->pD3DBuffer)) || 
		!this->vertexBuffers.AddItem(newVBuffer , pID))
	{
		delete newVBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerD3D11::CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , hq_uint32 *pID)
{
	HQIndexBufferD3D11* newIBuffer = HQ_NEW HQIndexBufferD3D11(dynamic , size , dataType);

	//tạo index buffer
	D3D11_BUFFER_DESC vbd;
	if (dynamic)
	{
		vbd.Usage = D3D11_USAGE_DYNAMIC;
		vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	}
	else
	{
		vbd.Usage = D3D11_USAGE_IMMUTABLE;
		vbd.CPUAccessFlags = 0;
	}
	vbd.ByteWidth = size;
	vbd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = initData;

	D3D11_SUBRESOURCE_DATA *pInitiData = (initData == NULL) ? NULL : &vinitData;

	if(FAILED(pD3DDevice->CreateBuffer(&vbd , pInitiData , &newIBuffer->pD3DBuffer)) || 
		!this->indexBuffers.AddItem(newIBuffer , pID))
	{
		delete newIBuffer;
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerD3D11::CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												hq_uint32 vertexShaderID , 
												hq_uint32 *pID)
{
	if (vAttribDesc == NULL)
		return HQ_FAILED;
	if (numAttrib >= MAX_VERTEX_ATTRIBS)
		return HQ_FAILED_TOO_MANY_ATTRIBUTES;

	if (HQ_NOT_USE_VSHADER == vertexShaderID)
		vertexShaderID = pShaderMan->GetFFVertexShaderForInputLayoutCreation();//get fixed function vertex shader
	
	ID3DBlob * pBlob = pShaderMan->GetCompiledVertexShader(vertexShaderID);
	if (pBlob == NULL)
		return HQ_FAILED;

	D3D11_INPUT_ELEMENT_DESC *elementDescs = HQ_NEW D3D11_INPUT_ELEMENT_DESC[numAttrib];
	if (elementDescs == NULL)
		return HQ_FAILED_MEM_ALLOC;
	
	for (hq_uint32 i = 0 ; i < numAttrib ; ++i)
		this->ConvertToElementDesc(vAttribDesc[i] , elementDescs[i]);

	HQVertexInputLayoutD3D11 *vLayout = HQ_NEW HQVertexInputLayoutD3D11();
	
	if (FAILED(pD3DDevice->CreateInputLayout(elementDescs , numAttrib , 
									  pBlob->GetBufferPointer(),
									  pBlob->GetBufferSize() , 
									  &vLayout->pD3DLayout)))
	{
		if (vertexShaderID == pShaderMan->GetFFVertexShaderForInputLayoutCreation())
			this->Log("Error : CreateVertexInputLayout() failed ! Input layout is not compatible with fixed function vertex shader");
		else
			this->Log("Error : CreateVertexInputLayout() failed ! Input layout is not compatible with vertex shader (%d)" , vertexShaderID);

		delete[] elementDescs;
		delete vLayout;
		return HQ_FAILED;
	}

	delete[] elementDescs;

	if (!this->inputLayouts.AddItem(vLayout , pID))
	{
		delete vLayout;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

void HQVertexStreamManagerD3D11::ConvertToElementDesc(const HQVertexAttribDesc &vAttribDesc ,D3D11_INPUT_ELEMENT_DESC &vElement)
{
	vElement.InputSlot = vAttribDesc.stream;
	vElement.AlignedByteOffset = vAttribDesc.offset;
	vElement.InstanceDataStepRate = 0;
	vElement.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	vElement.SemanticName = semanticName[vAttribDesc.usage];
	
	switch (vAttribDesc.usage)
	{
	case HQ_VAU_TEXCOORD1	:
		vElement.SemanticIndex = 1;
		break;
	case HQ_VAU_TEXCOORD2	:
		vElement.SemanticIndex = 2;
		break;
	case HQ_VAU_TEXCOORD3:
		vElement.SemanticIndex = 3;
		break;
	case HQ_VAU_TEXCOORD4:
		vElement.SemanticIndex = 4;
		break;
	case HQ_VAU_TEXCOORD5	:
		vElement.SemanticIndex = 5;
		break;
	case HQ_VAU_TEXCOORD6:
		vElement.SemanticIndex = 6;
		break;
	case HQ_VAU_TEXCOORD7	:
		vElement.SemanticIndex = 7;
		break;
	default:
		vElement.SemanticIndex = 0;
	}

	switch (vAttribDesc.dataType)
	{
	case HQ_VADT_FLOAT :
		vElement.Format = DXGI_FORMAT_R32_FLOAT;
		break;
	case HQ_VADT_FLOAT2 :
		vElement.Format = DXGI_FORMAT_R32G32_FLOAT;
		break;
	case HQ_VADT_FLOAT3 :
		vElement.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		break;
	case HQ_VADT_FLOAT4 :
		vElement.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		break;
	case HQ_VADT_UBYTE4 :
		vElement.Format = DXGI_FORMAT_R8G8B8A8_UINT;
		break;
	case HQ_VADT_SHORT :
		vElement.Format = DXGI_FORMAT_R16_SINT;
		break;
	case HQ_VADT_SHORT2 :
		vElement.Format = DXGI_FORMAT_R16G16_SINT;
		break;
	case HQ_VADT_SHORT4  :
		vElement.Format = DXGI_FORMAT_R16G16B16A16_SINT;
		break;
	case HQ_VADT_USHORT :
		vElement.Format = DXGI_FORMAT_R16_UINT;
		break;
	case HQ_VADT_USHORT2 :
		vElement.Format = DXGI_FORMAT_R16G16_UINT;
		break;
	case HQ_VADT_USHORT4  :
		vElement.Format = DXGI_FORMAT_R16G16B16A16_UINT;
		break;
	case HQ_VADT_USHORT2N :
		vElement.Format = DXGI_FORMAT_R16G16_UNORM;
		break;
	case HQ_VADT_USHORT4N :
		vElement.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
		break;
	case HQ_VADT_INT :
		vElement.Format = DXGI_FORMAT_R32_SINT;
		break;
	case HQ_VADT_INT2 :
		vElement.Format = DXGI_FORMAT_R32G32_SINT;
		break;
	case HQ_VADT_INT3 :
		vElement.Format = DXGI_FORMAT_R32G32B32_SINT;
		break;
	case HQ_VADT_INT4  :
		vElement.Format = DXGI_FORMAT_R32G32B32A32_SINT;
		break;
	case HQ_VADT_UINT :
		vElement.Format = DXGI_FORMAT_R32_UINT;
		break;
	case HQ_VADT_UINT2 :
		vElement.Format = DXGI_FORMAT_R32G32_UINT;
		break;
	case HQ_VADT_UINT3 :
		vElement.Format = DXGI_FORMAT_R32G32B32_UINT;
		break;
	case HQ_VADT_UINT4  :
		vElement.Format = DXGI_FORMAT_R32G32B32A32_UINT;
		break;
	case HQ_VADT_UBYTE4N:
		vElement.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		break;
	}
}

HQReturnVal HQVertexStreamManagerD3D11::SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride)
{
	if (streamIndex >= MAX_VERTEX_ATTRIBS)
		return HQ_FAILED;
	HQSharedPtr<HQBufferD3D11> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID);
	if (vBuffer != this->streams[streamIndex].vertexBuffer || stride != this->streams[streamIndex].stride)
	{
		if (vBuffer == NULL)
			pD3DContext->IASetVertexBuffers(streamIndex , 1 , &g_nullBuffer , &stride , &g_offset);
		else
			pD3DContext->IASetVertexBuffers(streamIndex , 1 , &vBuffer->pD3DBuffer , &stride , &g_offset);

		this->streams[streamIndex].vertexBuffer = vBuffer;
		this->streams[streamIndex].stride = stride;
	}
	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerD3D11::SetIndexBuffer(hq_uint32 indexBufferID)
{
	HQSharedPtr<HQIndexBufferD3D11> iBuffer = this->indexBuffers.GetItemPointer(indexBufferID);
	if (this->activeIndexBuffer != iBuffer)
	{
		if (iBuffer == NULL )
			pD3DContext->IASetIndexBuffer(NULL , DXGI_FORMAT_R16_UINT , 0);
		else
			pD3DContext->IASetIndexBuffer(iBuffer->pD3DBuffer , iBuffer->d3dDataType , 0);
		this->activeIndexBuffer = iBuffer;
	}

	return HQ_OK;
}

HQReturnVal  HQVertexStreamManagerD3D11::SetVertexInputLayout(hq_uint32 inputLayoutID) 
{
	HQSharedPtr<HQVertexInputLayoutD3D11> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	if (this->activeInputLayout != pVLayout)
	{
		if (pVLayout == NULL )
			pD3DContext->IASetInputLayout(NULL);
		else
			pD3DContext->IASetInputLayout(pVLayout->pD3DLayout);
		this->activeInputLayout = pVLayout;
	}
	

	return HQ_OK;

}

HQReturnVal HQVertexStreamManagerD3D11::MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) 
{
	HQBufferD3D11* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);

#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
	if (vBuffer->pD3DBuffer == NULL)
		return HQ_FAILED;
	if (vBuffer->isDynamic == false)
	{
		this->Log("Error : static buffer can't be mapped!");
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

	if (FAILED(pD3DContext->Map(vBuffer->pD3DBuffer , 0 , d3dmapType , 0 , &mappedSubResource)))
		return HQ_FAILED;

	*ppData = mappedSubResource.pData;

	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerD3D11::UnmapVertexBuffer(hq_uint32 vertexBufferID) 
{
	HQBufferD3D11* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);

#if defined _DEBUG || defined DEBUG
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
	if (vBuffer->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif
	pD3DContext->Unmap(vBuffer->pD3DBuffer , 0);
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerD3D11::MapIndexBuffer(HQMapType mapType , void **ppData) 
{

#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
	if (this->activeIndexBuffer->pD3DBuffer == NULL)
		return HQ_FAILED;
	if (this->activeIndexBuffer->isDynamic == false)
	{
		this->Log("Error : static buffer can't be mapped!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

	D3D11_MAP d3dmapType = D3D11_MAP_WRITE ;
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

	if (FAILED(pD3DContext->Map(activeIndexBuffer->pD3DBuffer , 0 , d3dmapType , 0 , &mappedSubResource)))
		return HQ_FAILED;

	*ppData = mappedSubResource.pData;

	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerD3D11::UnmapIndexBuffer() 
{
#if defined _DEBUG || defined DEBUG
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;

	if (this->activeIndexBuffer->pD3DBuffer == NULL)
		return HQ_FAILED;
#endif
	pD3DContext->Unmap(activeIndexBuffer->pD3DBuffer , 0);
	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerD3D11::UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQBufferD3D11* vBuffer = this->vertexBuffers.GetItemRawPointer(vertexBufferID);
#if defined _DEBUG || defined DEBUG	
	if (vBuffer == NULL)
		return HQ_FAILED_INVALID_ID;
	if (vBuffer->isDynamic == false)
	{
		this->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif
	
	hq_uint32 i = offset + size;
	if (i > vBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	D3D11_MAP mapType = D3D11_MAP_WRITE_NO_OVERWRITE;
	if (i == 0 )//update toàn bộ buffer
	{
		size = vBuffer->size; 
		mapType = D3D11_MAP_WRITE_DISCARD;
	}
	else if (offset == 0 && size == vBuffer->size)
		mapType = D3D11_MAP_WRITE_DISCARD;

	D3D11_MAPPED_SUBRESOURCE mapped;

	if(SUCCEEDED(pD3DContext->Map(vBuffer->pD3DBuffer , 0 , mapType , 0 , &mapped)))
	{
		memcpy((hq_ubyte8*)mapped.pData + offset , pData , size);

		pD3DContext->Unmap(vBuffer->pD3DBuffer , 0);
	}
	
	return HQ_OK;
}
HQReturnVal HQVertexStreamManagerD3D11::UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData)
{
#if defined _DEBUG || defined DEBUG	
	if (this->activeIndexBuffer == NULL)
		return HQ_FAILED;
	if (this->activeIndexBuffer->isDynamic == false)
	{
		this->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

	hq_uint32 i = offset + size;
	if (i > activeIndexBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	D3D11_MAP mapType = D3D11_MAP_WRITE_NO_OVERWRITE;
	if (i == 0 )//update toàn bộ buffer
	{
		size = activeIndexBuffer->size; 
		mapType = D3D11_MAP_WRITE_DISCARD;
	}
	else if (offset == 0 && size == activeIndexBuffer->size)
		mapType = D3D11_MAP_WRITE_DISCARD;

	D3D11_MAPPED_SUBRESOURCE mapped;

	if(SUCCEEDED(pD3DContext->Map(activeIndexBuffer->pD3DBuffer , 0 , mapType , 0 , &mapped)))
	{
		memcpy((hq_ubyte8*)mapped.pData + offset , pData , size);

		pD3DContext->Unmap(activeIndexBuffer->pD3DBuffer , 0);
	}

	return HQ_OK;
}

HQReturnVal HQVertexStreamManagerD3D11::UpdateIndexBuffer(hquint32 bufferID, hq_uint32 offset , hq_uint32 size , const void * pData)
{
	HQSharedPtr<HQIndexBufferD3D11> pBuffer = this->indexBuffers.GetItemPointer(bufferID);
#if defined _DEBUG || defined DEBUG	
	if (pBuffer == NULL)
		return HQ_FAILED;
	if (pBuffer->isDynamic == false)
	{
		this->Log("Error : static buffer can't be updated!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

	hq_uint32 i = offset + size;
	if (i > pBuffer->size)
		return HQ_FAILED_INVALID_SIZE;
	D3D11_MAP mapType = D3D11_MAP_WRITE_NO_OVERWRITE;
	if (i == 0 )//update toàn bộ buffer
	{
		size = pBuffer->size; 
		mapType = D3D11_MAP_WRITE_DISCARD;
	}
	else if (offset == 0 && size == pBuffer->size)
		mapType = D3D11_MAP_WRITE_DISCARD;

	D3D11_MAPPED_SUBRESOURCE mapped;

	if(SUCCEEDED(pD3DContext->Map(pBuffer->pD3DBuffer , 0 , mapType , 0 , &mapped)))
	{
		memcpy((hq_ubyte8*)mapped.pData + offset , pData , size);

		pD3DContext->Unmap(pBuffer->pD3DBuffer , 0);
	}

	return HQ_OK;
}


HQReturnVal HQVertexStreamManagerD3D11::RemoveVertexBuffer(hq_uint32 ID) 
{
	return (HQReturnVal)this->vertexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerD3D11::RemoveIndexBuffer(hq_uint32 ID) 
{
	return (HQReturnVal)this->indexBuffers.Remove(ID);
}
HQReturnVal HQVertexStreamManagerD3D11::RemoveVertexInputLayout(hq_uint32 ID) 
{
	return (HQReturnVal)this->inputLayouts.Remove(ID);
}
void HQVertexStreamManagerD3D11::RemoveAllVertexBuffer() 
{
	this->vertexBuffers.RemoveAll();
}
void HQVertexStreamManagerD3D11::RemoveAllIndexBuffer() 
{
	this->indexBuffers.RemoveAll();
}
void HQVertexStreamManagerD3D11::RemoveAllVertexInputLayout()
{
	this->inputLayouts.RemoveAll();
}

#if HQ_D3D_CLEAR_VP_USE_GS
void HQVertexStreamManagerD3D11::ChangeClearVBuffer(HQColorui color , hq_float32 depth)
{
	ClearBufferVertex *pV;
	D3D11_MAPPED_SUBRESOURCE mappedSubResource;

	pD3DContext->Map(this->pCleaVBuffer , 0 , D3D11_MAP_WRITE_DISCARD , 0 , &mappedSubResource);
	pV = (ClearBufferVertex *)mappedSubResource.pData;
	pV->color = color;
	pV->depth = depth;
	pD3DContext->Unmap(this->pCleaVBuffer , 0 );
}
#endif

void HQVertexStreamManagerD3D11::BeginClearViewport()
{
	pD3DContext->IASetInputLayout(this->pClearInputLayout);
	pD3DContext->IASetVertexBuffers(0 , 1 , &this->pCleaVBuffer ,&g_clearVBStride , &g_offset); 
}

void HQVertexStreamManagerD3D11::EndClearViewport()
{
	if (this->activeInputLayout != NULL)
		pD3DContext->IASetInputLayout(this->activeInputLayout->pD3DLayout);
	else
		pD3DContext->IASetInputLayout(NULL);
	if (this->streams[0].vertexBuffer != NULL)
		pD3DContext->IASetVertexBuffers(0 , 1 , &this->streams[0].vertexBuffer->pD3DBuffer ,&this->streams[0].stride , &g_offset); 
	else
		pD3DContext->IASetVertexBuffers(0 , 1 , &g_nullBuffer ,&this->streams[0].stride , &g_offset); 

}
