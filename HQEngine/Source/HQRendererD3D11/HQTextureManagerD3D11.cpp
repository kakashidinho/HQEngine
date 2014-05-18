/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQTextureManagerD3D11.h"
#include "HQDeviceD3D11.h"

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM || defined _STATIC_CRT)
#define HQ_USE_PVR_TEX_LIB 1
#else
#define HQ_USE_PVR_TEX_LIB 0
#endif


#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include <d3dx11.h>
#else
#include "../HQEngine/winstore/HQWinStoreFileSystem.h"
#endif
#include <math.h>

#if HQ_USE_PVR_TEX_LIB
#include "PVRTexLib.h"
using namespace pvrtexlib;
#endif //#if HQ_USE_PVR_TEX_LIB


//************************************************************
//định dạng của texture tương ứng với định dạng của pixel ảnh
//************************************************************
static DXGI_FORMAT GetTextureFmt(const SurfaceFormat imgFmt)
{
	switch (imgFmt)
	{
	case FMT_R32G32B32A32_FLOAT:
		return DXGI_FORMAT_R32G32B32A32_FLOAT;
	case FMT_R32G32_FLOAT:
		return DXGI_FORMAT_R32G32_FLOAT;
	case FMT_R32_FLOAT:
		return DXGI_FORMAT_R32_FLOAT;
	case FMT_R8G8B8:
	case FMT_B8G8R8:
	case FMT_A8R8G8B8:
	case FMT_X8R8G8B8:
	case FMT_R5G6B5:
	case FMT_B5G6R5:
	case FMT_L8:
	case FMT_A8L8:
	case FMT_A8B8G8R8:
	case FMT_X8B8G8R8:
	case FMT_R8G8B8A8:
	case FMT_B8G8R8A8:
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	case FMT_A8:
		return DXGI_FORMAT_A8_UNORM;
	case FMT_S3TC_DXT1:
		return DXGI_FORMAT_BC1_UNORM;
	case FMT_S3TC_DXT3:
		return DXGI_FORMAT_BC2_UNORM;
	case FMT_S3TC_DXT5:
		return DXGI_FORMAT_BC3_UNORM;
	default:
		return DXGI_FORMAT_UNKNOWN;
	}
}


static DXGI_FORMAT GetTextureBufferFormat(HQTextureBufferFormat format, hq_uint32 &texelSize)
{
	switch (format)
	{
	case HQ_TBFMT_R16_FLOAT:
		texelSize = 2;
		return DXGI_FORMAT_R16_FLOAT;
	case HQ_TBFMT_R16G16B16A16_FLOAT:
		texelSize = 8;
		return DXGI_FORMAT_R16G16B16A16_FLOAT;
	case HQ_TBFMT_R32_FLOAT:
		texelSize = 4;
		return DXGI_FORMAT_R32_FLOAT;
	case HQ_TBFMT_R32G32B32_FLOAT:
		texelSize = 12;
		return DXGI_FORMAT_R32G32B32_FLOAT;
	case HQ_TBFMT_R32G32B32A32_FLOAT:
		texelSize = 16;
		return DXGI_FORMAT_R32G32B32A32_FLOAT;
	case HQ_TBFMT_R8_INT:
		texelSize = 1;
		return DXGI_FORMAT_R8_SINT;
	case HQ_TBFMT_R8G8B8A8_INT:
		texelSize = 4;
		return DXGI_FORMAT_R8G8B8A8_SINT;
	case HQ_TBFMT_R8_UINT:
		texelSize = 1;
		return DXGI_FORMAT_R8_UINT;
	case HQ_TBFMT_R8G8B8A8_UINT:
		texelSize = 4;
		return DXGI_FORMAT_R8G8B8A8_UINT;
	case HQ_TBFMT_R16_INT:
		texelSize = 2;
		return DXGI_FORMAT_R16_SINT;
	case HQ_TBFMT_R16G16B16A16_INT:
		texelSize = 8;
		return DXGI_FORMAT_R16G16B16A16_SINT;
	case HQ_TBFMT_R16_UINT:
		texelSize = 2;
		return DXGI_FORMAT_R16_UINT;
	case HQ_TBFMT_R16G16B16A16_UINT:
		texelSize = 8;
		return DXGI_FORMAT_R16G16B16A16_UINT;
	case HQ_TBFMT_R32_INT:
		texelSize = 4;
		return DXGI_FORMAT_R32_SINT;
	case HQ_TBFMT_R32G32B32A32_INT:
		texelSize = 16;
		return DXGI_FORMAT_R32G32B32A32_SINT;
	case HQ_TBFMT_R32_UINT:
		texelSize = 4;
		return DXGI_FORMAT_R32_UINT;
	case HQ_TBFMT_R32G32B32A32_UINT:
		texelSize = 16;
		return DXGI_FORMAT_R32G32B32A32_UINT;
	case HQ_TBFMT_R8_UNORM:
		texelSize = 1;
		return DXGI_FORMAT_R8_UNORM;
	case HQ_TBFMT_R8G8B8A8_UNORM:
		texelSize = 4;
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	case HQ_TBFMT_R16_UNORM:
		texelSize = 2;
		return DXGI_FORMAT_R16_UNORM;
	case HQ_TBFMT_R16G16B16A16_UNORM:
		texelSize = 8;
		return DXGI_FORMAT_R16G16B16A16_UNORM;
	default:
		return DXGI_FORMAT_UNKNOWN;
	}
}


static hquint32 GetTexelSize(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_SINT: case DXGI_FORMAT_R8G8B8A8_SNORM:
	case DXGI_FORMAT_R8G8B8A8_UINT:
	case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_SINT:
	case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_TYPELESS:
		return 4;
	case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_TYPELESS:
		return 2;
	case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32_SINT:
	case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_TYPELESS:
		return 8;
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_SINT:
	case DXGI_FORMAT_R32G32B32A32_UINT:
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
		return 16;
	default:
		//TO DO
		return 0;
	}
}

/*--------------HQUAVTextureResourceD3D11---------------*/
struct HQUAVTextureResourceD3D11 : public HQTextureResourceD3D11
{
	struct LayerViewKey{
		UINT mipLevel;
		hquint32 flags;
		hquint32 hashCode;

		bool Equal(const LayerViewKey * key2) const { return mipLevel == key2->mipLevel && flags == key2->flags; }
		hquint32 HashCode() const{ return hashCode; }

		void cacheHashCode(){ hashCode = mipLevel * 29 + flags; }
	};

	struct LayerView {
		LayerView() : pD3DView(NULL){}
		~LayerView(){
			SafeRelease(pD3DView);
		}

		LayerViewKey key;
		ID3D11UnorderedAccessView* pD3DView;
	};

	ID3D11UnorderedAccessView * GetOrCreateNewUAV(ID3D11Device* creator, HQTextureType type, UINT miplevel, bool read);

	HQUAVTextureResourceD3D11();
	~HQUAVTextureResourceD3D11();

	typedef HQClosedPtrKeyHashTable<const LayerViewKey*, HQSharedPtr<LayerView> > LayeredViewTableType;
	LayeredViewTableType layeredViews;

	LayerViewKey cachedKey;
	ID3D11UnorderedAccessView * cachedView;

	HQTextureUAVFormat hqFormat;

	static DXGI_FORMAT GetD3DResFormat(HQTextureUAVFormat format);//get resource format
	static DXGI_FORMAT GetD3DViewFormat(HQTextureUAVFormat format);//get resource view format

private:
	DXGI_FORMAT GetNoReadUAViewFormat();//get UAV format that doesn't support read
	DXGI_FORMAT GetUAViewFormat();//get UAV format that supports read
};

HQUAVTextureResourceD3D11::HQUAVTextureResourceD3D11()
: cachedView(NULL)
{
}

HQUAVTextureResourceD3D11::~HQUAVTextureResourceD3D11()
{
}

HQ_FORCE_INLINE DXGI_FORMAT HQUAVTextureResourceD3D11::GetNoReadUAViewFormat(){
	switch (this->hqFormat) {
	case HQ_UAVTFMT_R8G8B8A8_UNORM://sepcial case, we created typeless texture, so we need to create formatted UAV view
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	default:
		return HQUAVTextureResourceD3D11::GetD3DViewFormat(this->hqFormat);//usa same format as resource view
	}
}

//return format that supports read
HQ_FORCE_INLINE DXGI_FORMAT HQUAVTextureResourceD3D11::GetUAViewFormat()
{
	switch (this->hqFormat) {
	case HQ_UAVTFMT_R8G8B8A8_UNORM://sepcial case, we created typeless texture, so we need to create formatted UAV view
		return DXGI_FORMAT_R32_UINT;
	case HQ_UAVTFMT_R16_FLOAT:
	case HQ_UAVTFMT_R16G16_FLOAT:
	case HQ_UAVTFMT_R16G16B16A16_FLOAT:
		return DXGI_FORMAT_R16_FLOAT;
	case HQ_UAVTFMT_R32_FLOAT:
	case HQ_UAVTFMT_R32G32_FLOAT:
	case HQ_UAVTFMT_R32G32B32A32_FLOAT:
		return DXGI_FORMAT_R32_FLOAT;
	case HQ_UAVTFMT_R32_INT:
	case HQ_UAVTFMT_R32G32_INT:
	case HQ_UAVTFMT_R32G32B32A32_INT:
		return DXGI_FORMAT_R32_SINT;
	case HQ_UAVTFMT_R32_UINT:
	case HQ_UAVTFMT_R32G32_UINT:
	case HQ_UAVTFMT_R32G32B32A32_UINT:
		return DXGI_FORMAT_R32_UINT;
	default:
		return DXGI_FORMAT_UNKNOWN;
	}
}

HQ_FORCE_INLINE ID3D11UnorderedAccessView * HQUAVTextureResourceD3D11::GetOrCreateNewUAV(ID3D11Device* creator, HQTextureType type, UINT miplevel, bool read)
{
	//try to return cached value first
	hquint32 l_flags = read ? 1 : 0;
	if (this->cachedView != NULL && this->cachedKey.mipLevel == miplevel && this->cachedKey.flags == l_flags)
		return this->cachedView;

	//try to find in table
	this->cachedKey.mipLevel = miplevel;
	this->cachedKey.flags = l_flags;
	this->cachedKey.cacheHashCode();
	bool found = false;
	HQSharedPtr<LayerView> view = this->layeredViews.GetItem(&this->cachedKey, found);
	if (!found){
		//create new view
		view = HQ_NEW LayerView();
		view->key = this->cachedKey;

		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		switch (type)
		{
		case HQ_TEXTURE_2D_UAV: case HQ_TEXTURE_2D_ARRAY_UAV:
		{
								  //texture desc
								  D3D11_TEXTURE2D_DESC l_t2DDesc;
								  static_cast<ID3D11Texture2D*>(this->pTexture)->GetDesc(&l_t2DDesc);
								  //UAV desc
								  if (read)
									  uavDesc.Format = this->GetUAViewFormat();
								  else
									  uavDesc.Format = this->GetNoReadUAViewFormat();
								  if (type == HQ_TEXTURE_2D_UAV)
								  {
									  uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
									  if (miplevel >= l_t2DDesc.MipLevels)
										  uavDesc.Texture2D.MipSlice = l_t2DDesc.MipLevels - 1;// mip level view
									  else
										  uavDesc.Texture2D.MipSlice = miplevel;// mip level view
								  }
								  else //array view
								  {
									  uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
									  uavDesc.Texture2DArray.ArraySize = l_t2DDesc.ArraySize;
									  uavDesc.Texture2DArray.FirstArraySlice = 0;
									  if (miplevel >= l_t2DDesc.MipLevels)
										  uavDesc.Texture2DArray.MipSlice = l_t2DDesc.MipLevels - 1;// mip level view
									  else
										  uavDesc.Texture2DArray.MipSlice = miplevel;// mip level view
								  }
		}
			break;
		}//switch (type)

		//create  UAV view
		HRESULT hr = creator->CreateUnorderedAccessView(this->pTexture, &uavDesc, &view->pD3DView);
		if (FAILED(hr))
			return NULL;

		this->layeredViews.Add(&view->key, view);//add to layered view table

	}//if (!found)

	//cache view
	this->cachedView = view->pD3DView;

	return view->pD3DView;
}

//get resource format
DXGI_FORMAT HQUAVTextureResourceD3D11::GetD3DResFormat(HQTextureUAVFormat format)
{
	switch (format)
	{
	case HQ_UAVTFMT_R16_FLOAT:
		return DXGI_FORMAT_R16_TYPELESS;
	case HQ_UAVTFMT_R16G16_FLOAT:
		return DXGI_FORMAT_R16G16_TYPELESS;
	case HQ_UAVTFMT_R16G16B16A16_FLOAT:
		return DXGI_FORMAT_R16G16B16A16_TYPELESS;
	case HQ_UAVTFMT_R32_FLOAT:
		return DXGI_FORMAT_R32_TYPELESS;
	case HQ_UAVTFMT_R32G32_FLOAT:
		return DXGI_FORMAT_R32G32_TYPELESS;
	case HQ_UAVTFMT_R32G32B32A32_FLOAT:
		return DXGI_FORMAT_R32G32B32A32_TYPELESS;
	case HQ_UAVTFMT_R32_INT:
		return DXGI_FORMAT_R32_TYPELESS;
	case HQ_UAVTFMT_R32G32_INT:
		return DXGI_FORMAT_R32G32_TYPELESS;
	case HQ_UAVTFMT_R32G32B32A32_INT:
		return DXGI_FORMAT_R32G32B32A32_TYPELESS;
	case HQ_UAVTFMT_R32_UINT:
		return DXGI_FORMAT_R32_TYPELESS;
	case HQ_UAVTFMT_R32G32_UINT:
		return DXGI_FORMAT_R32G32_TYPELESS;
	case HQ_UAVTFMT_R32G32B32A32_UINT:
		return DXGI_FORMAT_R32G32B32A32_TYPELESS;
	case HQ_UAVTFMT_R8G8B8A8_UNORM:
		return DXGI_FORMAT_R8G8B8A8_TYPELESS;
	default:
		return DXGI_FORMAT_UNKNOWN;
	}
}

//get resource view format
DXGI_FORMAT HQUAVTextureResourceD3D11::GetD3DViewFormat(HQTextureUAVFormat format)
{
	switch (format)
	{
	case HQ_UAVTFMT_R16_FLOAT:
		return DXGI_FORMAT_R16_FLOAT;
	case HQ_UAVTFMT_R16G16_FLOAT:
		return DXGI_FORMAT_R16G16_FLOAT;
	case HQ_UAVTFMT_R16G16B16A16_FLOAT:
		return DXGI_FORMAT_R16G16B16A16_FLOAT;
	case HQ_UAVTFMT_R32_FLOAT:
		return DXGI_FORMAT_R32_FLOAT;
	case HQ_UAVTFMT_R32G32_FLOAT:
		return DXGI_FORMAT_R32G32_FLOAT;
	case HQ_UAVTFMT_R32G32B32A32_FLOAT:
		return DXGI_FORMAT_R32G32B32A32_FLOAT;
	case HQ_UAVTFMT_R32_INT:
		return DXGI_FORMAT_R32_SINT;
	case HQ_UAVTFMT_R32G32_INT:
		return DXGI_FORMAT_R32G32_SINT;
	case HQ_UAVTFMT_R32G32B32A32_INT:
		return DXGI_FORMAT_R32G32B32A32_SINT;
	case HQ_UAVTFMT_R32_UINT:
		return DXGI_FORMAT_R32_UINT;
	case HQ_UAVTFMT_R32G32_UINT:
		return DXGI_FORMAT_R32G32_UINT;
	case HQ_UAVTFMT_R32G32B32A32_UINT:
		return DXGI_FORMAT_R32G32B32A32_UINT;
	case HQ_UAVTFMT_R8G8B8A8_UNORM:
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	default:
		return DXGI_FORMAT_UNKNOWN;
	}
}

/*---------HQTextureD3D11-----------------*/

HQSharedPtr<HQPoolMemoryManager> HQTextureD3D11::s_boundSlotsMemManager;

HQTextureD3D11::HQTextureD3D11(HQTextureType type)
:HQBaseTexture(),
boundSlots(s_boundSlotsMemManager),
uavBoundSlots( HQGenericBufferD3D11::s_uavBoundSlotsMemManager) //use same slots as uav buffer
{
	HQ_ASSERT(s_boundSlotsMemManager != NULL);

	this->type = type;
	switch (type)
	{
	case HQ_TEXTURE_2D_UAV: case HQ_TEXTURE_2D_ARRAY_UAV:
		pData = HQ_NEW HQUAVTextureResourceD3D11();
		break;
		//TO DO: other UAV texture type
	default:
		pData = HQ_NEW HQTextureResourceD3D11();
	}
}
HQTextureD3D11::~HQTextureD3D11()
{
	if(pData)
	{
		HQTextureResourceD3D11* pTex = (HQTextureResourceD3D11 *)pData;
		HQ_DELETE (pTex);
	}
}

hquint32 HQTextureD3D11::GetWidth() const
{
	switch (this->type)
	{
	case HQ_TEXTURE_2D:
	case HQ_TEXTURE_CUBE:
	case HQ_TEXTURE_2D_UAV:
	case HQ_TEXTURE_2D_ARRAY:
	case HQ_TEXTURE_2D_ARRAY_UAV:
		{
			ID3D11Texture2D *textureD3D = (ID3D11Texture2D *)((HQTextureResourceD3D11*)this->pData)->pTexture;
			D3D11_TEXTURE2D_DESC desc;
			textureD3D->GetDesc(&desc);
			return desc.Width;
		}
		break;
	default:
		return 0;
	}
}
hquint32 HQTextureD3D11::GetHeight() const
{
	switch (this->type)
	{
	case HQ_TEXTURE_2D:
	case HQ_TEXTURE_CUBE:
	case HQ_TEXTURE_2D_UAV:
	case HQ_TEXTURE_2D_ARRAY:
	case HQ_TEXTURE_2D_ARRAY_UAV:
		{
			ID3D11Texture2D *textureD3D = (ID3D11Texture2D *)((HQTextureResourceD3D11*)this->pData)->pTexture;
			D3D11_TEXTURE2D_DESC desc;
			textureD3D->GetDesc(&desc);
			return desc.Height;
		}
		break;
	default:
		return 0;
	}
}

HQReturnVal HQTextureD3D11::CopyFirstLevelContent(void *data)
{
	switch (this->type)
	{
	case HQ_TEXTURE_2D_UAV:
	{
		HQUAVTextureResourceD3D11* pD3Dres = (HQUAVTextureResourceD3D11*)this->pData;
		D3D11_TEXTURE2D_DESC desc;
		ID3D11Texture2D* pTexture = static_cast<ID3D11Texture2D*>(pD3Dres->pTexture);
		pTexture->GetDesc(&desc);
		hquint32 texelSize = GetTexelSize(desc.Format);
		if (texelSize != 0)
		{
			return CopyD3D11Texture2DContent(data, pTexture, desc.Width * desc.Height * texelSize);
		}
	}
	default:
		//TO DO
		return HQ_FAILED;
	}
}

HQReturnVal HQTextureD3D11::SetLevelContent(hquint32 level, const void *data)
{
	HQTextureResourceD3D11* pTex = (HQTextureResourceD3D11 *)pData;
	
	hquint32 subresIdx = 0;
	hquint32 srcRowPitch = 0;
	hquint32 srcDepthPitch = 0;
	ID3D11Resource* pD3Dres = pTex->pTexture;
	ID3D11Device * pD3DDevice; 
	ID3D11DeviceContext * pD3DContext;
	pD3Dres->GetDevice(&pD3DDevice);

	pD3DDevice->GetImmediateContext(&pD3DContext);

	switch (this->type)
	{
	case HQ_TEXTURE_2D_UAV:
	{
		HQUAVTextureResourceD3D11* pD3Dres = (HQUAVTextureResourceD3D11*)this->pData;
		D3D11_TEXTURE2D_DESC desc;
		ID3D11Texture2D* pTexture = static_cast<ID3D11Texture2D*>(pD3Dres->pTexture);
		pTexture->GetDesc(&desc);

		if (level >= desc.MipLevels)
			return HQ_FAILED;

		hquint32 texelSize = GetTexelSize(desc.Format);
		hquint32 levelWidth = desc.Width >> level;
		if (levelWidth == 0)
			levelWidth = 1;
		srcRowPitch = levelWidth * texelSize;
		subresIdx = D3D11CalcSubresource(level, 0, desc.MipLevels);
		break;
	}
	default:
		//TO DO
		return HQ_FAILED;
	}

	pD3DContext->UpdateSubresource(pD3Dres, subresIdx, NULL, data, srcRowPitch, srcDepthPitch);

	pD3DContext->Release();
	pD3DDevice->Release();
	return HQ_OK;
}

//implement HQGraphicsResourceRawRetrievable
void * HQTextureD3D11::GetRawHandle()
{
	if (this->pData == NULL)
		return NULL;
	return ((HQTextureResourceD3D11*)this->pData)->pTexture;
}


/*---------HQTextureBufferD3D11-----------------*/
struct HQTextureBufferD3D11 : public HQTextureD3D11{
	HQTextureBufferD3D11() : HQTextureD3D11(HQ_TEXTURE_BUFFER)
	{
	}

	virtual hquint32 GetWidth() const { return size; }

	virtual hquint32 GetHeight() const { return 1; }

	virtual hquint32 GetSize() const { return size; }

	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
	virtual HQReturnVal CopyContent(void *dest);


	ID3D11DeviceContext *pD3DContext;
	HQLoggableObject *pLog;
	hquint32 size;
	bool isDynamic;
};

HQReturnVal HQTextureBufferD3D11::Unmap()
{
	ID3D11Resource * pD3DBuffer = ((HQTextureResourceD3D11*)this->pData)->pTexture;
	pD3DContext->Unmap(pD3DBuffer, 0);
	return HQ_OK;
}

HQReturnVal HQTextureBufferD3D11::Update(hq_uint32 offset, hq_uint32 size, const void * pData)
{
	ID3D11Resource * pD3DBuffer = ((HQTextureResourceD3D11*)this->pData)->pTexture;
#if defined _DEBUG || defined DEBUG	
	if (offset != 0 || size != this->size)
	{
		pLog->Log("Error : texture buffer can't be updated partially!");
		return HQ_FAILED;
	}
#endif

	if (this->isDynamic == true)
	{
		//update entire buffer
		D3D11_MAPPED_SUBRESOURCE mappedSubResource;
		if (SUCCEEDED(pD3DContext->Map(pD3DBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubResource)))
		{
			//copy
			memcpy(mappedSubResource.pData, pData, this->size);

			pD3DContext->Unmap(pD3DBuffer, 0);
		}
	}
	else
	{
		//update entire buffer
		pD3DContext->UpdateSubresource(pD3DBuffer, 0, NULL, pData, 0, 0);
	}

	return HQ_OK;
}
HQReturnVal HQTextureBufferD3D11::GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size)
{
	ID3D11Resource * pD3DBuffer = ((HQTextureResourceD3D11*)this->pData)->pTexture;
#if defined DEBUG || defined _DEBUG
	if (!ppData)
		return HQ_FAILED;
	if (mapType != HQ_MAP_DISCARD)
		this->pLog->Log("Warning : texture buffer can only be mapped discard");
	if (offset != 0 || (size != this->size && size != 0))
	{
		pLog->Log("Error : texture buffer can't be updated partially!");
		return HQ_FAILED;
	}
#endif
	D3D11_MAPPED_SUBRESOURCE mappedSubResource;
	pD3DContext->Map(pD3DBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubResource);
	*ppData = mappedSubResource.pData;

	return HQ_OK;
}

HQReturnVal HQTextureBufferD3D11::CopyContent(void *dest)
{
	ID3D11Resource * pD3DBuffer = ((HQTextureResourceD3D11*)this->pData)->pTexture;
	return CopyD3D11BufferContent(dest, static_cast<ID3D11Buffer*>(pD3DBuffer));
}




/*
//HQTextureManagerD3D11 class
*/
/*
//constructor
*/
HQTextureManagerD3D11::HQTextureManagerD3D11(ID3D11Device* pDev , ID3D11DeviceContext *pContext, HQLogStream* logFileStream , bool flushLog)
: HQBaseTextureManager(logFileStream , "D3D11 Texture Manager :" , flushLog)
{
	pMasterDevice = g_pD3DDev;
	pD3DDevice=pDev;
	pD3DContext = pContext;
	//create memory manager for texture object
	HQTextureD3D11::s_boundSlotsMemManager = HQ_NEW HQPoolMemoryManager(sizeof(HQTextureD3D11::SlotList::LinkedListNodeType), pMasterDevice->GetCaps().maxTotalBoundTextures);


	bitmap.SetLoadedOutputRGBLayout(LAYOUT_BGR);
	bitmap.SetLoadedOutputRGB16Layout(LAYOUT_BGR);

	t2DDesc.ArraySize = 1;
	t2DDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	t2DDesc.Usage = D3D11_USAGE_DEFAULT;
	t2DDesc.CPUAccessFlags = 0;
	t2DDesc.SampleDesc.Count = 1;
	t2DDesc.SampleDesc.Quality = 0;
	t2DDesc.MiscFlags = 0;

	Log("Init done!");
	LogTextureCompressionSupportInfo();
}
/*
Destructor
*/
HQTextureManagerD3D11::~HQTextureManagerD3D11()
{
	Log("Released!");
}


//translate from UAV texture format to DXGI format
DXGI_FORMAT HQTextureManagerD3D11::GetD3DFormat(HQTextureUAVFormat format)
{
	return HQUAVTextureResourceD3D11::GetD3DViewFormat(format);
}

/*-------create new texture object---------*/
HQBaseTexture * HQTextureManagerD3D11::CreateNewTextureObject(HQTextureType type)
{
	if (type == HQ_TEXTURE_BUFFER)
		return HQ_NEW HQTextureBufferD3D11();
	return HQ_NEW HQTextureD3D11(type);
}
/*-------set texture ---------------------*/
HQReturnVal HQTextureManagerD3D11::SetTexture(HQShaderType shaderStage, hq_uint32 resourceSlot, HQTexture* textureID)
{
	//TO DO: unbind from all other slots in the same shader stage

	HQSharedPtr<HQBaseTexture> pTexture = this->textures.GetItemPointer(textureID);
	HQTextureD3D11* pTextureD3D11 = (HQTextureD3D11*)pTexture.GetRawPointer();
	hquint32 unifySlot = shaderStage | resourceSlot;
	TextureSlot *pTextureSlot;
	HQTextureD3D11* pCurrentTextureD3D11;

	//first make sure this texture will not be bound to any UAV slots
	this->UnbindTextureFromAllUAVSlots(pTexture);

	switch (shaderStage)
	{
	case HQ_VERTEX_SHADER:
#if defined _DEBUG || defined DEBUG
		if (resourceSlot >= this->pMasterDevice->GetCaps().maxVertexTextures)
		{
			Log("SetTexture() Error : texture slot=%u is out of range!", resourceSlot);
			return HQ_FAILED;
		}
#endif
		pTextureSlot = this->textureSlots[0] + resourceSlot;
		pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureSlot->pTexture.GetRawPointer();
		if (pTextureD3D11 != pCurrentTextureD3D11)
		{
			if (pCurrentTextureD3D11 != NULL)
			{
				//remove the link between old texture and texture slot
				pCurrentTextureD3D11->boundSlots.RemoveAt(pTextureSlot->textureLink);
			}

			ID3D11ShaderResourceView *pSRV;
			if (pTextureD3D11 != NULL)
			{
				pSRV = ((HQTextureResourceD3D11 *)pTexture->pData)->pResourceView;
				//link the texture with this slot
				pTextureSlot->textureLink = pTextureD3D11->boundSlots.PushBack(unifySlot);
			}
			else
				pSRV = NULL  ;
			//pD3DContext->VSSetShaderResources(resourceSlot , 1 , &pSRV);
			pMasterDevice->SetVSResource(resourceSlot, pSRV);
			pTextureSlot->pTexture = pTexture;
		}
		break;
	case HQ_GEOMETRY_SHADER:
#if defined _DEBUG || defined DEBUG
		if (resourceSlot >= this->pMasterDevice->GetCaps().maxGeometryTextures)
		{
			Log("SetTexture() Error : texture slot=%u is out of range!", resourceSlot);
			return HQ_FAILED;
		}
#endif
		pTextureSlot = this->textureSlots[1] + resourceSlot;
		pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureSlot->pTexture.GetRawPointer();
		if (pTextureD3D11 != pCurrentTextureD3D11)
		{
			if (pCurrentTextureD3D11 != NULL)
			{
				//remove the link between old texture and texture slot
				pCurrentTextureD3D11->boundSlots.RemoveAt(pTextureSlot->textureLink);
			}

			ID3D11ShaderResourceView *pSRV;
			if (pTextureD3D11 != NULL)
			{
				pSRV = ((HQTextureResourceD3D11 *)pTexture->pData)->pResourceView;
				//link the texture with this slot
				pTextureSlot->textureLink = pTextureD3D11->boundSlots.PushBack(unifySlot);
			}
			else
				pSRV = NULL;
			//pD3DContext->GSSetShaderResources(resourceSlot, 1, &pSRV);
			pMasterDevice->SetGSResource(resourceSlot, pSRV);
			pTextureSlot->pTexture = pTexture;
		}
		break;
	case HQ_PIXEL_SHADER:
#if defined _DEBUG || defined DEBUG
		if (resourceSlot >= this->pMasterDevice->GetCaps().maxPixelTextures)
		{
			Log("SetTexture() Error : texture slot=%u is out of range!", resourceSlot);
			return HQ_FAILED;
		}
#endif
		pTextureSlot = this->textureSlots[2] + resourceSlot;
		pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureSlot->pTexture.GetRawPointer();
		if (pTextureD3D11 != pCurrentTextureD3D11)
		{
			if (pCurrentTextureD3D11 != NULL)
			{
				//remove the link between old texture and texture slot
				pCurrentTextureD3D11->boundSlots.RemoveAt(pTextureSlot->textureLink);
			}

			ID3D11ShaderResourceView *pSRV;
			if (pTextureD3D11 != NULL)
			{
				pSRV = ((HQTextureResourceD3D11 *)pTexture->pData)->pResourceView;
				//link the texture with this slot
				pTextureSlot->textureLink = pTextureD3D11->boundSlots.PushBack(unifySlot);
			}
			else
				pSRV = NULL;
			//pD3DContext->PSSetShaderResources(resourceSlot, 1, &pSRV);
			pMasterDevice->SetPSResource(resourceSlot, pSRV);
			pTextureSlot->pTexture = pTexture;
		}
		break;
	case HQ_COMPUTE_SHADER:
#if defined _DEBUG || defined DEBUG
		if (resourceSlot >= this->pMasterDevice->GetCaps().maxComputeTextures)
		{
			Log("SetTexture() Error : texture slot=%u is out of range!", resourceSlot);
			return HQ_FAILED;
		}
#endif
		pTextureSlot = this->textureSlots[3] + resourceSlot;
		pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureSlot->pTexture.GetRawPointer();
		if (pTextureD3D11 != pCurrentTextureD3D11)
		{
			if (pCurrentTextureD3D11 != NULL)
			{
				//remove the link between old texture and texture slot
				pCurrentTextureD3D11->boundSlots.RemoveAt(pTextureSlot->textureLink);
			}

			ID3D11ShaderResourceView *pSRV;
			if (pTextureD3D11 != NULL)
			{
				pSRV = ((HQTextureResourceD3D11 *)pTexture->pData)->pResourceView;
				//link the texture with this slot
				pTextureSlot->textureLink = pTextureD3D11->boundSlots.PushBack(unifySlot);
			}
			else
				pSRV = NULL;
			//pD3DContext->CSSetShaderResources(resourceSlot, 1, &pSRV);
			pMasterDevice->SetCSResource(resourceSlot, pSRV);
			pTextureSlot->pTexture = pTexture;
		}
		break;
	default:
		return HQ_FAILED_INVALID_PARAMETER;
	}
	

	return HQ_OK;
}


HQReturnVal HQTextureManagerD3D11::SetTexture(hq_uint32 slot, HQTexture* textureID)
{
	hq_uint32 resourceSlot = slot & 0x0fffffff;

	hq_uint32 shaderStage = slot & 0xf0000000;

	HQReturnVal re = this->HQTextureManagerD3D11::SetTexture((HQShaderType)shaderStage, resourceSlot, textureID);

#if defined _DEBUG || defined DEBUG
	if (re == HQ_FAILED_INVALID_PARAMETER)
	{
		Log("Error : {slot} parameter passing to SetTexture() method didn't bitwise OR with HQShaderType enum value!");
	}
#endif

	return re;
}

HQReturnVal HQTextureManagerD3D11::SetTextureForPixelShader(hq_uint32 resourceSlot, HQTexture* textureID)
{
	//TO DO: unbind from all other slots in the same shader stage

	HQSharedPtr<HQBaseTexture> pTexture = this->textures.GetItemPointer(textureID);

#if defined _DEBUG || defined DEBUG
	if (resourceSlot >= this->pMasterDevice->GetCaps().maxPixelTextures)
	{
		Log("SetTextureForPixelShader() Error : texture slot=%u is out of range!", resourceSlot);
		return HQ_FAILED;
	}
#endif

	
	TextureSlot *pTextureSlot;
	HQTextureD3D11 * pCurrentTextureD3D11;

	pTextureSlot = this->textureSlots[2] + resourceSlot;
	HQTextureD3D11* pTextureD3D11 = (HQTextureD3D11*)pTexture.GetRawPointer();
	pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureSlot->pTexture.GetRawPointer();

	//first make sure this texture will not be bound to any UAV slots
	this->UnbindTextureFromAllUAVSlots(pTexture);

	if (pTextureD3D11 != pCurrentTextureD3D11)
	{
		if (pCurrentTextureD3D11 != NULL)
		{
			//remove the link between old texture and texture slot
			pCurrentTextureD3D11->boundSlots.RemoveAt(pTextureSlot->textureLink);
		}

		ID3D11ShaderResourceView *pSRV;
		if (pTextureD3D11 != NULL)
		{
			pSRV = ((HQTextureResourceD3D11 *)pTexture->pData)->pResourceView;
			//link the texture with this slot
			pTextureSlot->textureLink = pTextureD3D11->boundSlots.PushBack(HQ_PIXEL_SHADER | resourceSlot);
		}
		else
			pSRV = NULL;
		//pD3DContext->PSSetShaderResources(resourceSlot, 1, &pSRV);
		pMasterDevice->SetPSResource(resourceSlot, pSRV);
		pTextureSlot->pTexture = pTexture;
	}
	

	return HQ_OK;
}

HQReturnVal HQTextureManagerD3D11::SetTextureUAVForComputeShader(hq_uint32 uavSlot, HQTexture* textureID, hq_uint32 mipLevel, bool read)
{
#if defined _DEBUG || defined DEBUG
	if (uavSlot >= this->pMasterDevice->GetCaps().maxComputeUAVSlots)
	{
		Log("SetTextureUAVForComputeShader() Error : texture slot=%u is out of range!", uavSlot);
		return HQ_FAILED;
	}
#endif

	HQSharedPtr<HQBaseTexture> pTexture = this->textures.GetItemPointer(textureID);

	TextureSlot *pTextureUAVSlot;
	HQTextureD3D11* pCurrentTextureD3D11;
	HQTextureD3D11* pTextureD3D11 = (HQTextureD3D11*)pTexture.GetRawPointer();
	ID3D11UnorderedAccessView *pUAV = NULL;

	if (pTextureD3D11 != NULL)
	{
		//retrieve UAV
		HQUAVTextureResourceD3D11* pTextureUAVResD3D11 = (HQUAVTextureResourceD3D11 *)pTextureD3D11->pData;
		pUAV = pTextureUAVResD3D11->GetOrCreateNewUAV(this->pD3DDevice, pTextureD3D11->type, mipLevel, read);;

		//first unbind texture from all shader resource view slots
		this->UnbindTextureFromAllTextureSlots(pTexture);
	}

	//unbind any bound UAV buffer at the same slot
	static_cast<HQShaderManagerD3D11*>(this->pMasterDevice->GetShaderManager())->OnTextureBindToComputeShaderUAVSlot(uavSlot);


	
	pTextureUAVSlot = this->textureUAVSlots[1] + uavSlot;
	pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureUAVSlot->pTexture.GetRawPointer();
	
	if (pTextureD3D11 != pCurrentTextureD3D11)
	{
		if (pCurrentTextureD3D11 != NULL)
		{
			//remove the link between old texture and uav slot
			pCurrentTextureD3D11->uavBoundSlots.RemoveAt(pTextureUAVSlot->textureLink);
		}

		if (pTextureD3D11 != NULL)
		{
			//unbind from all other UAV slots
			this->UnbindTextureFromAllUAVSlots(pTexture);

			//link the texture with this slot
			pTextureUAVSlot->textureLink = pTextureD3D11->uavBoundSlots.PushBack(HQ_COMPUTE_SHADER | uavSlot);
			pTextureUAVSlot->pTexture = pTexture;//hold reference to texture
		}
	}

	//now set UAV
	this->pMasterDevice->SetUAVForComputeShader(uavSlot, pUAV);
	//pD3DContext->CSSetUnorderedAccessViews(uavSlot, 1, &pUAV, &uavInitialCount);



	return HQ_OK;
}

HQReturnVal HQTextureManagerD3D11::SetTextureUAVForGraphicsShader(hq_uint32 uavSlot, HQTexture* textureID, hq_uint32 mipLevel, bool read)
{
#if defined _DEBUG || defined DEBUG
	if (uavSlot >= this->pMasterDevice->GetCaps().maxPixelUAVSlots)
	{
		Log("SetTextureUAVForGraphicsShader() Error : texture slot=%u is out of range!", uavSlot);
		return HQ_FAILED;
	}
#endif
	HQRenderTargetManagerD3D11* pRenderTargetMan = static_cast<HQRenderTargetManagerD3D11*>(this->pMasterDevice->GetRenderTargetManager());

	HQSharedPtr<HQBaseTexture> pTexture = this->textures.GetItemPointer(textureID);

	TextureSlot *pTextureUAVSlot;
	HQTextureD3D11* pCurrentTextureD3D11;
	HQTextureD3D11* pTextureD3D11 = (HQTextureD3D11*)pTexture.GetRawPointer();
	ID3D11UnorderedAccessView *pUAV = NULL;

	UINT uavInitialCount = -1;

	if (pTextureD3D11 != NULL)
	{
		//retrieve UAV
		HQUAVTextureResourceD3D11* pTextureUAVResD3D11 = (HQUAVTextureResourceD3D11 *)pTextureD3D11->pData;
		pUAV = pTextureUAVResD3D11->GetOrCreateNewUAV(this->pD3DDevice, pTextureD3D11->type, mipLevel, read);;

		//first unbind texture from all shader resource view slots
		this->UnbindTextureFromAllTextureSlots(pTexture);
	}

	//unbind any bound UAV buffer at the same slot
	static_cast<HQShaderManagerD3D11*>(this->pMasterDevice->GetShaderManager())->OnTextureBindToGraphicsShaderUAVSlot(uavSlot);



	pTextureUAVSlot = this->textureUAVSlots[0] + uavSlot;
	pCurrentTextureD3D11 = (HQTextureD3D11*)pTextureUAVSlot->pTexture.GetRawPointer();

	if (pTextureD3D11 != pCurrentTextureD3D11)
	{
		if (pCurrentTextureD3D11 != NULL)
		{
			//remove the link between old texture and uav slot
			pCurrentTextureD3D11->uavBoundSlots.RemoveAt(pTextureUAVSlot->textureLink);
		}

		if (pTextureD3D11 != NULL)
		{
			//unbind from all other UAV slots
			this->UnbindTextureFromAllUAVSlots(pTexture);

			//link the texture with this slot
			pTextureUAVSlot->textureLink = pTextureD3D11->uavBoundSlots.PushBack(HQ_PIXEL_SHADER | uavSlot);
			pTextureUAVSlot->pTexture = pTexture;//hold reference to texture
		}
	}

	//now set UAV using render target manager, since D3D11 requires render targets and UAVs to be set at the same time
	return pRenderTargetMan->SetUAVForGraphicsShader(uavSlot, pUAV);
}


void HQTextureManagerD3D11::OnBufferBindToComputeShaderUAVSlot(hquint32 slot)
{
	TextureSlot * pTextureUAVSlot = this->textureUAVSlots[1] + slot;
	if (pTextureUAVSlot->pTexture != NULL)
	{
		HQTextureD3D11* pTextureD3D11 = (HQTextureD3D11*)pTextureUAVSlot->pTexture.GetRawPointer();
		//remove the link between this texture and the UAV slot, because we are switching to a buffer
		pTextureD3D11->uavBoundSlots.RemoveAt(pTextureUAVSlot->textureLink);
		pTextureUAVSlot->pTexture.ToNull();
	}
}

void HQTextureManagerD3D11::OnBufferBindToGraphicsShaderUAVSlot(hquint32 slot)
{
	TextureSlot * pTextureUAVSlot = this->textureUAVSlots[0] + slot;
	if (pTextureUAVSlot->pTexture != NULL)
	{
		HQTextureD3D11* pTextureD3D11 = (HQTextureD3D11*)pTextureUAVSlot->pTexture.GetRawPointer();
		//remove the link between this texture and the UAV slot, because we are switching to a buffer
		pTextureD3D11->uavBoundSlots.RemoveAt(pTextureUAVSlot->textureLink);
		pTextureUAVSlot->pTexture.ToNull();
	}
}

void HQTextureManagerD3D11::UnbindTextureFromAllTextureSlots(const HQSharedPtr<HQBaseTexture> &pTexture)
{
	if (pTexture == NULL)
		return;
	HQTextureD3D11 *pTextureD3D11 = (HQTextureD3D11*)pTexture.GetRawPointer();

	ID3D11ShaderResourceView *nullView = NULL;

	HQTextureD3D11::SlotList::Iterator ite;
	for (pTextureD3D11->boundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
	{
		hquint32 slot = *ite;
		hq_uint32 resourceSlot = slot & 0x0fffffff;
		hq_uint32 shaderStage = slot & 0xf0000000;

		TextureSlot *pTextureSlot = NULL;

		switch (shaderStage)
		{
		case HQ_VERTEX_SHADER:
			pTextureSlot = this->textureSlots[0] + resourceSlot;
			pMasterDevice->SetVSResource(resourceSlot, NULL);
			//pD3DContext->VSSetShaderResources(resourceSlot, 1, &nullView);
			break;
		case HQ_GEOMETRY_SHADER:
			pTextureSlot = this->textureSlots[1] + resourceSlot;
			pMasterDevice->SetGSResource(resourceSlot, NULL);
			//pD3DContext->GSSetShaderResources(resourceSlot, 1, &nullView);
			break;
		case HQ_PIXEL_SHADER:
			pTextureSlot = this->textureSlots[2] + resourceSlot;
			pMasterDevice->SetPSResource(resourceSlot, NULL);
			//pD3DContext->PSSetShaderResources(resourceSlot, 1, &nullView);
			break;
		case HQ_COMPUTE_SHADER:
			pTextureSlot = this->textureSlots[3] + resourceSlot;
			pMasterDevice->SetCSResource(resourceSlot, NULL);
			//pD3DContext->CSSetShaderResources(resourceSlot, 1, &nullView);
			break;
		}//switch (shaderStage)

		//remove the link between this texture and the slot
		pTextureD3D11->boundSlots.RemoveAt(pTextureSlot->textureLink);
		pTextureSlot->pTexture = HQSharedPtr<HQBaseTexture>::null;
	}//for (pTextureD3D11->boundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
}

void HQTextureManagerD3D11::UnbindTextureFromAllUAVSlots(const HQSharedPtr<HQBaseTexture> &pTexture)
{
	if (pTexture == NULL)
		return;
	HQRenderTargetManagerD3D11* pRenderTargetMan = static_cast<HQRenderTargetManagerD3D11*>(this->pMasterDevice->GetRenderTargetManager());

	HQTextureD3D11 *pTextureD3D11 = (HQTextureD3D11*)pTexture.GetRawPointer();

	ID3D11UnorderedAccessView *nullView = NULL;
	UINT uavInitialCount = -1;

	HQTextureD3D11::SlotList::Iterator ite;
	for (pTextureD3D11->uavBoundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
	{
		hquint32 slot = *ite;
		hq_uint32 uavSlot = slot & 0x0fffffff;
		hq_uint32 shaderStage = slot & 0xf0000000;

		TextureSlot *pTextureUAVSlot = NULL;

		switch (shaderStage)
		{
		case HQ_COMPUTE_SHADER:
			pTextureUAVSlot = this->textureUAVSlots[1] + uavSlot;
			this->pMasterDevice->SetUAVForComputeShader(uavSlot, NULL);
			//pD3DContext->CSSetUnorderedAccessViews(uavSlot, 1, &nullView, &uavInitialCount);
			break;
		case HQ_PIXEL_SHADER:
			pTextureUAVSlot = this->textureUAVSlots[0] + uavSlot;
			//now unset UAV using render target manager, since D3D11 requires render targets and UAVs to be set at the same time
			pRenderTargetMan->SetUAVForGraphicsShader(uavSlot, NULL);
		default:
			break;//TO DO
		}//switch (shaderStage)

		//remove the link between this texture and the UAV slot
		pTextureD3D11->uavBoundSlots.RemoveAt(pTextureUAVSlot->textureLink);
		pTextureUAVSlot->pTexture.ToNull();
	}//for (pTextureD3D11->uavBoundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
}

/*
Load texture from file
*/
HQReturnVal HQTextureManagerD3D11::LoadTextureFromStream(HQDataReaderStream* dataStream, HQBaseTexture * pTex)
{
	if(!pD3DDevice)
		return HQ_FAILED;

	const char nullStreamName[] = "";
	const char *streamName = dataStream->GetName() != NULL? dataStream->GetName(): nullStreamName;

	//các thông tin cơ sở
	hq_uint32 w,h;//width,height
	short bpp;//bits per pixel
	SurfaceFormat format;//định dạng
	SurfaceComplexity complex;//độ phức tạp
	ImgOrigin origin;//vị trí pixel đầu 

	//load bitmap file
	int result; 

	
	result = bitmap.LoadFromStream(dataStream);
	
	char errBuffer[128];

	if(result!=IMG_OK)
	{
		if(result == IMG_FAIL_BAD_FORMAT)
		{
			//TO DO
		}
		else if (result == IMG_FAIL_NOT_ENOUGH_CUBE_FACES)
		{
			Log("Load cube texture from stream %s error : File doesn't have enough 6 cube faces!",streamName);
			return HQ_FAILED_NOT_ENOUGH_CUBE_FACES;
		}
		bitmap.GetErrorDesc(result,errBuffer);
		Log("Load texture from stream %s error : %s",streamName,errBuffer);
		return HQ_FAILED;
	}
	w=bitmap.GetWidth();
	h=bitmap.GetHeight();
	bpp=bitmap.GetBits();
	format=bitmap.GetSurfaceFormat();
	origin=bitmap.GetPixelOrigin();
	complex=bitmap.GetSurfaceComplex();

	if(complex.dwComplexFlags & SURFACE_COMPLEX_VOLUME)//it's volume texture
	{
		Log("Load texture from stream %s error : can't load volume texture", streamName);
		return HQ_FAILED;
	}

	if (format == FMT_ETC1)
	{
		Log("ETC compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		
		if(bitmap.DeCompressETC(true)==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing ETC compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (not support ETC)
	
	hq_uint32 nMips = 1;//số lượng mipmap level
	
	if(generateMipmap)
	{
		//full range mipmap
		if(IsPowerOfTwo(max(w,h),&nMips))//nMips=1+floor(log2(max(w,h)))
			nMips++;
	
		if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}
	}
	
	//đưa pixel đầu tiên lên vị trí góc trên bên trái
	result = bitmap.SetPixelOrigin(ORIGIN_TOP_LEFT);
	
	if(result!=IMG_OK)
	{
		if (bitmap.IsCompressed())
		{
			Log ("Warning : Couldn't flip texture's image origin to top left corner because it's compressed!");
		}
		else
		{
			bitmap.GetErrorDesc(result,errBuffer);
			Log("Error when trying to flip texture : %s",errBuffer);
		}
	}
	
	if(InitTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}

/*
Load cube texture from 6 files
*/
HQReturnVal HQTextureManagerD3D11::LoadCubeTextureFromStreams(HQDataReaderStream* dataStreams[6] , HQBaseTexture * pTex)
{
	if(!pD3DDevice)
		return HQ_FAILED;


	//các thông tin cơ sở
	hq_uint32 w,h;//width,height
	short bpp;//bits per pixel
	SurfaceFormat format;//định dạng
	SurfaceComplexity complex;//độ phức tạp
	ImgOrigin origin;//vị trí pixel đầu 

	//load bitmap files
	int result;

	result=bitmap.LoadCubeFaces(dataStreams ,ORIGIN_TOP_LEFT, this->generateMipmap);
	
	char errBuffer[128];
	
	if(result == IMG_FAIL_CANT_GENERATE_MIPMAPS)
	{
		Log("Load cube texture from streams warning : can't generate mipmaps");
		this->generateMipmap = false;
	}
	else if(result!=IMG_OK)
	{
		bitmap.GetErrorDesc(result,errBuffer);
		Log("Load cube texture from streams error : %s",errBuffer);
		return HQ_FAILED;
	}
	w=bitmap.GetWidth();
	h=bitmap.GetHeight();
	bpp=bitmap.GetBits();
	format=bitmap.GetSurfaceFormat();
	origin=bitmap.GetPixelOrigin();
	complex=bitmap.GetSurfaceComplex();
	
	if (format == FMT_ETC1)
	{
		Log("ETC compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		
		if(bitmap.DeCompressETC(true)==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing ETC compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (not support ETC)
	
	hq_uint32 nMips = 1;//số lượng mipmap level
	
	if(generateMipmap)
	{
		if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}
	}
	
	if(InitTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}

HQReturnVal HQTextureManagerD3D11::InitSingleColorTexture(HQBaseTexture *pTex,HQColorui color)
{
	t2DDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	t2DDesc.Width = 1;
	t2DDesc.Height =1;
	t2DDesc.MipLevels = 1;
	t2DDesc.MiscFlags = 0;
	t2DDesc.ArraySize = 1;
	
	D3D11_SUBRESOURCE_DATA initData;
	initData.pSysMem = &color;
	initData.SysMemPitch = 4;

	HQTextureResourceD3D11 *pT = (HQTextureResourceD3D11*) pTex->pData;

	if(FAILED(pD3DDevice->CreateTexture2D(&t2DDesc,&initData , (ID3D11Texture2D**)&pT->pTexture)) ||
		this->CreateShaderResourceView(pTex)!= HQ_OK)
	{
		Log("Failed to create new texture that has single color %u",color);
		return HQ_FAILED;
	}

	return HQ_OK;
}
/*
create texture object from pixel data
*/
HQReturnVal HQTextureManagerD3D11::InitTexture(bool changeAlpha,hq_uint32 numMipmaps,HQBaseTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	if(bitmap.IsCompressed() && complex.nMipMap < 2 && numMipmaps > 1 )//dữ liệu ảnh dạng nén và chỉ có 1 mipmap level sẵn có,trong khi ta cần nhiều hơn 1 mipmap level
	{
		if (bitmap.DeCompress(true) == IMG_FAIL_NOT_SUPPORTED)//nếu giải nén vẫn không được
			numMipmaps=1;//cho số mipmap level bằng 1
	}

	if(changeAlpha)//có thay đổi alpha
	{
		int result=IMG_OK;
		if (bitmap.IsCompressed())
			result = bitmap.DeCompress(true);

		if(result==IMG_OK)
		{
			switch(bitmap.GetSurfaceFormat())
			{
			case FMT_R5G6B5: 
				result=bitmap.RGB16ToRGBA(true);//=>A8B8G8R8
				break;
			case FMT_B5G6R5:
				result = bitmap.RGB16ToRGBA();//=>A8B8G8R8
				break;
			case FMT_R8G8B8: 
				result=bitmap.RGB24ToRGBA(true);//=>A8B8G8R8
				break;
			case FMT_B8G8R8:
				result=bitmap.RGB24ToRGBA();//=>A8B8G8R8
				break;
			case FMT_L8:
				result=bitmap.L8ToRGBA(true);
				break;
			}
			if(result==IMG_OK)
				this->ChangePixelData(pTex);
		}

	}//if (alpha)

	
	//chuyển format lần cuối
	switch(bitmap.GetSurfaceFormat())
	{
	case FMT_B8G8R8A8:
		bitmap.FlipRGB();//to R8G8B8A8
	case FMT_R8G8B8A8: 
		bitmap.FlipRGBA();//to A8B8G8R8
		break;
	case FMT_R5G6B5:
		bitmap.RGB16ToRGBA(true);//to A8B8G8R8
		break;
	case FMT_B5G6R5:
		bitmap.RGB16ToRGBA();//to A8B8G8R8
		break;
	case FMT_R8G8B8:
		bitmap.RGB24ToRGBA(true);//to A8B8G8R8
		break;
	case FMT_B8G8R8:
		bitmap.RGB24ToRGBA();//to A8B8G8R8
		break;
	case FMT_A8R8G8B8 :case  FMT_X8R8G8B8:
		bitmap.FlipRGB();//BGR->RGB (red ở byte trọng số nhỏ nhất)
		break;
	case FMT_L8:
		bitmap.L8ToRGBA(true);
		break;
	case FMT_A8L8:
		bitmap.AL16ToRGBA(true);
		break;
	}

	//định dạng của texture lấy tương ứng theo định dạng của pixel data
	format=bitmap.GetSurfaceFormat();
	if (bitmap.IsPVRTC())
	{
#if HQ_USE_PVR_TEX_LIB
		t2DDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
#else
		Log("Error : PVRTC texture is not supported!");
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
#endif
	}
	else
		t2DDesc.Format =GetTextureFmt(format);
	if(pTex->type == HQ_TEXTURE_CUBE)
	{
		t2DDesc.ArraySize = 6;
		if((complex.dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0 && 
			complex.nMipMap < 2 && numMipmaps > 1 )//dữ liệu ảnh dạng cube map và chỉ có 1 mipmap level sẵn có,trong khi ta cần nhiều hơn 1 mipmap level
			numMipmaps = 1;
		t2DDesc.Width = bitmap.GetWidth();
		t2DDesc.Height = bitmap.GetWidth();
		t2DDesc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
	}
	else
	{
		t2DDesc.ArraySize = 1;
		t2DDesc.Width = bitmap.GetWidth();
		t2DDesc.Height = bitmap.GetHeight();
		t2DDesc.MiscFlags = 0;
	}
	
	t2DDesc.MipLevels = numMipmaps;

	//non power of 2 texture restriction
	hquint32 Exp;

	if(!IsPowerOfTwo(bitmap.GetWidth(),&Exp) || !IsPowerOfTwo(bitmap.GetHeight(),&Exp))
	{
		if (!this->pMasterDevice->IsNpotTextureFullySupported(pTex->type))//only 1 mipmap level is allowed
		{
			t2DDesc.MipLevels = 1;
			Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture ");
		}
	}
	
	HQTextureResourceD3D11 *pT = (HQTextureResourceD3D11*) pTex->pData;

	if(FAILED(pD3DDevice->CreateTexture2D(&t2DDesc , NULL , (ID3D11Texture2D**)&pT->pTexture)))
	{
		//thử lại với chỉ 1 mipmap level
		numMipmaps = 1;
		t2DDesc.MipLevels = numMipmaps;

		if(FAILED(pD3DDevice->CreateTexture2D(&t2DDesc , NULL , (ID3D11Texture2D**)&pT->pTexture)))
		{
			return HQ_FAILED;
		}
	}
	HQReturnVal re;
	switch(pTex->type)
	{
	case HQ_TEXTURE_2D:
		re = this->Init2DTexture(numMipmaps , pTex);
		break;
	case HQ_TEXTURE_CUBE:
		re = this->InitCubeTexture(numMipmaps , pTex);
		break;
	}
	if (HQFailed(re))
		return re;
	return this->CreateShaderResourceView(pTex);
}
HQReturnVal HQTextureManagerD3D11::Init2DTexture(hq_uint32 numMipmaps,HQBaseTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	unsigned long rowSize;//độ lớn 1 hàng của 1 mipmap level trong dữ liệu ảnh
	unsigned long lvlSize;
	hq_ubyte8 *pSource;//pixel data

	hq_uint32 w = bitmap.GetWidth();
	hq_uint32 h = bitmap.GetHeight();
	//tạo bộ mipmap

	/*
	D3D11_BOX box;
	box.back =1;
	box.front =0;
	box.top = 0;
	box.left = 0;
	*/
	HQTextureResourceD3D11 *pT = (HQTextureResourceD3D11*) pTex->pData;
	
#if HQ_USE_PVR_TEX_LIB
	// declare an empty texture to decompress pvr texture into 
	CPVRTexture	sDecompressedTexture;
#endif
	if (bitmap.IsPVRTC())//decompress pvrtc
	{
#if HQ_USE_PVR_TEX_LIB
		Log("PVRTC compressed texture is not supported!Now trying to decompress it!");

		PVRTRY 
		{ 
			// get the utilities instance 
			PVRTextureUtilities sPVRU = PVRTextureUtilities(); 

			/*----------original texture-----------*/
			//create pvr header
			PVR_Header header_str;
			bitmap.GetPVRHeader(header_str);

			PixelType pixelType;
			switch (bitmap.GetSurfaceFormat())
			{
			case FMT_PVRTC_RGBA_4BPP : case FMT_PVRTC_RGB_4BPP:
				pixelType = MGLPT_PVRTC4;
				break;
			case FMT_PVRTC_RGBA_2BPP : case FMT_PVRTC_RGB_2BPP:
				pixelType = MGLPT_PVRTC2;
				break;
			}

			CPVRTextureHeader header(header_str.dwWidth, header_str.dwHeight, header_str.dwMipMapCount, 
				header_str.dwNumSurfs, false, false, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_VOLUME) != 0,
				false , header_str.dwAlphaBitMask != 0, 
				false,//(header_str.dwpfFlags & 0x00010000) != 0, //flipped
				pixelType, 0.0f);
				
			//texture data
			CPVRTextureData textureData(bitmap.GetPixelData(), bitmap.GetImgSize());
			bitmap.ClearData();
			//original texture
			CPVRTexture sOriginalTexture(header, textureData); ; 

			// decompress the compressed texture
			sPVRU.DecompressPVR(sOriginalTexture,sDecompressedTexture); 
			
			bitmap.Set(NULL, w, h, 32, 0, FMT_A8B8G8R8, bitmap.GetPixelOrigin(), bitmap.GetSurfaceComplex());

			pSource = sDecompressedTexture.getData().getData();
		 
		} 
		PVRCATCH(myException) 
		{ 
			// handle any exceptions here 
			Log("Error when trying to decompress PVRT compressed data: %s",myException.what()); 
			return HQ_FAILED;
		}
#else
		Log("Error : PVRTC texture is not supported!");
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
#endif//#if HQ_USE_PVR_TEX_LIB
	}
	else
		pSource =bitmap.GetPixelData();
	for(hq_uint32 level=0;level<numMipmaps;++level)
	{
		if (level >= this->t2DDesc.MipLevels)
			break;//skip this level and the rest
		rowSize=bitmap.CalculateRowSize(w);
		lvlSize = bitmap.CalculateSize(w,h);
		
		
		pD3DContext->UpdateSubresource(pT->pTexture , level , NULL , pSource , rowSize ,0);
		

		if (w > 1 )
			w >>=1; //w/=2
		if (h > 1)
			h >>=1; //h/=2

		if(complex.nMipMap < 2 && numMipmaps > 1) //nếu trong dữ liệu ảnh chỉ có sẵn 1 mipmap level,tự tạo dữ liệu ảnh ở level thấp hơn bằng cách phóng nhỏ hình ảnh
		{
			bitmap.Scalei(w,h);
			pSource=bitmap.GetPixelData();
		}
		else 
			pSource += lvlSize;
	}//for (level)

	return HQ_OK;
}
HQReturnVal HQTextureManagerD3D11::InitCubeTexture(hq_uint32 numMipmaps,HQBaseTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	unsigned long rowSize;//độ lớn 1 hàng của 1 mipmap level trong dữ liệu ảnh
	unsigned long lvlSize;
	hq_ubyte8 *pSource;//pixel data
	hq_uint32 Width = bitmap.GetWidth();
	//tạo bộ mipmap

	/*
	D3D11_BOX box;
	box.back =1;
	box.front =0;
	box.top = 0;
	box.left = 0;
	*/
	HQTextureResourceD3D11 *pT = (HQTextureResourceD3D11*) pTex->pData;

#if HQ_USE_PVR_TEX_LIB
	// declare an empty texture to decompress pvr texture into 
	CPVRTexture	sDecompressedTexture;
#endif
	if (bitmap.IsPVRTC())//decompress pvrtc
	{
#if HQ_USE_PVR_TEX_LIB
		Log("PVRTC compressed texture is not supported!Now trying to decompress it!");

		PVRTRY 
		{ 
			// get the utilities instance 
			PVRTextureUtilities sPVRU = PVRTextureUtilities(); 

			/*----------original texture-----------*/
			//create pvr header
			PVR_Header header_str;
			bitmap.GetPVRHeader(header_str);

			PixelType pixelType;
			switch (bitmap.GetSurfaceFormat())
			{
			case FMT_PVRTC_RGBA_4BPP : case FMT_PVRTC_RGB_4BPP:
				pixelType = MGLPT_PVRTC4;
				break;
			case FMT_PVRTC_RGBA_2BPP : case FMT_PVRTC_RGB_2BPP:
				pixelType = MGLPT_PVRTC2;
				break;
			}

			CPVRTextureHeader header(header_str.dwWidth, header_str.dwHeight, header_str.dwMipMapCount, 
				header_str.dwNumSurfs, false, false, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_VOLUME) != 0,
				false , header_str.dwAlphaBitMask != 0, 
				false,//(header_str.dwpfFlags & 0x00010000) != 0, //flipped
				pixelType, 0.0f);
				
			//texture data
			CPVRTextureData textureData(bitmap.GetPixelData(), bitmap.GetImgSize());
			bitmap.ClearData();
			//original texture
			CPVRTexture sOriginalTexture(header, textureData); ; 

			// decompress the compressed texture
			sPVRU.DecompressPVR(sOriginalTexture,sDecompressedTexture); 
			
			bitmap.Set(NULL, Width, Width, 32, 0, FMT_A8B8G8R8, bitmap.GetPixelOrigin(), bitmap.GetSurfaceComplex());

			pSource = sDecompressedTexture.getData().getData();
		 
		} 
		PVRCATCH(myException) 
		{ 
			// handle any exceptions here 
			Log("Error when trying to decompress PVRT compressed data: %s",myException.what()); 
			return HQ_FAILED;
		}
#else
		Log("Error : PVRTC texture is not supported!");
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
#endif//#if HQ_USE_PVR_TEX_LIB
	}
	else
		pSource = bitmap.GetPixelData();
	for (int face = 0 ; face < 6 ; ++face)
	{
		hq_uint32 w = Width;
		for(hq_uint32 level=0;level<numMipmaps;++level)
		{
			bool skip = level >= this->t2DDesc.MipLevels;//skip this level and the rest

			rowSize=bitmap.CalculateRowSize(w);
			lvlSize = bitmap.CalculateSize(w,w);
			
			if (!skip)
				pD3DContext->UpdateSubresource(pT->pTexture , D3D11CalcSubresource(level , face , numMipmaps) ,
					NULL , pSource , rowSize ,0);

			if(w > 1) w >>=1; //w/=2

			pSource += lvlSize;
		}//for (level)
	}//for (face)
	return HQ_OK;
}


/*
set alphakey
*/
HQReturnVal HQTextureManagerD3D11::SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A)
{
	hq_ubyte8* pData=bitmap.GetPixelData();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	HQColorui color;

	if(pData==NULL)
		return HQ_FAILED;

	switch(format)
	{
	case FMT_A8R8G8B8:
		color=HQColoruiRGBA(R,G,B,A,CL_BGRA);
		break;
	case FMT_A8B8G8R8:
		color=HQColoruiRGBA(R,G,B,A,CL_RGBA);
		break;
	case FMT_A8L8:
		color=HQColoruiRGBA(R,0,0,A,CL_BGRA);
		break;
	default:
		return HQ_FAILED;
	}

	hq_ubyte8 *pCur=pData;
	hq_ubyte8 *pEnd=pData + bitmap.GetImgSize();

	hq_ushort16 pixelSize=bitmap.GetBits()/8;

	for(;pCur < pEnd; pCur+=pixelSize)//với từng pixel
	{
		if(format==FMT_A8L8)
		{
			if(*pCur==R)
				*((hq_ushort16*)pCur)=(hq_ushort16)(color >> 16);//(A,R,0,0) => (A,L)
		}
		else{
			HQColorui * pPixel=(HQColorui*)pCur;
			if(((*pPixel) & 0xffffff )== (color & 0xffffff))//cùng giá trị RGB
				*pPixel=color;
		}
	}

	return HQ_OK;
}

/*
set transparency
*/
HQReturnVal HQTextureManagerD3D11::SetTransparency(hq_float32 alpha)
{
	hq_ubyte8* pData=bitmap.GetPixelData();
	SurfaceFormat format=bitmap.GetSurfaceFormat();

	if(pData==NULL)
		return HQ_FAILED;

	hq_ubyte8 *pCur=pData;
	hq_ubyte8 *pEnd=pData + bitmap.GetImgSize();

	hq_ushort16 pixelSize=bitmap.GetBits()/8;
	hq_ubyte8 A=(hq_ubyte8)(((hq_uint32)(alpha*255)) & 0xff);//alpha range 0->255
	for(;pCur < pEnd; pCur+=pixelSize)//với từng pixel
	{
		switch(format)
		{
		case FMT_A8R8G8B8:case FMT_A8B8G8R8:
			if(pCur[3] > A)
				pCur[3]=A;
			break;
		case FMT_A8L8:
			if(pCur[1] > A )
				pCur[1]=A;
			break;
		default:
			return HQ_FAILED;
		}
	}
	return HQ_OK;
}


HQReturnVal HQTextureManagerD3D11::CreateShaderResourceView(HQBaseTexture * pTex)
{
	HQTextureResourceD3D11 * pT = (HQTextureResourceD3D11*)pTex->pData;
	D3D11_SHADER_RESOURCE_VIEW_DESC vDesc;
	vDesc.Format = this->t2DDesc.Format;
	switch(pTex->type)
	{
	case HQ_TEXTURE_2D: case HQ_TEXTURE_2D_UAV:
		vDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		vDesc.Texture2D.MipLevels = this->t2DDesc.MipLevels;
		vDesc.Texture2D.MostDetailedMip = 0;
		break;
	case HQ_TEXTURE_2D_ARRAY: case HQ_TEXTURE_2D_ARRAY_UAV:
		vDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
		vDesc.Texture2DArray.MipLevels = this->t2DDesc.MipLevels;
		vDesc.Texture2DArray.MostDetailedMip = 0;
		vDesc.Texture2DArray.ArraySize = this->t2DDesc.ArraySize;
		vDesc.Texture2DArray.FirstArraySlice = 0;
		break;
	case HQ_TEXTURE_CUBE:
		vDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
		vDesc.TextureCube.MipLevels = this->t2DDesc.MipLevels;
		vDesc.TextureCube.MostDetailedMip = 0;
		break;
	case HQ_TEXTURE_BUFFER:
		vDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		vDesc.Buffer.ElementOffset = 0;
		vDesc.Buffer.FirstElement = 0;
		vDesc.Buffer.ElementWidth = this->t2DDesc.Width;
		vDesc.Buffer.NumElements = this->t2DDesc.Width;
		break;
	}
	if(FAILED(pD3DDevice->CreateShaderResourceView(pT->pTexture ,&vDesc, &pT->pResourceView)))
	{
		pT->pResourceView = 0;
		return HQ_FAILED;
	}

	return HQ_OK;
}

HQReturnVal HQTextureManagerD3D11::InitTextureBuffer(HQBaseTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size , void *initData, bool isDynamic)
{
	HQTextureBufferD3D11 *pTexBufD3D = static_cast<HQTextureBufferD3D11*> (pTex);
	pTexBufD3D->isDynamic = isDynamic;
	pTexBufD3D->size = size;
	pTexBufD3D->pD3DContext = this->pD3DContext;
	pTexBufD3D->pLog = this;

	hq_uint32 texelSize;

	if (this->pMasterDevice->GetFeatureLevel() < D3D_FEATURE_LEVEL_10_0)
	{
		Log ("Error : Texture buffer is not supported");
		return HQ_FAILED;
	}

	t2DDesc.Format = GetTextureBufferFormat(format , texelSize);//store in <t2DDesc> for later use in CreateShaderResourceView()
	if (t2DDesc.Format == DXGI_FORMAT_UNKNOWN)
		return HQ_FAILED_FORMAT_NOT_SUPPORT;

	//again store buffer width in <t2DDesc> for later use in CreateShaderResourceView()
	t2DDesc.Width = size / texelSize;
	
	D3D11_BUFFER_DESC bd;
	bd.ByteWidth = size;
	bd.MiscFlags = 0;
	bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	if (isDynamic)
	{
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		bd.Usage = D3D11_USAGE_DYNAMIC;
	}
	else
	{
		bd.CPUAccessFlags = 0;
		bd.Usage = D3D11_USAGE_DEFAULT;
	}

	D3D11_SUBRESOURCE_DATA d3dinitData;
	d3dinitData.pSysMem = initData;
	
	HQTextureResourceD3D11 *pT = (HQTextureResourceD3D11*) pTex->pData;

	if(FAILED(pD3DDevice->CreateBuffer(&bd , (initData != NULL)? &d3dinitData : NULL ,(ID3D11Buffer **)&pT->pTexture)) )
	{
		return HQ_FAILED;
	}

	return this->CreateShaderResourceView(pTex);
}


HQReturnVal HQTextureManagerD3D11::InitTextureUAV(HQBaseTexture *pTex, HQTextureUAVFormat format, hquint32 width, hquint32 height, hquint32 depth, bool hasMipmaps)
{
	return this->InitTextureUAVEx(pTex, format, width, height, depth, hasMipmaps, false);
}

HQReturnVal HQTextureManagerD3D11::InitTextureUAVEx(HQBaseTexture *pTex, HQTextureUAVFormat format, hquint32 width, hquint32 height, hquint32 depth, bool hasMipmaps, bool renderTarget)
{
	if (this->pMasterDevice->IsUAVTextureFormatSupported(format, pTex->type, hasMipmaps) == false)
	{
		Log("Error : UAV Texture creation with format = %u is not supported", (hquint32)format);
		return HQ_FAILED;
	}

	hq_uint32 numMipmaps = 1;
	if (hasMipmaps)
		numMipmaps = HQBaseTextureManager::CalculateFullNumMipmaps(width, height);//full range mipmap level

	DXGI_FORMAT d3dformat = HQUAVTextureResourceD3D11::GetD3DResFormat(format);

	//texture desc
	this->t2DDesc.Format = d3dformat;
	this->t2DDesc.Width = width;
	this->t2DDesc.Height = height;
	this->t2DDesc.MipLevels = numMipmaps;
	this->t2DDesc.SampleDesc.Count = 1;
	this->t2DDesc.SampleDesc.Quality = 0;
	this->t2DDesc.Usage = D3D11_USAGE_DEFAULT;
	this->t2DDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	if (renderTarget)
		this->t2DDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
	this->t2DDesc.CPUAccessFlags = 0;
	this->t2DDesc.MiscFlags = 0;

	//create texture and UAV view
	HQUAVTextureResourceD3D11* pTexResD3D = (HQUAVTextureResourceD3D11 *)pTex->pData;
	pTexResD3D->hqFormat = format;
	HRESULT hr;
	switch (pTex->type)
	{
	case HQ_TEXTURE_2D_UAV: case HQ_TEXTURE_2D_ARRAY_UAV:
		//texture desc
		this->t2DDesc.ArraySize = depth;

		//create texture
		hr = pD3DDevice->CreateTexture2D(&this->t2DDesc, NULL, (ID3D11Texture2D**)&pTexResD3D->pTexture);
		if (!FAILED(hr))
		{
			//create first level UAV view
			if (pTexResD3D->GetOrCreateNewUAV(this->pD3DDevice, pTex->type, 0, false) == NULL)
			{
				return HQ_FAILED;
			}
		}
		break;
	default:
		//TO DO: other type of texture
		return HQ_FAILED;
	}

	if (FAILED(hr))
		return HQ_FAILED;

	//get view format
	this->t2DDesc.Format = HQUAVTextureResourceD3D11::GetD3DViewFormat(format);

	return this->CreateShaderResourceView(pTex);
}


HQTextureCompressionSupport HQTextureManagerD3D11::IsCompressionSupported(HQTextureType textureType,HQTextureCompressionFormat type)
{
	switch (type)
	{
	case HQ_TC_S3TC_DTX1:
	case HQ_TC_S3TC_DXT3:
	case HQ_TC_S3TC_DXT5:
		return HQ_TCS_ALL;
	case HQ_TC_ETC1: 
		return HQ_TCS_SW;
	case HQ_TC_PVRTC_RGB_2BPP :
	case HQ_TC_PVRTC_RGB_4BPP: 
	case HQ_TC_PVRTC_RGBA_2BPP:
	case HQ_TC_PVRTC_RGBA_4BPP:
#if HQ_USE_PVR_TEX_LIB
		return HQ_TCS_SW;
#endif
	default:
		return HQ_TCS_NONE;
	}
}

HQBaseRawPixelBuffer* HQTextureManagerD3D11::CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height)
{
	switch (intendedFormat)
	{
	case HQ_RPFMT_R32_FLOAT: case HQ_RPFMT_R32G32_FLOAT: case HQ_RPFMT_R32G32B32A32_FLOAT:
		return HQ_NEW HQBaseRawPixelBuffer(intendedFormat, width, height);
	default:
		return HQ_NEW HQBaseRawPixelBuffer(HQ_RPFMT_R8G8B8A8, width, height);//only 32 bit pixel buffer is supported
	}
}

HQReturnVal HQTextureManagerD3D11::InitTexture(HQBaseTexture *pTex, const HQBaseRawPixelBuffer* color)
{

	color->MakeWrapperBitmap(bitmap);

	hquint32 w=bitmap.GetWidth();
	hquint32 h=bitmap.GetHeight();
	hquint32 bpp=bitmap.GetBits();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	ImgOrigin origin=bitmap.GetPixelOrigin();
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();

	hq_uint32 nMips = 1;//số lượng mipmap level

	if(generateMipmap)
	{
		//full range mipmap
		if(IsPowerOfTwo(max(w,h),&nMips))//nMips=1+floor(log2(max(w,h)))
			nMips++;
	
		if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}
	}

	
	if(InitTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}
