/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQDeviceD3D11.h"

#define HQ_UAV_SLOTS_CHANGED 0x1
#define HQ_KEEP_RTV 0

/*-------------------------*/

struct HQDepthStencilBufferD3D : public HQBaseDepthStencilBufferView
{
	HQDepthStencilBufferD3D(ID3D11Device * pD3DDevice , hq_uint32 width ,hq_uint32 height ,
							HQMultiSampleType multiSampleType,
							DXGI_FORMAT format)
			: HQBaseDepthStencilBufferView(width , height , 
									  multiSampleType)
	{
		this->format = format;
		this->pD3DDevice = pD3DDevice;
	}
	~HQDepthStencilBufferD3D()
	{
		SafeRelease(pDSView);
		SafeRelease(pDSTexture);
	}
	void *GetData()
	{
		return pDSView;
	}

	HQReturnVal Init()
	{
		UINT multisampleQuality;
		UINT sampleCount = (this->multiSampleType > 0 )?(UINT)this->multiSampleType : 1;
		pD3DDevice->CheckMultisampleQualityLevels(this->format ,
			sampleCount ,&multisampleQuality);

		D3D11_TEXTURE2D_DESC depthStencilDesc;
		depthStencilDesc.Width = this->width;
		depthStencilDesc.Height = this->height;
		depthStencilDesc.MipLevels = 1;
		depthStencilDesc.ArraySize = 1;
		depthStencilDesc.Format = this->format;
		depthStencilDesc.SampleDesc.Count = sampleCount;
		depthStencilDesc.SampleDesc.Quality = multisampleQuality - 1;
		depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
		depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		depthStencilDesc.CPUAccessFlags = 0;
		depthStencilDesc.MiscFlags = 0;
		
		if (FAILED(pD3DDevice->CreateTexture2D(&depthStencilDesc , 0 , &pDSTexture)))
		{
			this->pDSTexture = NULL;
			return HQ_FAILED;
		}

		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Flags = 0;
		dsvDesc.Format = this->format;
		if(sampleCount > 1)
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
		else
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		dsvDesc.Texture2D.MipSlice = 0;

		if(FAILED(pD3DDevice->CreateDepthStencilView(pDSTexture , &dsvDesc ,&pDSView)))
		{
			SafeRelease(pDSTexture);
			this->pDSView = NULL;
			return HQ_FAILED;
		}

		return HQ_OK;
	}

	DXGI_FORMAT format;
	ID3D11Texture2D *pDSTexture;
	ID3D11DepthStencilView* pDSView;
	ID3D11Device* pD3DDevice;
};

/*-----------------------------------------*/
struct HQRenderTargetTextureD3D : public HQBaseRenderTargetTexture
{
	HQRenderTargetTextureD3D(ID3D11Device* pD3DDevice,
							hq_uint32 width, hq_uint32 height, hq_uint32 arraySize,
							HQMultiSampleType multiSampleType,
							DXGI_FORMAT format , hq_uint32 numMipmaps,
							HQSharedPtr<HQBaseTexture> pTex)
		: HQBaseRenderTargetTexture(width , height ,
							multiSampleType, numMipmaps,
							pTex)
	{
		this->format = format;
		this->pD3DDevice = pD3DDevice;
		switch (pTex->type)
		{
		case HQ_TEXTURE_2D: case HQ_TEXTURE_2D_UAV:
			this->ppRTView = HQ_NEW ID3D11RenderTargetView *[1];
			this->ppRTView[0] = NULL;
			this->numViews = 1;
			break;
		case HQ_TEXTURE_CUBE:
			this->numViews = 6;
			this->ppRTView = HQ_NEW ID3D11RenderTargetView *[6];
			for(hq_uint32 i = 0 ; i < 6 ; ++i)
				this->ppRTView[i] = NULL;
			break;
		case HQ_TEXTURE_2D_ARRAY: case HQ_TEXTURE_2D_ARRAY_UAV:
			this->numViews = arraySize;
			this->ppRTView = HQ_NEW ID3D11RenderTargetView *[this->numViews];
			for (hq_uint32 i = 0; i < this->numViews; ++i)
				this->ppRTView[i] = NULL;
			break;
		}

	}
	~HQRenderTargetTextureD3D()
	{
		if (ppRTView != NULL)
		{
			for (hq_uint32 i = 0; i < this->numViews; ++i)
				SafeRelease(this->ppRTView[i]);

			delete[] ppRTView;
		}
	}

	void *GetData()
	{
		return ppRTView;
	}

	HQReturnVal Init()
	{
		HQTextureResourceD3D11* pTex = (HQTextureResourceD3D11 *)pTexture->pData;

		//this method will create texture and render target view
		//Note: future improvement should move texture creation to texture manager
		if (pTex->pTexture == NULL)//make sure texture resource hasn't been created
		{
			HRESULT hr;
			/*------get multisample quality------------*/
			UINT multisampleQuality;
			UINT sampleCount = (this->multiSampleType > 0) ? (UINT)this->multiSampleType : 1;
			pD3DDevice->CheckMultisampleQualityLevels(this->format,
				sampleCount, &multisampleQuality);

			D3D11_TEXTURE2D_DESC rtDesc;
			D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

			//texture desc
			rtDesc.Width = this->width;
			rtDesc.Height = this->height;
			rtDesc.MipLevels = this->numMipmaps;
			rtDesc.Format = this->format;
			rtDesc.SampleDesc.Count = sampleCount;
			rtDesc.SampleDesc.Quality = multisampleQuality - 1;
			rtDesc.Usage = D3D11_USAGE_DEFAULT;
			rtDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
			rtDesc.CPUAccessFlags = 0;
			if (this->numMipmaps != 1)
				rtDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
			else
				rtDesc.MiscFlags = 0;

			//resource view desc
			srvDesc.Format = this->format;

			switch (this->pTexture->type)
			{
			case HQ_TEXTURE_2D:
			{
				//texture desc
				rtDesc.ArraySize = 1;
				//create texture
				hr = pD3DDevice->CreateTexture2D(&rtDesc, NULL, (ID3D11Texture2D**)&pTex->pTexture);

				if (FAILED(hr))
				{
					pTex->pTexture = NULL;
					pTex->pResourceView = NULL;
					break;
				}

				if (sampleCount > 1)//multisample texture
				{
					//resource view desc
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
				}
				else
				{
					//resource view desc
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
					srvDesc.Texture2D.MipLevels = this->numMipmaps;
					srvDesc.Texture2D.MostDetailedMip = 0;
				}
				//create shader resource view
				hr = pD3DDevice->CreateShaderResourceView(pTex->pTexture, &srvDesc, &pTex->pResourceView);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					pTex->pResourceView = NULL;
					break;
				}
			}
				break;
			case HQ_TEXTURE_2D_ARRAY: case HQ_TEXTURE_2D_ARRAY_UAV:
			{
				//texture desc
				rtDesc.ArraySize = this->numViews;
				//create texture
				hr = pD3DDevice->CreateTexture2D(&rtDesc, NULL, (ID3D11Texture2D**)&pTex->pTexture);

				if (FAILED(hr))
				{
					pTex->pTexture = NULL;
					pTex->pResourceView = NULL;
					break;
				}

				if (sampleCount > 1)//multisample texture
				{
					//resource view desc
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
				}
				else
				{
					//resource view desc
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
					srvDesc.Texture2DArray.ArraySize = rtDesc.ArraySize;
					srvDesc.Texture2DArray.FirstArraySlice = 0;
					srvDesc.Texture2DArray.MipLevels = this->numMipmaps;
					srvDesc.Texture2DArray.MostDetailedMip = 0;
				}
				//create shader resource view
				hr = pD3DDevice->CreateShaderResourceView(pTex->pTexture, &srvDesc, &pTex->pResourceView);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					pTex->pResourceView = NULL;
					break;
				}
			}
				break;
			case HQ_TEXTURE_CUBE:
			{
				//texture desc
				rtDesc.MiscFlags |= D3D11_RESOURCE_MISC_TEXTURECUBE;
				rtDesc.ArraySize = 6;

				//create texture
				hr = pD3DDevice->CreateTexture2D(&rtDesc, NULL, (ID3D11Texture2D**)&pTex->pTexture);

				if (FAILED(hr))
				{
					pTex->pTexture = NULL;
					pTex->pResourceView = NULL;
					break;
				}

				//resource view desc
				srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
				srvDesc.TextureCube.MipLevels = this->numMipmaps;
				srvDesc.TextureCube.MostDetailedMip = 0;

				//create shader resource view
				hr = pD3DDevice->CreateShaderResourceView(pTex->pTexture, &srvDesc, &pTex->pResourceView);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					pTex->pResourceView = NULL;
					break;
				}
			}
				break;
			}//switch texture's type

			if (FAILED(hr))
				return HQ_FAILED;
		}//if (pTex->pTexture == NULL)

		return this->InitRenderTargetView();
	}

	HQReturnVal InitRenderTargetView()
	{
		//this method will create only render target view
		HRESULT hr;

		D3D11_TEXTURE2D_DESC rtDesc;
		D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;

		//render target view desc
		rtvDesc.Format = this->format;

		HQTextureResourceD3D11* pTex = (HQTextureResourceD3D11 *)pTexture->pData;
		switch (this->pTexture->type)
		{
		case HQ_TEXTURE_2D: case HQ_TEXTURE_2D_UAV:
		{
			//texture desc
			((ID3D11Texture2D*)pTex->pTexture)->GetDesc(&rtDesc);

			if (rtDesc.SampleDesc.Count > 1)//multisample texture
			{
				//render target view desc
				rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
			}
			else
			{
				//render target view desc
				rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
				rtvDesc.Texture2D.MipSlice = 0;
			}
			//create render target view
			hr = pD3DDevice->CreateRenderTargetView(pTex->pTexture, &rtvDesc, &this->ppRTView[0]);
			if (FAILED(hr))
			{
				SafeRelease(pTex->pTexture);
				SafeRelease(pTex->pResourceView);
				break;
			}

		}
			break;
		case HQ_TEXTURE_CUBE: case HQ_TEXTURE_2D_ARRAY: case HQ_TEXTURE_2D_ARRAY_UAV:
		{
			//texture desc
			((ID3D11Texture2D*)pTex->pTexture)->GetDesc(&rtDesc);

#if 1
			//render target view desc
			rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
			rtvDesc.Texture2DArray.ArraySize = 1;
			rtvDesc.Texture2DArray.MipSlice = 0;


			//create render target views for each array slices
			for (hquint32 i = 0; i < this->numViews; ++i)
			{
				rtvDesc.Texture2DArray.FirstArraySlice = i;
				hr = pD3DDevice->CreateRenderTargetView(pTex->pTexture, &rtvDesc, &this->ppRTView[i]);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					SafeRelease(pTex->pResourceView);
					break;
				}
			}
#else
			/*-----------View as 2D texture----------------*/
			//render target view desc
			rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;

			//create render target views for each array slices
			for (hquint32 i = 0; i < this->numViews; ++i)
			{
				rtvDesc.Texture2D.MipSlice = D3D11CalcSubresource(0, i, rtDesc.MipLevels);
				hr = pD3DDevice->CreateRenderTargetView(pTex->pTexture, &rtvDesc, &this->ppRTView[i]);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					SafeRelease(pTex->pResourceView);
					break;
				}
			}
#endif
		}
			break;
		}//switch texture's type

		if (FAILED(hr))
			return HQ_FAILED;
		return HQ_OK;

	}
	DXGI_FORMAT format;
	ID3D11RenderTargetView **ppRTView;//can be one pointer if texture is 2D or array of pointers if texture is cube/array
	hquint32 numViews;//number of elements in {ppRTView} array
	ID3D11Device* pD3DDevice;
};

/*------------------------------*/
struct HQRenderTargetGroupD3D11: public HQBaseRenderTargetGroup{
public:
	HQRenderTargetGroupD3D11(hquint32 numRenderTargets)
		: HQBaseRenderTargetGroup(numRenderTargets),
		  renderTargetViews(new ID3D11RenderTargetView *[numRenderTargets])
	{
	}
	~HQRenderTargetGroupD3D11(){
		delete[] renderTargetViews;
	}

	ID3D11RenderTargetView **renderTargetViews;//current active render target views
	ID3D11DepthStencilView *pDepthStencilView;//current active depth stencil view
};

/*------------HQRenderTargetManagerD3D11------------------*/
DXGI_FORMAT HQRenderTargetManagerD3D11::GetD3DFormat(HQRenderTargetFormat format)
{
	switch(format)
	{
	case HQ_RTFMT_R_FLOAT32:
		return DXGI_FORMAT_R32_FLOAT;
	case HQ_RTFMT_R_FLOAT16:
		return DXGI_FORMAT_R16_FLOAT;
	case HQ_RTFMT_RGBA_32:
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	case HQ_RTFMT_A_UINT8:
		return DXGI_FORMAT_A8_UNORM;
	case HQ_RTFMT_R_UINT8:
		return DXGI_FORMAT_R8_UNORM;
	case HQ_RTFMT_RGBA_FLOAT64:
		return DXGI_FORMAT_R16G16B16A16_FLOAT;
	case HQ_RTFMT_RG_FLOAT32:
		return DXGI_FORMAT_R16G16_FLOAT;
	case HQ_RTFMT_RGBA_FLOAT128:
		return DXGI_FORMAT_R32G32B32A32_FLOAT;
	case HQ_RTFMT_RG_FLOAT64:
		return DXGI_FORMAT_R32G32_FLOAT;
	}
	return DXGI_FORMAT_FORCE_UINT;
}

DXGI_FORMAT HQRenderTargetManagerD3D11::GetD3DFormat(HQDepthStencilFormat format)
{
	switch(format)
	{
	case HQ_DSFMT_DEPTH_16:
		return DXGI_FORMAT_D16_UNORM;
	case HQ_DSFMT_DEPTH_24:
		return DXGI_FORMAT_D24_UNORM_S8_UINT;
	case HQ_DSFMT_DEPTH_32:
		return DXGI_FORMAT_D32_FLOAT;
	case HQ_DSFMT_STENCIL_8:
		return DXGI_FORMAT_FORCE_UINT;
	case HQ_DSFMT_DEPTH_24_STENCIL_8:
		return DXGI_FORMAT_D24_UNORM_S8_UINT;
	}
	return DXGI_FORMAT_FORCE_UINT;
}

HQRenderTargetManagerD3D11::HQRenderTargetManagerD3D11(ID3D11Device * pD3DDevice, 
										 ID3D11DeviceContext *pD3DContext,
										 ID3D11RenderTargetView* pD3DBackBuffer,//default back buffer
										 ID3D11DepthStencilView* pD3DDSBuffer,//default depth stencil buffer
										 HQBaseTextureManager *pTexMan,
										 HQLogStream* logFileStream , bool flushLog)
: HQBaseRenderTargetManager(g_pD3DDev->GetCaps().maxMRTs , pTexMan , logFileStream , "D3D11 Render Target Manager :" , flushLog)
{
	this->pD3DDevice = pD3DDevice;
	this->pD3DContext = pD3DContext;
	this->pD3DBackBuffer = pD3DBackBuffer;
	this->pD3DDSBuffer = pD3DDSBuffer;
	
	this->renderTargetViews = &this->pD3DBackBuffer;
	this->pDepthStencilView = this->pD3DDSBuffer;

	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();

	for (hquint32 i = 0; i < D3D11_PS_CS_UAV_REGISTER_COUNT; ++i)
	{
		this->pUAVSlots[i] = NULL;
		this->UAVInitialCounts[i] = -1;
	}
	this->minUsedUAVSlot = D3D11_PS_CS_UAV_REGISTER_COUNT;
	this->maxUsedUAVSlot = -1;
	this->flags = 0;

	Log("Init done!");
}

HQRenderTargetManagerD3D11::~HQRenderTargetManagerD3D11()
{
	this->renderTargets.RemoveAll();
	this->depthStencilBuffers.RemoveAll();
	Log("Released!");
}





HQReturnVal HQRenderTargetManagerD3D11::CreateRenderTargetTexture(hq_uint32 width, hq_uint32 height, hq_uint32 arraySize, 
												bool hasMipmaps,
											   HQRenderTargetFormat format, HQMultiSampleType multisampleType, 
											   HQTextureType textureType, 
											   HQRenderTargetView **pRenderTargetID_Out, 
											   HQTexture **pTextureID_Out)
{
	HQTextureManagerD3D11* pTextureManagerD3D11 = static_cast<HQTextureManagerD3D11*>(this->pTextureManager);
	bool isTextureUAV = false;
	HQTextureUAVFormat uavFormat;
	switch (textureType)
	{
	case HQ_TEXTURE_2D:  case HQ_TEXTURE_2D_ARRAY:
		break;
	case HQ_TEXTURE_2D_UAV: case HQ_TEXTURE_2D_ARRAY_UAV:
		isTextureUAV = true;
		uavFormat = HQBaseTextureManager::GetTextureUAVFormat(format);
		break;
	case HQ_TEXTURE_CUBE:
		break;
	default:
		Log("CreateRenderTargetTexture() failed : unsuported texture type=%u!", (hquint32)textureType);
		return HQ_FAILED;
	}

	char str[256];
	if (!g_pD3DDev->IsRTTFormatSupported(format , textureType , hasMipmaps)
		|| (isTextureUAV && !g_pD3DDev->IsUAVTextureFormatSupported(uavFormat, textureType, hasMipmaps)))
	{
		HQBaseRenderTargetManager::GetString(format , str);
		if(!hasMipmaps)
			Log("CreateRenderTargetTexture() failed : creating render target texture with format = %s is not supported!" , str);
		else
			Log("CreateRenderTargetTexture() failed : creating render target texture with format = %s and full mipmap levels is not supported!" , str);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}

	if (!g_pD3DDev->IsRTTMultisampleTypeSupported(format , multisampleType , textureType))
	{
		HQBaseRenderTargetManager::GetString(multisampleType , str);
		Log("CreateRenderTargetTexture() failed : %s is not supported!" , str);
		return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT;
	}
	HQTexture* textureID = 0;
	HQSharedPtr<HQBaseTexture> pNewTex = pTextureManagerD3D11->AddEmptyTexture(textureType, &textureID);
	if (pNewTex == NULL)
		return HQ_FAILED_MEM_ALLOC;

	/*---------texture type specific config------------*/
	if (textureType == HQ_TEXTURE_CUBE)
	{
		height = width;
		if (multisampleType != HQ_MST_NONE)
		{
			Log("warning : create render target cube texture with multisample is not supported, texture will be created with no multisample");
			multisampleType = HQ_MST_NONE;
		}
	}
	else if (isTextureUAV)//unordered access supported texture
	{
		if (multisampleType != HQ_MST_NONE)
		{
			Log("warning : create render target UAV texture with multisample is not supported, texture will be created with no multisample");
			multisampleType = HQ_MST_NONE;
		}
		HQReturnVal re = HQ_FAILED_INVALID_PARAMETER;
		switch (textureType)
		{
		case HQ_TEXTURE_2D_UAV:
			//create texture from texture manager side, there is almost need for further config, since non-power of two,etc are guaranteed to be supported
			re = pTextureManagerD3D11->InitTextureUAVEx(pNewTex.GetRawPointer(), uavFormat, width, height, 1, hasMipmaps, true);
			break;
		case HQ_TEXTURE_2D_ARRAY_UAV:
			//create texture from texture manager side, there is almost need for further config, since non-power of two,etc are guaranteed to be supported
			re = pTextureManagerD3D11->InitTextureUAVEx(pNewTex.GetRawPointer(), uavFormat, width, height, arraySize, hasMipmaps, true);
			break;
		default:
			//TO DO
			break;
		}
		if (HQFailed(re))
		{
			pTextureManager->RemoveTexture(textureID);
			return re;
		}
	}
	
	DXGI_FORMAT D3Dformat = HQRenderTargetManagerD3D11::GetD3DFormat(format);
	
	hq_uint32 numMipmaps = 1;
	if (hasMipmaps && multisampleType == HQ_MST_NONE)
		numMipmaps = HQBaseTextureManager::CalculateFullNumMipmaps(width , height);//full range mipmap level
	
	hquint32 Exp;
	if (!HQBaseTextureManager::IsPowerOfTwo(width, &Exp) || !HQBaseTextureManager::IsPowerOfTwo(height, &Exp))
	{
		if (numMipmaps > 1 && !g_pD3DDev->IsNpotTextureFullySupported(textureType))//only one mipmap level is supported
		{
			numMipmaps = 1;
			Log("warning : creating a non-power-of-2 render target texture with more than 1 mipmap level is not supported");
		}
	}

	HQRenderTargetTextureD3D *pNewRenderTarget = 
		new HQRenderTargetTextureD3D(this->pD3DDevice,
									width , height , arraySize,
									multisampleType ,D3Dformat , 
									numMipmaps , pNewTex
									);
	if (pNewRenderTarget == NULL || HQFailed(pNewRenderTarget->Init()))
	{
		pTextureManager->RemoveTexture(textureID);
		return HQ_FAILED_MEM_ALLOC;
	}

	
	if(!this->renderTargets.AddItem(pNewRenderTarget , pRenderTargetID_Out))
	{
		HQ_DELETE (pNewRenderTarget);
		pTextureManager->RemoveTexture(textureID);
		return HQ_FAILED_MEM_ALLOC;
	}
	
	if(pTextureID_Out != NULL)
		*pTextureID_Out = textureID;

	return HQ_OK;
}


HQReturnVal HQRenderTargetManagerD3D11::CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										HQDepthStencilBufferView **pDepthStencilBufferID_Out)
{
	char str[256];
	if(!g_pD3DDev->IsDSFormatSupported(format))
	{
		HQBaseRenderTargetManager::GetString(format , str);
		Log("CreateDepthStencilBuffer() failed : %s is not supported!" , str);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	if (!g_pD3DDev->IsDSMultisampleTypeSupported(format , multisampleType))
	{
		HQBaseRenderTargetManager::GetString(multisampleType , str);
		Log("CreateDepthStencilBuffer() failed : %s is not supported!" , str);
		return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT;
	}
	
	DXGI_FORMAT D3Dformat = HQRenderTargetManagerD3D11::GetD3DFormat(format);
	
	HQDepthStencilBufferD3D *pNewBuffer = 
		new HQDepthStencilBufferD3D(
				this->pD3DDevice,
				width , height ,
				multisampleType ,
				D3Dformat);
	
	if (pNewBuffer == NULL || HQFailed(pNewBuffer->Init()))
		return HQ_FAILED_MEM_ALLOC;
	if (!this->depthStencilBuffers.AddItem(pNewBuffer , pDepthStencilBufferID_Out))
	{
		HQ_DELETE (pNewBuffer);
		return  HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}



HQReturnVal HQRenderTargetManagerD3D11::CreateRenderTargetGroupImpl(
											 const HQRenderTargetDesc *renderTargetDescs, 
											 HQDepthStencilBufferView* depthStencilBufferID,
											 hq_uint32 numRenderTargets,
											 HQBaseRenderTargetGroup **ppRenderTargetGroupOut)
{

	if (numRenderTargets > this->maxActiveRenderTargets)
	{
		this->Log("Error : CreateRenderTargetGroupImpl() failed because parameter <numRenderTargets> is larger than %d!" , this->maxActiveRenderTargets);
		return HQ_FAILED;
	}


	if (renderTargetDescs == NULL || numRenderTargets == 0)
	{
		return HQ_FAILED_INVALID_PARAMETER;
	}//if (renderTargetDescs == NULL || numRenderTargets == 0)
	
	bool allNull = true;//all render targets in array is invalid
	bool differentSize = false;//render targets don't have same size
	
	HQRenderTargetGroupD3D11* newGroup = new HQRenderTargetGroupD3D11(numRenderTargets);

	for (hq_uint32 i = 0 ; i < numRenderTargets ; ++i)
	{
		HQSharedPtr<HQBaseRenderTargetView> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDescs[i].renderTargetID);
		if (pRenderTarget == NULL)
		{
			newGroup->renderTargetViews[i] = NULL;
			newGroup->renderTargets[i].pRenderTarget = HQSharedPtr<HQBaseRenderTargetView> ::null;

			continue;
		}
		if (allNull)//now we know that at least one render target is valid .If not , this line can't be reached
		{
			allNull = false;
			newGroup->commonWidth = pRenderTarget->width;
			newGroup->commonHeight = pRenderTarget->height;
		}
		else if (pRenderTarget->width != newGroup->commonWidth || pRenderTarget->height != newGroup->commonHeight)
		{
			//this render target has different size
			differentSize = true;
			break;//stop
		}
		
		if (pRenderTarget->IsTexture())
		{
			ID3D11RenderTargetView** ppRTView = (ID3D11RenderTargetView**)pRenderTarget->GetData();

			if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
				newGroup->renderTargetViews[i] = ppRTView [renderTargetDescs[i].cubeFace];
			else if (pRenderTarget->GetTexture()->type == HQ_TEXTURE_2D_ARRAY || 
				pRenderTarget->GetTexture()->type == HQ_TEXTURE_2D_ARRAY_UAV)
			{
				if (renderTargetDescs[i].arraySlice < static_cast<HQRenderTargetTextureD3D*> (pRenderTarget.GetRawPointer())->numViews)
					newGroup->renderTargetViews[i] = ppRTView[renderTargetDescs[i].arraySlice];
			}
			else
				newGroup->renderTargetViews[i] = ppRTView[0];
		}//if (pRenderTarget->IsTexture())

		newGroup->renderTargets[i].pRenderTarget = pRenderTarget;
		newGroup->renderTargets[i].cubeFace = renderTargetDescs[i].cubeFace;
	
	}//for (i)

	if (differentSize)
	{
		delete newGroup;

		this->Log("Error : CreateRenderTargetGroupImpl() failed because render targets don't have same size!");
		return HQ_FAILED_DIFFERENT_RENDER_TARGETS_SIZE;
	}

	if (allNull)//all render targets is invalid
	{
		delete newGroup;

		this->Log("Error : CreateRenderTargetGroupImpl() failed because every specified render targets are invalid!");
		return HQ_FAILED;
	}
	
	// depth stencil buffer

	HQSharedPtr<HQBaseDepthStencilBufferView> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);
	if (pDepthStencilBuffer == NULL)
	{
		newGroup->pDepthStencilView = NULL;
	}
	else
	{
		if (pDepthStencilBuffer->width < newGroup->commonWidth || pDepthStencilBuffer->height < newGroup->commonHeight)
		{
			delete newGroup;

			this->Log("Error : CreateRenderTargetGroupImpl() failed because depth stencil buffer is too small!");
			return HQ_FAILED_DEPTH_STENCIL_BUFFER_TOO_SMALL;
		}

		newGroup->pDepthStencilView = (ID3D11DepthStencilView *)pDepthStencilBuffer->GetData();
	}

	newGroup->pDepthStencilBuffer = pDepthStencilBuffer;

	*ppRenderTargetGroupOut = newGroup;

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D11::ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& base_group)
{
	HQRenderTargetGroupD3D11* group = static_cast <HQRenderTargetGroupD3D11*> (base_group.GetRawPointer());

	if (group == NULL)
	{
		//active default back buffer and depth stencil buffer
		this->ActiveDefaultFrameBuffer();
		return HQ_OK;
	}//if (group == NULL)
	
	//must unbind every bound textures before activate these render targets
	for (hq_uint32 i = 0; i < group->numRenderTargets; ++i)
	{
		if (group->renderTargetViews[i] != NULL)
		{
			static_cast<HQTextureManagerD3D11*> (this->pTextureManager)->UnbindTextureFromAllTextureSlots(group->renderTargets[i].pRenderTarget->GetTexture());
			static_cast<HQTextureManagerD3D11*> (this->pTextureManager)->UnbindTextureFromAllUAVSlots(group->renderTargets[i].pRenderTarget->GetTexture());
		}
	}

	this->renderTargetWidth = group->commonWidth;
	this->renderTargetHeight = group->commonHeight;

	this->renderTargetViews = group->renderTargetViews;
	this->pDepthStencilView = group->pDepthStencilView;
	
	this->pD3DContext->OMSetRenderTargets(group->numRenderTargets , group->renderTargetViews  , group->pDepthStencilView);
	
	g_pD3DDev->ResetViewports();//reset viewport

	return HQ_OK;
}

void HQRenderTargetManagerD3D11::ActiveDefaultFrameBuffer()
{
	//active default back buffer
	this->renderTargetViews = &this->pD3DBackBuffer;
	this->pDepthStencilView = this->pD3DDSBuffer;

	this->pD3DContext->OMSetRenderTargets(1 , this->renderTargetViews , this->pDepthStencilView);
	
	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();

	g_pD3DDev->ResetViewports();//reset viewport
}



HQReturnVal HQRenderTargetManagerD3D11::GenerateMipmaps(HQRenderTargetView* renderTargetTextureID)
{
	HQBaseCustomRenderBuffer* pRenderTarget = this->renderTargets.GetItemRawPointer(renderTargetTextureID);
	
#if defined _DEBUG || defined DEBUG
	if (pRenderTarget == NULL || !pRenderTarget->IsTexture())
		return HQ_FAILED_INVALID_ID;
#endif
	HQTextureResourceD3D11* pTextureResource = (HQTextureResourceD3D11 *)pRenderTarget->GetTexture()->pData;
	UINT numMipmaps = ((HQBaseRenderTargetTexture*)pRenderTarget)->numMipmaps;
	if (pTextureResource->pResourceView != NULL && numMipmaps > 1)
		pD3DContext->GenerateMips(pTextureResource->pResourceView);

	return HQ_OK;
}

void HQRenderTargetManagerD3D11::SetDefaultRenderTargetView(ID3D11RenderTargetView *_pD3DBackBuffer)
{
	this->pD3DBackBuffer = _pD3DBackBuffer;
}
void HQRenderTargetManagerD3D11::SetDefaultDepthStencilView(ID3D11DepthStencilView *_pD3DDSBuffer)
{
	if(this->IsUsingDefaultFrameBuffer())
		this->pDepthStencilView = _pD3DDSBuffer;
	this->pD3DDSBuffer = _pD3DDSBuffer;
}


HQReturnVal HQRenderTargetManagerD3D11::SetUAVForGraphicsShader(hquint32 slot, ID3D11UnorderedAccessView * pUAV)
{
	if (this->pUAVSlots[slot] != pUAV)
	{
		this->pUAVSlots[slot] = pUAV;
		if (pUAV == NULL)
		{
			if (this->minUsedUAVSlot == slot)
			{
				while (this->minUsedUAVSlot < this->maxUsedUAVSlot && this->pUAVSlots[this->minUsedUAVSlot] == NULL)
				{
					this->minUsedUAVSlot++;
				}
			}

			if (this->maxUsedUAVSlot == slot)
			{
				while (this->maxUsedUAVSlot >= this->minUsedUAVSlot && this->pUAVSlots[this->maxUsedUAVSlot] == NULL)
				{
					this->maxUsedUAVSlot--;
				}
			}
		}//if (pUAV == NULL)
		else
		{
			if ((hqint32)slot < this->minUsedUAVSlot)
				this->minUsedUAVSlot = slot;
			if ((hqint32)slot > this->maxUsedUAVSlot)
				this->maxUsedUAVSlot = slot;
		}
		this->flags |= HQ_UAV_SLOTS_CHANGED;
	}//if (this->pUAVSlots[slot] != pUAV)

	return HQ_OK;
}

void HQRenderTargetManagerD3D11::OnDrawOrDispatch()
{
	if (this->flags & HQ_UAV_SLOTS_CHANGED)
	{
		hquint32 numRTVs = this->GetNumActiveRenderTargets();
		hqint32 numUAVs = this->maxUsedUAVSlot - this->minUsedUAVSlot + 1;


		if (numUAVs > 0)
		{
#if HQ_KEEP_RTV
			numRTVs = D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL;
#else
			numRTVs = min(numRTVs, (hquint32)this->minUsedUAVSlot);//UAV will overwrite RTV slot
			if (numRTVs != this->GetNumActiveRenderTargets())
				this->force_reactive_rtgroup = true;
#endif
			//set UAVs
			this->pD3DContext->OMSetRenderTargetsAndUnorderedAccessViews(
				numRTVs,
				this->renderTargetViews,
				this->pDepthStencilView,
				(UINT) this->minUsedUAVSlot,
				(UINT)numUAVs,
				this->pUAVSlots + this->minUsedUAVSlot,
				this->UAVInitialCounts + this->minUsedUAVSlot
				);
		}//if (numUAVs > 0)
		else
		{
#if HQ_KEEP_RTV
			numRTVs = D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL;
#endif
			//unset UAVs
			this->pD3DContext->OMSetRenderTargetsAndUnorderedAccessViews(
				numRTVs,
				this->renderTargetViews,
				this->pDepthStencilView,
				0,
				0,
				NULL,
				this->UAVInitialCounts
				);
		}

		this->flags = 0;
	}
}