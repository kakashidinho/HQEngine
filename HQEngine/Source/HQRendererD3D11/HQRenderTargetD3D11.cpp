/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQDeviceD3D11.h"

/*-------------------------*/

struct HQDepthStencilBufferD3D : public HQBaseCustomRenderBuffer
{
	HQDepthStencilBufferD3D(ID3D11Device * pD3DDevice , hq_uint32 width ,hq_uint32 height ,
							HQMultiSampleType multiSampleType,
							DXGI_FORMAT format)
			: HQBaseCustomRenderBuffer(width , height , 
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

	int Init()
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

struct HQRenderTargetTextureD3D : public HQBaseRenderTargetTexture
{
	HQRenderTargetTextureD3D(ID3D11Device* pD3DDevice,
							hq_uint32 width ,hq_uint32 height ,
							HQMultiSampleType multiSampleType,
							DXGI_FORMAT format , hq_uint32 numMipmaps,
							hq_uint32 textureID , HQSharedPtr<HQTexture> pTex)
		: HQBaseRenderTargetTexture(width , height ,
							multiSampleType, numMipmaps,
							textureID , pTex)
	{
		this->format = format;
		this->pD3DDevice = pD3DDevice;
		switch (pTex->type)
		{
		case HQ_TEXTURE_2D:
			this->ppRTView = HQ_NEW ID3D11RenderTargetView *[1];
			this->ppRTView[0] = NULL;
			break;
		case HQ_TEXTURE_CUBE:
			this->ppRTView = HQ_NEW ID3D11RenderTargetView *[6];
			for(hq_uint32 i = 0 ; i < 6 ; ++i)
				this->ppRTView[i] = NULL;
			break;
		}

	}
	~HQRenderTargetTextureD3D()
	{
		if (ppRTView != NULL)
		{
			if (pTexture->type == HQ_TEXTURE_CUBE)
			{
				for(hq_uint32 i = 0 ; i < 6 ; ++i)
					SafeRelease(this->ppRTView[i]);
			}
			else
				SafeRelease(ppRTView[0]);

			delete[] ppRTView;
		}
	}

	void *GetData()
	{
		return ppRTView;
	}

	int Init()
	{
		HRESULT hr ; 
		/*------get multisample quality------------*/
		UINT multisampleQuality;
		UINT sampleCount = (this->multiSampleType > 0 )?(UINT)this->multiSampleType : 1;
		pD3DDevice->CheckMultisampleQualityLevels(this->format ,
			sampleCount ,&multisampleQuality);

		D3D11_TEXTURE2D_DESC rtDesc;
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
		
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
		//render target view desc
		rtvDesc.Format = this->format;
		
		HQTextureResourceD3D11* pTex = (HQTextureResourceD3D11 *)pTexture->pData;
		switch(this->pTexture->type)
		{
		case HQ_TEXTURE_2D:
			{
				//texture desc
				rtDesc.ArraySize = 1;
				//create texture
				hr = pD3DDevice->CreateTexture2D(&rtDesc , NULL , (ID3D11Texture2D**)&pTex->pTexture);

				if(FAILED(hr))
				{
					pTex->pTexture = NULL;
					pTex->pResourceView = NULL;
					break;
				}
				
				if (sampleCount > 1)//multisample texture
				{
					//resource view desc
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
					//render target view desc
					rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
				}
				else
				{
					//resource view desc
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
					srvDesc.Texture2D.MipLevels = this->numMipmaps;
					srvDesc.Texture2D.MostDetailedMip = 0;

					//render target view desc
					rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
					rtvDesc.Texture2D.MipSlice = 0;
				}
				//create shader resource view
				hr = pD3DDevice->CreateShaderResourceView(pTex->pTexture , &srvDesc , &pTex->pResourceView);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					pTex->pResourceView = NULL;
					break;
				}
				//create render target view
				hr = pD3DDevice->CreateRenderTargetView(pTex->pTexture , &rtvDesc , &this->ppRTView[0]);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					SafeRelease(pTex->pResourceView);
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
				hr = pD3DDevice->CreateTexture2D(&rtDesc , NULL , (ID3D11Texture2D**)&pTex->pTexture);

				if(FAILED(hr))
				{
					pTex->pTexture = NULL;
					pTex->pResourceView = NULL;
					break;
				}
				
				//resource view desc
				srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
				srvDesc.TextureCube.MipLevels = this->numMipmaps;
				srvDesc.TextureCube.MostDetailedMip = 0;

				//render target view desc
				rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
				rtvDesc.Texture2DArray.MipSlice = 0;
				rtvDesc.Texture2DArray.ArraySize = 1;
				
				//create shader resource view
				hr = pD3DDevice->CreateShaderResourceView(pTex->pTexture , &srvDesc , &pTex->pResourceView);
				if (FAILED(hr))
				{
					SafeRelease(pTex->pTexture);
					pTex->pResourceView = NULL;
					break;
				}
				//create 6 render target views for 6 faces
				for (int i = 0; i < 6 ; ++i)
				{
					rtvDesc.Texture2DArray.FirstArraySlice = i;
					hr = pD3DDevice->CreateRenderTargetView(pTex->pTexture , &rtvDesc , &this->ppRTView[i]);
					if (FAILED(hr))
					{
						SafeRelease(pTex->pTexture);
						SafeRelease(pTex->pResourceView);
						break;
					}
				}
			}
			break;
		}//switch texture's type

		if (FAILED(hr))
			return HQ_FAILED;
		return HQ_OK;
	}

	DXGI_FORMAT format;
	ID3D11RenderTargetView **ppRTView;//can be one pointer if texture is 2D or 6 pointers if texture is cube
	ID3D11Device* pD3DDevice;
};

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
	
	this->numActiveRenderTargets = 1;
	this->renderTargetViews[0] = this->pD3DBackBuffer;
	this->pDepthStencilView = this->pD3DDSBuffer;
	for (int i = 1 ; i < 8 ; ++i)
		this->renderTargetViews[i] = NULL;

	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();

	Log("Init done!");
}

HQRenderTargetManagerD3D11::~HQRenderTargetManagerD3D11()
{
	this->renderTargets.RemoveAll();
	this->depthStencilBuffers.RemoveAll();
	Log("Released!");
}





HQReturnVal HQRenderTargetManagerD3D11::CreateRenderTargetTexture(hq_uint32 width, hq_uint32 height, bool hasMipmaps, 
											   HQRenderTargetFormat format, HQMultiSampleType multisampleType, 
											   HQTextureType textureType, 
											   hq_uint32 *pRenderTargetID_Out, 
											   hq_uint32 *pTextureID_Out)
{
	char str[256];
	if (!g_pD3DDev->IsRTTFormatSupported(format , textureType , hasMipmaps))
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
	hq_uint32 textureID = 0;
	HQSharedPtr<HQTexture> pNewTex = this->pTextureManager->CreateEmptyTexture(textureType , &textureID);
	if (pNewTex == NULL)
		return HQ_FAILED_MEM_ALLOC;

	if (textureType == HQ_TEXTURE_CUBE)
	{
		height = width;
		if (multisampleType != HQ_MST_NONE)
		{
			Log("warning : create render target cube texture with multisample is not supported, texture will be created with no multisample");
			multisampleType = HQ_MST_NONE;
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
									width , height ,
									multisampleType ,D3Dformat , 
									numMipmaps , textureID , pNewTex
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
										hq_uint32 *pDepthStencilBufferID_Out)
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

HQReturnVal HQRenderTargetManagerD3D11::ActiveRenderTarget(const HQRenderTargetDesc &renderTargetDesc, 
											 hq_uint32 depthStencilBufferID)
{
	
	HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDesc.renderTargetID);
	
#if defined _DEBUG || defined DEBUG
	if (pRenderTarget == NULL)
	{
		//active default frame buffer
		this->ActiveDefaultFrameBuffer();

		return HQ_FAILED;
	}
#endif
	
	this->renderTargetWidth = pRenderTarget->width;
	this->renderTargetHeight = pRenderTarget->height;

	
	if (pRenderTarget->IsTexture())
	{
		ID3D11RenderTargetView** ppRTView = (ID3D11RenderTargetView**)pRenderTarget->GetData();

		if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
			this->renderTargetViews[0] = ppRTView [renderTargetDesc.cubeFace];
		else
			this->renderTargetViews[0] = ppRTView[0];
	}//if (pRenderTarget->IsTexture())

	this->activeRenderTargets[0].pRenderTarget = pRenderTarget;
	this->activeRenderTargets[0].cubeFace = renderTargetDesc.cubeFace;
	
	//release old pointers
	for (hq_uint32 i = 1 ; i < this->numActiveRenderTargets ; ++i)
		this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;

	this->numActiveRenderTargets = 1;
	
	//active depth stencil buffer
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);
	if (this->currentUseDefaultBuffer || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
	{
		if (pDepthStencilBuffer == NULL)
		{
			this->pDepthStencilView = NULL;
		}
		else
		{
			this->pDepthStencilView = (ID3D11DepthStencilView *)pDepthStencilBuffer->GetData();
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	
	this->currentUseDefaultBuffer = false;
	
	this->pD3DContext->OMSetRenderTargets(1 , this->renderTargetViews  , this->pDepthStencilView);
	
	g_pD3DDev->SetViewPort(g_pD3DDev->GetViewPort());//reset viewport

	return HQ_OK;
}


HQReturnVal HQRenderTargetManagerD3D11::ActiveRenderTargets(const HQRenderTargetDesc *renderTargetDescs, 
											 hq_uint32 depthStencilBufferID, 
											 hq_uint32 numRenderTargets)
{
#if defined _DEBUG || defined DEBUG
	if (numRenderTargets > this->maxActiveRenderTargets)
	{
		this->Log("Error : ActiveRenderTargets() failed because parameter <numRenderTargets> is larger than %d!" , this->maxActiveRenderTargets);
		return HQ_FAILED;
	}
#endif

	if (renderTargetDescs == NULL || numRenderTargets == 0)
	{
		//active default back buffer and depth stencil buffer
		this->ActiveDefaultFrameBuffer();
		return HQ_OK;
	}//if (renderTargetDescs == NULL || numRenderTargets == 0)
	
	bool allNull = true;//all render targets in array is invalid
	
	for (hq_uint32 i = 0 ; i < numRenderTargets ; ++i)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDescs[i].renderTargetID);
		if (pRenderTarget == NULL)
		{
			this->renderTargetViews[i] = NULL;
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;

			continue;
		}
		if (allNull)//now we know that at least one render target is valid .If not , this line can't be reached
		{
			allNull = false;
			this->renderTargetWidth = pRenderTarget->width;
			this->renderTargetHeight = pRenderTarget->height;
		}
		
		if (pRenderTarget->IsTexture())
		{
			ID3D11RenderTargetView** ppRTView = (ID3D11RenderTargetView**)pRenderTarget->GetData();

			if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
				this->renderTargetViews[i] = ppRTView [renderTargetDescs[i].cubeFace];
			else
				this->renderTargetViews[i] = ppRTView[0];
		}//if (pRenderTarget->IsTexture())

		this->activeRenderTargets[i].pRenderTarget = pRenderTarget;
		this->activeRenderTargets[i].cubeFace = renderTargetDescs[i].cubeFace;
	
	}//for (i)

	if (allNull)//all render targets is invalid
	{
		//active default frame buffer
		this->ActiveDefaultFrameBuffer();

		return HQ_FAILED;
	}
	//release old pointers
	for (hq_uint32 i = numRenderTargets ; i < this->numActiveRenderTargets ; ++i)
		this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;

	this->numActiveRenderTargets = numRenderTargets;
	
	//active depth stencil buffer

	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);
	if (this->currentUseDefaultBuffer || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
	{
		if (pDepthStencilBuffer == NULL)
		{
			this->pDepthStencilView = NULL;
		}
		else
		{
			this->pDepthStencilView = (ID3D11DepthStencilView *)pDepthStencilBuffer->GetData();
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	
	this->currentUseDefaultBuffer = false;
	
	this->pD3DContext->OMSetRenderTargets(this->numActiveRenderTargets , this->renderTargetViews  , this->pDepthStencilView);
	
	g_pD3DDev->SetViewPort(g_pD3DDev->GetViewPort());//reset viewport

	return HQ_OK;
}

void HQRenderTargetManagerD3D11::ActiveDefaultFrameBuffer()
{
	if (this->currentUseDefaultBuffer)
		return;
	this->renderTargetViews[0] = this->pD3DBackBuffer;//default back buffer
	this->pDepthStencilView = this->pD3DDSBuffer;//default depth stencil buffer
	//active default back buffer
	this->pD3DContext->OMSetRenderTargets(1 , this->renderTargetViews , this->pDepthStencilView);
	
	//release old pointers
	for (hq_uint32 i = 0 ; i < this->numActiveRenderTargets ; ++i)
		this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
	
	this->pActiveDepthStencilBuffer = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
	
	this->currentUseDefaultBuffer = true;
	
	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();

	this->numActiveRenderTargets = 1;

	g_pD3DDev->SetViewPort(g_pD3DDev->GetViewPort());//reset viewport
}


HQReturnVal HQRenderTargetManagerD3D11::RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList)
{
	bool allNull = true;//all render targets in array is invalid
	
	for (hq_uint32 i = 0 ; i < savedList.numActiveRenderTargets ; ++i)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = savedList[i].pRenderTarget;
		if (pRenderTarget == NULL)
		{
			this->renderTargetViews[i] = NULL;
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;

			continue;
		}
		if (allNull)//now we know that at least one render target is valid .If not , this line can't be reached
		{
			allNull = false;
			this->renderTargetWidth = pRenderTarget->width;
			this->renderTargetHeight = pRenderTarget->height;
		}
		
		if (pRenderTarget->IsTexture())
		{
			ID3D11RenderTargetView** ppRTView = (ID3D11RenderTargetView**)pRenderTarget->GetData();

			if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
				this->renderTargetViews[i] = ppRTView [savedList[i].cubeFace];
			else
				this->renderTargetViews[i] = ppRTView[0];
		}//if (pRenderTarget->IsTexture())

		this->activeRenderTargets[i].pRenderTarget = pRenderTarget;
		this->activeRenderTargets[i].cubeFace = savedList[i].cubeFace;
	
	}//for (i)

	if (allNull)//all render targets is invalid
	{
		//active default frame buffer
		this->ActiveDefaultFrameBuffer();

		return HQ_OK;;
	}
	//release old pointers
	for (hq_uint32 i = savedList.numActiveRenderTargets ; i < this->numActiveRenderTargets ; ++i)
		this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;

	this->numActiveRenderTargets = savedList.numActiveRenderTargets;
	
	//active depth stencil buffer

	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = savedList.pActiveDepthStencilBuffer;
	if (this->currentUseDefaultBuffer || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
	{
		if (pDepthStencilBuffer == NULL)
		{
			this->pDepthStencilView = NULL;
		}
		else
		{
			this->pDepthStencilView = (ID3D11DepthStencilView *)pDepthStencilBuffer->GetData();
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	
	this->currentUseDefaultBuffer = false;
	
	this->pD3DContext->OMSetRenderTargets(this->numActiveRenderTargets , this->renderTargetViews  , this->pDepthStencilView);
	
	g_pD3DDev->SetViewPort(g_pD3DDev->GetViewPort());//reset viewport

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D11::GenerateMipmaps(hq_uint32 renderTargetTextureID)
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
	if (this->IsUsingDefaultFrameBuffer())
		this->renderTargetViews[0] = _pD3DBackBuffer;
	
	this->pD3DBackBuffer = _pD3DBackBuffer;
}
void HQRenderTargetManagerD3D11::SetDefaultDepthStencilView(ID3D11DepthStencilView *_pD3DDSBuffer)
{
	if(this->IsUsingDefaultFrameBuffer())
		this->pDepthStencilView = _pD3DDSBuffer;
	this->pD3DDSBuffer = _pD3DDSBuffer;
}
