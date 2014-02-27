/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQDeviceD3D9.h"

/*-------------------------*/

struct HQDepthStencilBufferD3D9 : public HQBaseCustomRenderBuffer
{
	HQDepthStencilBufferD3D9(LPDIRECT3DDEVICE9 pD3DDevice , hq_uint32 width ,hq_uint32 height ,
							HQMultiSampleType multiSampleType,
							D3DFORMAT format)
			: HQBaseCustomRenderBuffer(width , height , 
									  multiSampleType)
	{
		this->format = format;
		this->pD3DDevice = pD3DDevice;
		OnResetDevice();
	}
	~HQDepthStencilBufferD3D9()
	{
		OnLostDevice();
	}
	void *GetData()
	{
		return pD3DSurface;
	}
	void OnLostDevice()
	{
		SafeRelease(pD3DSurface);
	}

	void OnResetDevice()
	{
		DWORD multisampleQuality;
		g_pD3DDev->IsMultiSampleSupport(this->format ,
				(D3DMULTISAMPLE_TYPE) this->multiSampleType ,
				&multisampleQuality);

		if (FAILED(pD3DDevice->CreateDepthStencilSurface(
				this->width , this->height ,
				this->format , (D3DMULTISAMPLE_TYPE) this->multiSampleType ,
				multisampleQuality , FALSE , 
				&this->pD3DSurface , NULL)))
			this->pD3DSurface = NULL;
	}

	D3DFORMAT format;
	LPDIRECT3DSURFACE9 pD3DSurface;
	LPDIRECT3DDEVICE9 pD3DDevice;
};

struct HQRenderTargetTextureD3D9 : public HQBaseRenderTargetTexture
{
	HQRenderTargetTextureD3D9(LPDIRECT3DDEVICE9 pD3DDevice,
							hq_uint32 width ,hq_uint32 height ,
							HQMultiSampleType multiSampleType,
							D3DFORMAT format , hq_uint32 numMipmaps,
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
			this->pD3DSurfaces = new LPDIRECT3DSURFACE9[1];
			this->pD3DSurfaces[0] = NULL;
			break;
		case HQ_TEXTURE_CUBE:
			this->pD3DSurfaces = new LPDIRECT3DSURFACE9[6];
			for(hq_uint32 i = 0 ; i < 6 ; ++i)
				this->pD3DSurfaces[i] = NULL;
			break;
		}

		OnResetDevice();
	}
	~HQRenderTargetTextureD3D9()
	{
		OnLostDevice();
		SafeDeleteArray(pD3DSurfaces);
	}

	void *GetData()
	{
		return pD3DSurfaces;
	}

	void OnLostDevice()
	{
		if (pD3DSurfaces)
		{
			if (this->pTexture->type == HQ_TEXTURE_CUBE)
			{
				for(hq_uint32 i = 0 ; i < 6 ; ++i)
					SafeRelease(pD3DSurfaces[i]);
			}
			else
				SafeRelease(pD3DSurfaces[0]);
		}

		LPDIRECT3DBASETEXTURE9 pD3DTexture = (LPDIRECT3DBASETEXTURE9)this->pTexture->pData;
		if(pD3DTexture)
		{
			pD3DTexture->Release();
			this->pTexture->pData = NULL;
		}
	}
	void OnResetDevice()
	{
		HRESULT hr ; 
		DWORD usage = D3DUSAGE_RENDERTARGET;
		if (this->numMipmaps == 0)
			usage |= D3DUSAGE_AUTOGENMIPMAP;

		switch(this->pTexture->type)
		{
		case HQ_TEXTURE_2D:
			{
				hr = pD3DDevice->CreateTexture(this->width , this->height , this->numMipmaps ,
					usage , this->format , D3DPOOL_DEFAULT , 
					(LPDIRECT3DTEXTURE9*)&(this->pTexture->pData) , NULL);

				if(!FAILED(hr))
				{
					LPDIRECT3DTEXTURE9 pD3DTex = (LPDIRECT3DTEXTURE9)this->pTexture->pData;
					pD3DTex->GetSurfaceLevel( 0 , &this->pD3DSurfaces[0]);
				}
				else
				{
					this->pTexture->pData = NULL;
					this->pD3DSurfaces[0] = NULL;
				}
			}
			break;
		case HQ_TEXTURE_CUBE:
			{
				hr = pD3DDevice->CreateCubeTexture(this->width , this->numMipmaps ,
					usage , this->format , D3DPOOL_DEFAULT , 
					(LPDIRECT3DCUBETEXTURE9*)&(this->pTexture->pData) , NULL);
				if(!FAILED(hr))
				{
					LPDIRECT3DCUBETEXTURE9 pD3DTex = (LPDIRECT3DCUBETEXTURE9)this->pTexture->pData;
					for (hq_uint32 i = 0 ; i < 6 ; ++i)
						pD3DTex->GetCubeMapSurface((D3DCUBEMAP_FACES) i , 0 , &this->pD3DSurfaces[i]);
				}
				else
				{
					this->pTexture->pData = NULL;
					for (hq_uint32 i = 0 ; i < 6 ; ++i)
						this->pD3DSurfaces[i] = NULL;
				}
			}
			break;
		}
	}

	D3DFORMAT format;
	LPDIRECT3DSURFACE9 *pD3DSurfaces;
	LPDIRECT3DDEVICE9 pD3DDevice;
};

D3DFORMAT HQRenderTargetManagerD3D9::GetD3DFormat(HQRenderTargetFormat format)
{
	switch(format)
	{
	case HQ_RTFMT_R_FLOAT32:
		return D3DFMT_R32F;
	case HQ_RTFMT_R_FLOAT16:
		return D3DFMT_R16F;
	case HQ_RTFMT_RGBA_32:
		return D3DFMT_A8R8G8B8;
	case HQ_RTFMT_A_UINT8:
		return D3DFMT_A8;
	case HQ_RTFMT_R_UINT8:
		return D3DFMT_L8;
	case HQ_RTFMT_RGBA_FLOAT64:
		return D3DFMT_A16B16G16R16F;
	case HQ_RTFMT_RG_FLOAT32:
		return D3DFMT_G16R16F;
	case HQ_RTFMT_RGBA_FLOAT128:
		return D3DFMT_A32B32G32R32F;
	case HQ_RTFMT_RG_FLOAT64:
		return D3DFMT_G32R32F;
	}
	return D3DFMT_UNKNOWN;
}

D3DFORMAT HQRenderTargetManagerD3D9::GetD3DFormat(HQDepthStencilFormat format)
{
	switch(format)
	{
	case HQ_DSFMT_DEPTH_16:
		return D3DFMT_D16;
	case HQ_DSFMT_DEPTH_24:
		return D3DFMT_D24X8;
	case HQ_DSFMT_DEPTH_32:
		return D3DFMT_D32;
	case HQ_DSFMT_STENCIL_8:
		return D3DFMT_UNKNOWN;
	case HQ_DSFMT_DEPTH_24_STENCIL_8:
		return D3DFMT_D24S8;
	}
	return D3DFMT_UNKNOWN;
}

HQRenderTargetManagerD3D9::HQRenderTargetManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice, 
										 hq_uint32 maxActiveRenderTargets,
										 HQBaseTextureManager *pTexMan,
										 HQLogStream *logFileStream , bool flushLog)
						: HQBaseRenderTargetManager(maxActiveRenderTargets , 
										 pTexMan , logFileStream , 
										 "D3D9 Render Target Manager :" , flushLog)
{
	this->pD3DDevice = pD3DDevice;
	this->pD3DBackBuffer = NULL;
	

	this->OnResetDevice();

	Log("Init done!");
}

HQRenderTargetManagerD3D9::~HQRenderTargetManagerD3D9()
{
	OnLostDevice();
	Log("Released!");
}



void HQRenderTargetManagerD3D9::OnLostDevice()
{
	HQItemManager<HQBaseCustomRenderBuffer>::Iterator ite;
	this->renderTargets.GetIterator(ite);
	while(!ite.IsAtEnd())
	{
		ite->OnLostDevice();
		++ite;
	}

	this->depthStencilBuffers.GetIterator(ite);
	while(!ite.IsAtEnd())
	{
		ite->OnLostDevice();
		++ite;
	}

	SafeRelease(this->pD3DBackBuffer);
	SafeRelease(this->pD3DDSBuffer);
}

void HQRenderTargetManagerD3D9::OnResetDevice()
{
	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();

	HQItemManager<HQBaseCustomRenderBuffer>::Iterator ite;
	this->renderTargets.GetIterator(ite);
	while(!ite.IsAtEnd())
	{
		ite->OnResetDevice();
		++ite;
	}

	this->depthStencilBuffers.GetIterator(ite);
	while(!ite.IsAtEnd())
	{
		ite->OnResetDevice();
		++ite;
	}

	pD3DDevice->GetRenderTarget(0, &pD3DBackBuffer);
	if (pD3DDevice->GetDepthStencilSurface(&pD3DDSBuffer) == D3DERR_NOTFOUND)
		pD3DDSBuffer = NULL;

	if (!this->currentUseDefaultBuffer)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pBuffer = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
		for (hq_uint32 i = 0; i < this->numActiveRenderTargets ; ++i)
		{
			pBuffer = activeRenderTargets[i].pRenderTarget;
			if (pBuffer == NULL)
				continue;
			if (pBuffer->IsTexture())
			{
				LPDIRECT3DSURFACE9 *pD3DSurfaces = (LPDIRECT3DSURFACE9 *)pBuffer->GetData();
				if (pBuffer->GetTexture()->type == HQ_TEXTURE_CUBE)
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[this->activeRenderTargets[i].cubeFace]);
				else
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[0]);
			}
		}

		if (pActiveDepthStencilBuffer != NULL)
		{
			LPDIRECT3DSURFACE9 pD3DSurface = (LPDIRECT3DSURFACE9)pActiveDepthStencilBuffer->GetData();
			pD3DDevice->SetDepthStencilSurface(pD3DSurface);
		}
	}
}

HQReturnVal HQRenderTargetManagerD3D9::CreateRenderTargetTexture(hq_uint32 width, hq_uint32 height, bool hasMipmaps, 
											   HQRenderTargetFormat format, HQMultiSampleType multisampleType, 
											   HQTextureType textureType, 
											   hq_uint32 *pRenderTargetID_Out, 
											   hq_uint32 *pTextureID_Out)
{
	if (!g_pD3DDev->IsNpotTextureSupported(textureType))//texture size must be power of two
	{
		hq_uint32 exp;
		if (!HQBaseTextureManager::IsPowerOfTwo( width  , &exp ))
			width = 0x1 << exp;
		if (!HQBaseTextureManager::IsPowerOfTwo( height  , &exp ))
			height = 0x1 << exp;
	}
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
		height = width;
	
	D3DFORMAT D3Dformat = HQRenderTargetManagerD3D9::GetD3DFormat(format);
	
	hq_uint32 numMipmaps = 1;
	if (hasMipmaps)
		numMipmaps = 0;//full range mipmap level
	
	HQRenderTargetTextureD3D9 *pNewRenderTarget = 
		new HQRenderTargetTextureD3D9(this->pD3DDevice,
									width , height ,
									multisampleType ,D3Dformat , 
									numMipmaps , textureID , pNewTex
									);
	if (pNewRenderTarget == NULL || pNewTex->pData == NULL)
	{
		pTextureManager->RemoveTexture(textureID);
		return HQ_FAILED_MEM_ALLOC;
	}

	
	if(!this->renderTargets.AddItem(pNewRenderTarget , pRenderTargetID_Out))
	{
		delete pNewRenderTarget;
		pTextureManager->RemoveTexture(textureID);
		return HQ_FAILED_MEM_ALLOC;
	}
	
	if(pTextureID_Out != NULL)
		*pTextureID_Out = textureID;

	return HQ_OK;
}


HQReturnVal HQRenderTargetManagerD3D9::CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
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
	D3DFORMAT D3Dformat = HQRenderTargetManagerD3D9::GetD3DFormat(format);
	
	if (!g_pD3DDev->IsMultiSampleSupport(D3Dformat , (D3DMULTISAMPLE_TYPE) multisampleType ,NULL))
	{
		HQBaseRenderTargetManager::GetString(multisampleType , str);
		Log("CreateDepthStencilBuffer() failed : %s is not supported!" , str);
		return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT;
	}
	

	HQDepthStencilBufferD3D9 *pNewBuffer = 
		new HQDepthStencilBufferD3D9(
				this->pD3DDevice,
				width , height ,
				multisampleType ,
				D3Dformat);
	
	if (pNewBuffer == NULL || pNewBuffer->pD3DSurface == NULL)
		return HQ_FAILED_MEM_ALLOC;
	if (!this->depthStencilBuffers.AddItem(pNewBuffer , pDepthStencilBufferID_Out))
	{
		delete pNewBuffer;
		return  HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D9::ActiveRenderTarget(const HQRenderTargetDesc &renderTargetDesc , 
														  hq_uint32 depthStencilBufferID )
{
	if (g_pD3DDev->GetFlags() & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
	
	HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDesc.renderTargetID);

	if (pRenderTarget == NULL)
	{
		//active default back buffer and depth stencil buffer
		this->InvalidateRenderTargets();
		return HQ_FAILED;
	}

	this->renderTargetWidth = pRenderTarget->width;
	this->renderTargetHeight = pRenderTarget->height;
	
	if (pRenderTarget == this->activeRenderTargets[0].pRenderTarget)//no change 
	{
		if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		{
			if (renderTargetDesc.cubeFace != this->activeRenderTargets[0].cubeFace)//different cube face
			{
				LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();
				pD3DDevice->SetRenderTarget( 0 , pD3DSurfaces[ renderTargetDesc.cubeFace ]);
			}//if (renderTargetDesc.cubeFace != this->activeRenderTargets[0].cubeFace)
		}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
	}//pRenderTarget == this->activeRenderTargets[0].pRenderTarget
	else//different render target
	{
		if (pRenderTarget->IsTexture())
		{
			LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();

			if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
				pD3DDevice->SetRenderTarget( 0 , pD3DSurfaces[ renderTargetDesc.cubeFace ]);
			else
				pD3DDevice->SetRenderTarget( 0 , pD3DSurfaces[0]);
		}//if (pRenderTarget->IsTexture())
	}//else of if (pRenderTarget == this->activeRenderTargets[0].pRenderTarget)
	
	this->activeRenderTargets[0].pRenderTarget = pRenderTarget;
	this->activeRenderTargets[0].cubeFace = renderTargetDesc.cubeFace;

	
	
	//deactive other render targets
	for (hq_uint32 i = 1 ; i < this->numActiveRenderTargets ; ++i)
	{
		if (this->activeRenderTargets[i].pRenderTarget != NULL)
		{
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
			pD3DDevice->SetRenderTarget(i , NULL);
		}
	}
	

	this->numActiveRenderTargets = 1;
	
	//active depth stencil buffer
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);
	if (this->currentUseDefaultBuffer || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
	{
		if (pDepthStencilBuffer == NULL)
		{
			pD3DDevice->SetDepthStencilSurface(NULL);
		}
		else
		{
			pD3DDevice->SetDepthStencilSurface((LPDIRECT3DSURFACE9)pDepthStencilBuffer->GetData());
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	this->currentUseDefaultBuffer = false;

	this->ResetViewPort();
	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D9::ActiveRenderTargets(const HQRenderTargetDesc *renderTargetDescs, 
											 hq_uint32 depthStencilBufferID, 
											 hq_uint32 numRenderTargets)
{
	if (g_pD3DDev->GetFlags() & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
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
		this->InvalidateRenderTargets();
		return HQ_OK;
	}//if (renderTargetDescs == NULL || numRenderTargets == 0)
	
	bool allNull = true;//all render targets in array is invalid
	
	for (hq_uint32 i = 0 ; i < numRenderTargets ; ++i)
	{
		
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDescs[i].renderTargetID);
		if (pRenderTarget == NULL)
		{
			pD3DDevice->SetRenderTarget(i , NULL);
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
			continue;
		}
		if (allNull)//now we know that at least one render target is valid .If not , this line can't be reached
		{
			allNull = false;
			this->renderTargetWidth = pRenderTarget->width;
			this->renderTargetHeight = pRenderTarget->height;
		}
		
		if (pRenderTarget == this->activeRenderTargets[i].pRenderTarget)//no change 
		{
			if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
			{
				if (renderTargetDescs[i].cubeFace != this->activeRenderTargets[i].cubeFace)//different cube face
				{
					LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[ renderTargetDescs[i].cubeFace ]);
				}//if (renderTargetDescs[i].cubeFace != this->activeRenderTargets[i].cubeFace)
			}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		}//pRenderTarget == this->activeRenderTargets[i].pRenderTarget
		else//different render target
		{
			if (pRenderTarget->IsTexture())
			{
				LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();

				if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[ renderTargetDescs[i].cubeFace ]);
				else
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[0]);
			}//if (pRenderTarget->IsTexture())
		}//else of if (pRenderTarget == this->activeRenderTargets[i].pRenderTarget)
		
		this->activeRenderTargets[i].pRenderTarget = pRenderTarget;
		this->activeRenderTargets[i].cubeFace = renderTargetDescs[i].cubeFace;
	}//for (i)

	
	
	//deactive other render targets
	for (hq_uint32 i = numRenderTargets ; i < this->numActiveRenderTargets ; ++i)
	{
		if (this->activeRenderTargets[i].pRenderTarget != NULL)
		{
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
			pD3DDevice->SetRenderTarget(i , NULL);
		}
	}
	//all render targets are invalid , so we need to active default frame buffer
	if (allNull && !this->currentUseDefaultBuffer)
	{
		this->ResetToDefaultFrameBuffer();

		return HQ_FAILED;
	}

	this->numActiveRenderTargets = numRenderTargets;
	
	//active depth stencil buffer
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);
	if (this->currentUseDefaultBuffer || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
	{
		if (pDepthStencilBuffer == NULL)
		{
			pD3DDevice->SetDepthStencilSurface(NULL);
		}
		else
		{
			pD3DDevice->SetDepthStencilSurface((LPDIRECT3DSURFACE9)pDepthStencilBuffer->GetData());
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	this->currentUseDefaultBuffer = false;

	this->ResetViewPort();
	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D9::RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList)
{
	bool allNull = true;//all render targets in array is invalid
	
	for (hq_uint32 i = 0 ; i < savedList.numActiveRenderTargets ; ++i)
	{
		
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = savedList[i].pRenderTarget;
		if (pRenderTarget == NULL)
		{
			pD3DDevice->SetRenderTarget(i , NULL);
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
			continue;
		}
		if (allNull)//now we know that at least one render target is valid .If not , this line can't be reached
		{
			allNull = false;
			this->renderTargetWidth = pRenderTarget->width;
			this->renderTargetHeight = pRenderTarget->height;
		}
		
		if (pRenderTarget == this->activeRenderTargets[i].pRenderTarget)//no change 
		{
			if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
			{
				if (savedList[i].cubeFace != this->activeRenderTargets[i].cubeFace)//different cube face
				{
					LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[ savedList[i].cubeFace ]);
				}//if (renderTargetDescs[i].cubeFace != this->activeRenderTargets[i].cubeFace)
			}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		}//pRenderTarget == this->activeRenderTargets[i].pRenderTarget
		else//different render target
		{
			if (pRenderTarget->IsTexture())
			{
				LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();

				if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[ savedList[i].cubeFace ]);
				else
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[0]);
			}//if (pRenderTarget->IsTexture())
		}//else of if (pRenderTarget == this->activeRenderTargets[i].pRenderTarget)
		
		this->activeRenderTargets[i].pRenderTarget = pRenderTarget;
		this->activeRenderTargets[i].cubeFace = savedList[i].cubeFace;
	}//for (i)

	
	
	//deactive other render targets
	for (hq_uint32 i = savedList.numActiveRenderTargets ; i < this->numActiveRenderTargets ; ++i)
	{
		if (this->activeRenderTargets[i].pRenderTarget != NULL)
		{
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
			pD3DDevice->SetRenderTarget(i , NULL);
		}
	}
	//all render targets are invalid , so we need to active default frame buffer
	if (allNull && !this->currentUseDefaultBuffer)
	{
		this->ResetToDefaultFrameBuffer();

		return HQ_OK;
	}

	this->numActiveRenderTargets = savedList.numActiveRenderTargets;;
	
	//active depth stencil buffer
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = savedList.pActiveDepthStencilBuffer;
	if (this->currentUseDefaultBuffer || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
	{
		if (pDepthStencilBuffer == NULL)
		{
			pD3DDevice->SetDepthStencilSurface(NULL);
		}
		else
		{
			pD3DDevice->SetDepthStencilSurface((LPDIRECT3DSURFACE9)pDepthStencilBuffer->GetData());
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	this->currentUseDefaultBuffer = false;

	this->ResetViewPort();

	return HQ_OK;
}

void HQRenderTargetManagerD3D9::InvalidateRenderTargets()
{
	if (this->currentUseDefaultBuffer)
		return;

	this->activeRenderTargets[0].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;

	for (hq_uint32 i = 1 ; i < this->numActiveRenderTargets ; ++i)
	{
		pD3DDevice->SetRenderTarget( i , NULL);
		this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
	}

	this->ResetToDefaultFrameBuffer();
}

void HQRenderTargetManagerD3D9::ResetToDefaultFrameBuffer()
{
	//active default back buffer
	pD3DDevice->SetRenderTarget(0 , pD3DBackBuffer);
	//active default depth stencil buffer
	pD3DDevice->SetDepthStencilSurface(pD3DDSBuffer);

	this->pActiveDepthStencilBuffer = HQSharedPtr<HQBaseCustomRenderBuffer> ::null;
	this->currentUseDefaultBuffer = true;
	this->numActiveRenderTargets = 0;

	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();
	
	this->ResetViewPort();
}

void HQRenderTargetManagerD3D9::ResetViewPort()
{
	const HQViewPort& viewport = g_pD3DDev->GetViewPort();
	if (viewport.x != 0 || viewport.y !=0 || 
		viewport.width != this->renderTargetWidth ||
		viewport.height != this->renderTargetHeight)
		g_pD3DDev->SetViewPort(viewport);
}

HQReturnVal HQRenderTargetManagerD3D9::GenerateMipmaps(hq_uint32 renderTargetTextureID)
{
	HQBaseCustomRenderBuffer* pRenderTarget = this->renderTargets.GetItemRawPointer(renderTargetTextureID);
#if defined _DEBUG || defined DEBUG
	if (pRenderTarget == NULL || !pRenderTarget->IsTexture())
		return HQ_FAILED_INVALID_ID;
#endif
	LPDIRECT3DBASETEXTURE9 pD3Dtexture = (LPDIRECT3DBASETEXTURE9)pRenderTarget->GetTexture()->pData;
	if (pD3Dtexture != NULL)
		pD3Dtexture->GenerateMipSubLevels();

	return HQ_OK;
}
