/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQDeviceD3D9.h"

/*-------------------------*/

struct HQDepthStencilBufferD3D9 : public HQBaseDepthStencilBufferView
{
	HQDepthStencilBufferD3D9(LPDIRECT3DDEVICE9 pD3DDevice , hq_uint32 width ,hq_uint32 height ,
							HQMultiSampleType multiSampleType,
							D3DFORMAT format)
							: HQBaseDepthStencilBufferView(width, height,
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
							HQSharedPtr<HQBaseTexture> pTex)
		: HQBaseRenderTargetTexture(width , height ,
							multiSampleType, numMipmaps,
							pTex)
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

/*----------------HQRenderTargetManagerD3D9-----------*/
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

	this->activeRenderTargets = HQ_NEW HQRenderTargetInfo[maxActiveRenderTargets];
	this->numActiveRenderTargets = 0;

	this->OnResetDevice();

	Log("Init done!");
}

HQRenderTargetManagerD3D9::~HQRenderTargetManagerD3D9()
{
	delete[] this->activeRenderTargets;
	OnLostDevice();
	Log("Released!");
}



void HQRenderTargetManagerD3D9::OnLostDevice()
{
	HQItemManager<HQBaseRenderTargetView>::Iterator ite;
	this->renderTargets.GetIterator(ite);
	while(!ite.IsAtEnd())
	{
		ite->OnLostDevice();
		++ite;
	}

	HQItemManager<HQBaseDepthStencilBufferView>::Iterator ite2;
	this->depthStencilBuffers.GetIterator(ite2);
	while(!ite2.IsAtEnd())
	{
		ite2->OnLostDevice();
		++ite2;
	}

	SafeRelease(this->pD3DBackBuffer);
	SafeRelease(this->pD3DDSBuffer);
}

void HQRenderTargetManagerD3D9::OnResetDevice()
{
	this->renderTargetWidth = g_pD3DDev->GetWidth();
	this->renderTargetHeight = g_pD3DDev->GetHeight();

	HQItemManager<HQBaseRenderTargetView>::Iterator ite;
	this->renderTargets.GetIterator(ite);
	while (!ite.IsAtEnd())
	{
		ite->OnResetDevice();
		++ite;
	}

	HQItemManager<HQBaseDepthStencilBufferView>::Iterator ite2;
	this->depthStencilBuffers.GetIterator(ite2);
	while (!ite2.IsAtEnd())
	{
		ite2->OnResetDevice();
		++ite2;
	}

	pD3DDevice->GetRenderTarget(0, &pD3DBackBuffer);
	if (pD3DDevice->GetDepthStencilSurface(&pD3DDSBuffer) == D3DERR_NOTFOUND)
		pD3DDSBuffer = NULL;

	if (this->currentActiveRTGroup != NULL)
	{
		HQSharedPtr<HQBaseRenderTargetView> pBuffer = HQSharedPtr<HQBaseRenderTargetView>::null;
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

HQReturnVal HQRenderTargetManagerD3D9::CreateRenderTargetTexture(hq_uint32 width, hq_uint32 height, hq_uint32 arraySize, bool hasMipmaps,
											   HQRenderTargetFormat format, HQMultiSampleType multisampleType, 
											   HQTextureType textureType, 
											   HQRenderTargetView **pRenderTargetID_Out, 
											   HQTexture **pTextureID_Out)
{
	switch (textureType)
	{
	case HQ_TEXTURE_2D:
		break;
	case HQ_TEXTURE_CUBE:
		break;
	default:
		Log("CreateRenderTargetTexture() failed : unsuported texture type=%u!", (hquint32)textureType);
		return HQ_FAILED;
	}

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
	HQSharedPtr<HQBaseTexture> pNewTex = this->pTextureManager->AddEmptyTexture(textureType , NULL);
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
									numMipmaps , pNewTex
									);
	if (pNewRenderTarget == NULL || pNewTex->pData == NULL)
	{
		pTextureManager->RemoveTexture(pNewTex.GetRawPointer());
		return HQ_FAILED_MEM_ALLOC;
	}

	
	if(!this->renderTargets.AddItem(pNewRenderTarget , pRenderTargetID_Out))
	{
		delete pNewRenderTarget;
		pTextureManager->RemoveTexture(pNewTex.GetRawPointer());
		return HQ_FAILED_MEM_ALLOC;
	}
	
	if(pTextureID_Out != NULL)
		*pTextureID_Out = pNewTex.GetRawPointer();

	return HQ_OK;
}


HQReturnVal HQRenderTargetManagerD3D9::CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
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


HQReturnVal HQRenderTargetManagerD3D9::CreateRenderTargetGroupImpl(
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

	HQBaseRenderTargetGroup* newGroup = new HQBaseRenderTargetGroup(numRenderTargets);
	
	for (hq_uint32 i = 0 ; i < numRenderTargets ; ++i)
	{
		
		HQSharedPtr<HQBaseRenderTargetView> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDescs[i].renderTargetID);
		if (pRenderTarget == NULL)
		{
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
	if (pDepthStencilBuffer != NULL)
	{
		if (pDepthStencilBuffer->width < newGroup->commonWidth || pDepthStencilBuffer->height < newGroup->commonHeight)
		{
			delete newGroup;

			this->Log("Error : CreateRenderTargetGroupImpl() failed because depth stencil buffer is too small!");
			return HQ_FAILED_DEPTH_STENCIL_BUFFER_TOO_SMALL;
		}
	}

	newGroup->pDepthStencilBuffer = pDepthStencilBuffer;

	*ppRenderTargetGroupOut = newGroup;

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D9::ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& group)
{
	return ActiveRenderTargetsRawPtrImpl(group.GetRawPointer());
}

HQReturnVal HQRenderTargetManagerD3D9::ActiveRenderTargetsRawPtrImpl(HQBaseRenderTargetGroup* group)
{
	if (g_pD3DDev->GetFlags() & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;

	if (group == NULL)
	{
		//active default back buffer and depth stencil buffer
		this->InvalidateRenderTargets();
		return HQ_OK;
	}
	
	for (hq_uint32 i = 0 ; i < group->numRenderTargets ; ++i)
	{
		
		HQSharedPtr<HQBaseRenderTargetView> pRenderTarget = group->renderTargets[i].pRenderTarget;
		if (pRenderTarget == NULL)
		{
			pD3DDevice->SetRenderTarget(i , NULL);
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseRenderTargetView> ::null;
			continue;
		}
		
		if (pRenderTarget == this->activeRenderTargets[i].pRenderTarget)//no change 
		{
			if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
			{
				if (group->renderTargets[i].cubeFace != this->activeRenderTargets[i].cubeFace)//different cube face
				{
					LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[ group->renderTargets[i].cubeFace ]);
				}//if (renderTargetDescs[i].cubeFace != this->activeRenderTargets[i].cubeFace)
			}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		}//pRenderTarget == this->activeRenderTargets[i].pRenderTarget
		else//different render target
		{
			if (pRenderTarget->IsTexture())
			{
				LPDIRECT3DSURFACE9* pD3DSurfaces = (LPDIRECT3DSURFACE9*)pRenderTarget->GetData();

				if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[ group->renderTargets[i].cubeFace ]);
				else
					pD3DDevice->SetRenderTarget( i , pD3DSurfaces[0]);
			}//if (pRenderTarget->IsTexture())
		}//else of if (pRenderTarget == this->activeRenderTargets[i].pRenderTarget)
		
		this->activeRenderTargets[i].pRenderTarget = pRenderTarget;
		this->activeRenderTargets[i].cubeFace = group->renderTargets[i].cubeFace;
	}//for (i)

	
	
	//deactive other render targets
	for (hq_uint32 i = group->numRenderTargets ; i < this->numActiveRenderTargets ; ++i)
	{
		if (this->activeRenderTargets[i].pRenderTarget != NULL)
		{
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseRenderTargetView> ::null;
			pD3DDevice->SetRenderTarget(i , NULL);
		}
	}

	this->numActiveRenderTargets = group->numRenderTargets;
	
	//active depth stencil buffer
	HQSharedPtr<HQBaseDepthStencilBufferView> pDepthStencilBuffer = group->pDepthStencilBuffer;
	if (this->currentActiveRTGroup == NULL || this->pActiveDepthStencilBuffer != pDepthStencilBuffer)
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
	this->renderTargetWidth = group->commonWidth;
	this->renderTargetHeight = group->commonHeight;

	this->ResetViewPort();
	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerD3D9::ClearRenderTargets(hquint32 numRTsToClear)
{
	if (!this->IsUsingDefaultFrameBuffer())
	{
		HQBaseRenderTargetGroup* group = static_cast <HQBaseRenderTargetGroup*> (this->currentActiveRTGroup);
		
		//deactive render targets starting from {numRTsToClear}
		for (hq_uint32 i = numRTsToClear; i < this->numActiveRenderTargets; ++i)
		{
			if (this->activeRenderTargets[i].pRenderTarget != NULL)
			{
				this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseRenderTargetView> ::null;
				pD3DDevice->SetRenderTarget(i, NULL);
			}
		}

		//clear
		pD3DDevice->Clear(0, 0, D3DCLEAR_TARGET, g_pD3DDev->GetD3DClearColor(), 0, 0);

		//reactivate inactive render targets
		this->ActiveRenderTargetsRawPtrImpl(group);
	}
	return HQ_OK;
}

void HQRenderTargetManagerD3D9::InvalidateRenderTargets()
{
	this->activeRenderTargets[0].pRenderTarget = HQSharedPtr<HQBaseRenderTargetView> ::null;

	for (hq_uint32 i = 1 ; i < this->numActiveRenderTargets ; ++i)
	{
		pD3DDevice->SetRenderTarget( i , NULL);
		this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseRenderTargetView> ::null;
	}

	this->ResetToDefaultFrameBuffer();
}

void HQRenderTargetManagerD3D9::ResetToDefaultFrameBuffer()
{
	//active default back buffer
	pD3DDevice->SetRenderTarget(0 , pD3DBackBuffer);
	//active default depth stencil buffer
	pD3DDevice->SetDepthStencilSurface(pD3DDSBuffer);

	this->pActiveDepthStencilBuffer = HQSharedPtr<HQBaseDepthStencilBufferView> ::null;
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
		g_pD3DDev->SetViewport(viewport);
}

HQReturnVal HQRenderTargetManagerD3D9::GenerateMipmaps(HQRenderTargetView* renderTargetTextureID)
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
