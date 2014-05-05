/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _RENDER_TARGET_MANAGER_D3D_H_
#define _RENDER_TARGET_MANAGER_D3D_H_

#include "../BaseImpl/HQBaseImplCommon.h"
#include "../BaseImpl/HQRenderTargetManagerBaseImpl.h"
#include "d3d9.h"


class HQRenderTargetManagerD3D9 : public HQBaseRenderTargetManager 
{
private:

	LPDIRECT3DDEVICE9 pD3DDevice;
	LPDIRECT3DSURFACE9 pD3DBackBuffer;//default back buffer
	LPDIRECT3DSURFACE9 pD3DDSBuffer;//default depth stencil buffer

	HQSharedPtr<HQBaseDepthStencilBufferView> pActiveDepthStencilBuffer;//current active depth stencil buffer
	HQRenderTargetInfo *activeRenderTargets;//active Render targets

	hquint32 numActiveRenderTargets;//number of currently active render targets

	void ResetToDefaultFrameBuffer();
	void ResetViewPort();
public:
	HQRenderTargetManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice , 
		hq_uint32 maxActiveRenderTarget,
		HQBaseTextureManager *pTexMan,
		HQLogStream *logFileStream , bool flushLog);
	
	~HQRenderTargetManagerD3D9();

	void OnLostDevice();
	void OnResetDevice();

	static D3DFORMAT GetD3DFormat(HQRenderTargetFormat format);
	static D3DFORMAT GetD3DFormat(HQDepthStencilFormat format);

	//generate full range mipmaps for render target texture <renderTargetTextureID>
	HQReturnVal GenerateMipmaps(HQRenderTargetView* renderTargetTextureID);

	//create render target texture
	//<pRenderTargetID_Out> - will store ID of newly created render target
	//<pTextureID_Out> - will store ID of texture in material manager
	//Note :
	//-if <textureType> = HQ_TEXTURE_CUBE , new texture will be created with size <width> x <width>
	//-return HQ_FAILED_FORMAT_NOT_SUPPORT if <format> is not supported
	//-return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if <multisampleType> is not supported
	HQReturnVal CreateRenderTargetTexture(hq_uint32 width , hq_uint32 height,
								  hq_uint32 arraySize,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType,
								  HQRenderTargetView** pRenderTargetID_Out,
								  HQTexture** pTextureID_Out);
	//create custom depth stencil buffer
	//return HQ_FAILED_FORMAT_NOT_SUPPORT if <format> is not supported
	//return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if <multisampleType> is not supported
	HQReturnVal CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										HQDepthStencilBufferView **pDepthStencilBufferID_Out);
	
	///
	///see HQBaseRenderTargetManager base class
	///
	HQReturnVal ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& group);
	///
	///see HQBaseRenderTargetManager base class
	///
	HQReturnVal CreateRenderTargetGroupImpl(const HQRenderTargetDesc *renderTargetDescs , 
									HQDepthStencilBufferView* depthStencilBufferID,
									hq_uint32 numRenderTargets,//number of render targers
									HQBaseRenderTargetGroup **ppRenderTargetGroupOut
									) ;

	void InvalidateRenderTargets();

};


#endif
