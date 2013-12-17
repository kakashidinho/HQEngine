/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _RENDER_TARGET_MANAGER_D3D_H_
#define _RENDER_TARGET_MANAGER_D3D_H_

#include "../BaseImpl/HQRenderTargetManagerBaseImpl.h"
#include "d3d11.h"


class HQRenderTargetManagerD3D11 : public HQBaseRenderTargetManager
{
private:

	ID3D11Device * pD3DDevice;
	ID3D11DeviceContext *pD3DContext;
	ID3D11RenderTargetView* pD3DBackBuffer;//default back buffer
	ID3D11DepthStencilView* pD3DDSBuffer;//default depth stencil buffer

	ID3D11RenderTargetView *renderTargetViews[8];//current active render target views
	ID3D11DepthStencilView *pDepthStencilView;//current active depth stencil view
	
public:
	HQRenderTargetManagerD3D11(ID3D11Device * pD3DDevice , 
		ID3D11DeviceContext *pD3DContext,
		ID3D11RenderTargetView* pD3DBackBuffer,//default back buffer
		ID3D11DepthStencilView* pD3DDSBuffer,//default depth stencil buffer
		HQBaseTextureManager *pTexMan,
		HQLogStream* logFileStream , bool flushLog);
	
	~HQRenderTargetManagerD3D11();
	
	inline ID3D11RenderTargetView * const *GetRenderTargetViews() {return renderTargetViews;}
	inline ID3D11DepthStencilView * GetDepthStencilView() {return pDepthStencilView;}
	
	void SetDefaultRenderTargetView(ID3D11RenderTargetView *pD3DBackBuffer);
	void SetDefaultDepthStencilView(ID3D11DepthStencilView *pD3DDSBuffer);

	
	//generate full range mipmaps for render target texture <renderTargetTextureID>
	HQReturnVal GenerateMipmaps(hq_uint32 renderTargetTextureID);

	//create render target texture
	//<pRenderTargetID_Out> - will store ID of newly created render target
	//<pTextureID_Out> - will store ID of texture in material manager
	//Note :
	//-if <textureType> = HQ_TEXTURE_CUBE , new texture will be created with size <width> x <width>
	//-return HQ_FAILED_FORMAT_NOT_SUPPORT if <format> is not supported
	//-return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if <multisampleType> is not supported
	HQReturnVal CreateRenderTargetTexture(hq_uint32 width , hq_uint32 height,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType,
								  hq_uint32 *pRenderTargetID_Out,
								  hq_uint32 *pTextureID_Out);
	//create custom depth stencil buffer
	//return HQ_FAILED_FORMAT_NOT_SUPPORT if <format> is not supported
	//return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if <multisampleType> is not supported
	HQReturnVal CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										hq_uint32 *pDepthStencilBufferID_Out);
	//Set new main render target specified in <renderTargetDescs> .
	//Depth stencil buffer <depthStencilBufferID >will be used as main depth stencil buffer.
	//If ID of render target is invalid ,  default frame buffer and depth stencil buffer will be used as main render target and depth stencil buffer.
	//Else if <depthStencilBufferID> is not a valid id , no depth stencil buffer will be used.
	//Depth stencil buffer and render target must have the same multimsample type and width,height of depth stencil buffer must be larger than or equal to render target's.
	//If current viewport area can't fit in render area ,viewport will be resized to the same size as render target
	HQReturnVal ActiveRenderTarget(const HQRenderTargetDesc &renderTargetDesc , 
									hq_uint32 depthStencilBufferID 
								   );
	//Set new main render targets specified in <renderTargetDescs> array.
	//Depth stencil buffer <depthStencilBufferID >will be used as main depth stencil buffer.
	//If <renderTargetDescs> = NULL or <numRenderTargets> = 0 or all render target IDs is not valid , default frame buffer and depth stencil buffer will be used as main render target and depth stencil buffer.
	//If ID of render target in (ith) element of <renderTargetDescs> array is invalid , render target at (ith) slot will be deactivated
	//Else if <depthStencilBufferID> is not a valid id , no depth stencil buffer will be used.
	//Depth stencil buffer and render target must have the same multimsample type and width,height of depth stencil buffer must be larger than or equal render target.
	//All render targets must have compatible formats and the same width x height.
	//return HQ_FAILED if <numRenderTargets> is larger than value retrieved by method GetMaxActiveRenderTargets()
	HQReturnVal ActiveRenderTargets(const HQRenderTargetDesc *renderTargetDescs , 
							hq_uint32 depthStencilBufferID ,
							hq_uint32 numRenderTargets = 1//number of render targers
							);

	void ActiveDefaultFrameBuffer();

	HQReturnVal RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList);

	
	static DXGI_FORMAT GetD3DFormat(HQRenderTargetFormat format);
	static DXGI_FORMAT GetD3DFormat(HQDepthStencilFormat format);
};


#endif
