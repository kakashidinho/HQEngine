/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
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

	ID3D11RenderTargetView **renderTargetViews;//current active render target views
	ID3D11DepthStencilView *pDepthStencilView;//current active depth stencil view

	//-------------graphics UAV slots--------------------
	ID3D11UnorderedAccessView *pUAVSlots[D3D11_PS_CS_UAV_REGISTER_COUNT];
	UINT UAVInitialCounts[D3D11_PS_CS_UAV_REGISTER_COUNT];
	hqint32 minUsedUAVSlot;//min used UAV slot
	hqint32 maxUsedUAVSlot;//max used UAV slot. -1 if no slot is used
	
	hquint32 flags;
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
								  HQTexture **pTextureID_Out);
	//create custom depth stencil buffer
	//return HQ_FAILED_FORMAT_NOT_SUPPORT if <format> is not supported
	//return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if <multisampleType> is not supported
	HQReturnVal CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										HQDepthStencilBufferView **pDepthStencilBufferID_Out);
	

	void ActiveDefaultFrameBuffer();

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


	HQReturnVal SetUAVForGraphicsShader(hquint32 slot, ID3D11UnorderedAccessView * UAV);

	void OnDrawOrDispatch();//this is called before drawing

	static DXGI_FORMAT GetD3DFormat(HQRenderTargetFormat format);
	static DXGI_FORMAT GetD3DFormat(HQDepthStencilFormat format);
};


#endif
