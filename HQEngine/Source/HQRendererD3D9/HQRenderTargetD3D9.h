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
									hq_uint32 depthStencilBufferID  );
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

	void InvalidateRenderTargets();

	HQReturnVal RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList);
};


#endif