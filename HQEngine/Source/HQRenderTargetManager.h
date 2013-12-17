/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_RENDER_TARGET_MANAGER_H_
#define _HQ_RENDER_TARGET_MANAGER_H_

#include "HQRendererCoreType.h"
#include "HQReferenceCountObj.h"

typedef HQReferenceCountObj HQSavedActiveRenderTargets;

/*-------------------------------------------------------
Important : 
-multisample render target and depth stencil 
buffer may not work together.
-Direct3d 10 & 11 : depth stencil buffer doesn't work 
with cube texture render target yet.It can cause driver 
crash.cube render taret texture can't active as the same time
as 2d render target texture
-------------------------------------------------------*/

class HQRenderTargetManager
{
protected:
	virtual ~HQRenderTargetManager() {};
public:
	///
	///generate full range mipmaps for render target texture {renderTargetTextureID}
	///this operation may do nothing if mipmaps generation for render target texture is not supported
	///this method must not be called when render target is active , or it can make application crash
	///
	virtual HQReturnVal GenerateMipmaps(hq_uint32 renderTargetTextureID) = 0;

	///
	///create render target texture. 
	///{pRenderTargetID_Out} - will store ID of newly created render target. 
	///{pTextureID_Out} - will store ID of texture in material manager. 
	///{hasMipmaps} - this texture has full range mipmap or not. 
	///Note : 
	///-if {textureType} = HQ_TEXTURE_CUBE , new texture will be created with size {width} x {width}. 
	///-openGL ES 2.0 device always create texture with full range mipmap levels. 
	///-return HQ_FAILED_FORMAT_NOT_SUPPORT if {format} is not supported. 
	///-return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if {multisampleType} is not supported. 
	///
	virtual HQReturnVal CreateRenderTargetTexture(hq_uint32 width , hq_uint32 height,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType,
								  hq_uint32 *pRenderTargetID_Out,
								  hq_uint32 *pTextureID_Out) = 0;
	///
	///create custom depth stencil buffer. 
	///return HQ_FAILED_FORMAT_NOT_SUPPORT if {format} is not supported. 
	///return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if {multisampleType} is not supported. 
	///
	virtual HQReturnVal CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										hq_uint32 *pDepthStencilBufferID_Out) = 0;
	
	///
	///Set new main render target specified in {renderTargetDescs} .All other render targets will be deactivated. 
	///Depth stencil buffer {depthStencilBufferID} will be used as main depth stencil buffer. 
	///If ID of render target is invalid ,  default frame buffer and depth stencil buffer will be used as main render target and depth stencil buffer. 
	///Else if {depthStencilBufferID} is not a valid id , no depth stencil buffer will be used. 
	///Depth stencil buffer and render target must have the same multimsample type and width,height of depth stencil buffer must be larger than or equal to render target's. 
	///If current viewport area can't fit in render area ,viewport will be resized to the same size as render target. 
	///
	virtual HQReturnVal ActiveRenderTarget(const HQRenderTargetDesc &renderTargetDesc , 
									hq_uint32 depthStencilBufferID 
								   )= 0;
	
	///
	///Set new main render targets specified in {renderTargetDescs} array. 
	///Depth stencil buffer {depthStencilBufferID} will be used as main depth stencil buffer. 
	///If ID of render target in (ith) element of {renderTargetDescs} array is invalid , render target at (ith) slot will be deactivated. 
	///If {renderTargetDescs} = NULL or {numRenderTargets} = 0 or all render target IDs is not valid , default frame buffer and depth stencil buffer will be used as main render target and depth stencil buffer. 
	///Else if {depthStencilBufferID} is not a valid id , no depth stencil buffer will be used. 
	///Depth stencil buffer and render target must have the same multimsample type and width,height of depth stencil buffer must be larger than or equal to render target's. 
	///Render targets must have compatible formats and same width and height. 
	///First render target slot must be a valid render target.Otherwise it can cause undefined behavior.Some render device may switch to default frame buffer. 
	///If current viewport area can't fit in render area ,viewport will be resized to the same size as render target. 
	///return HQ_FAILED if [numRenderTargets} is larger than value retrieved by method GetMaxActiveRenderTargets()
	///
	virtual HQReturnVal ActiveRenderTargets(const HQRenderTargetDesc *renderTargetDescs , 
									hq_uint32 depthStencilBufferID ,
								   hq_uint32 numRenderTargets = 1//number of render targers
								   )= 0;

	///
	///Save the current active render targets and active depth buffer list so that they can be reactive by calling RestoreRenderTargets(). 
	///The returned pointer must be released before destroying the render device
	///
	virtual HQSavedActiveRenderTargets* CreateAndSaveRenderTargetsList() = 0;

	///
	///save the current active render targets and active depth buffer list so that they can be reactive by calling RestoreRenderTargets()
	///
	virtual HQReturnVal SaveRenderTargetsList(HQSavedActiveRenderTargets* savedList) = 0;

	///
	///restore from the saved active render targets and depth buffer list.
	///
	virtual HQReturnVal RestoreRenderTargets(const HQSavedActiveRenderTargets *savedList) = 0;
	
	///
	///If render target is texture , it also will be removed from texture manager
	///
	virtual HQReturnVal RemoveRenderTarget(hq_uint32 renderTargetID) =0;
	virtual void RemoveAllRenderTarget() = 0;
	
	virtual HQReturnVal RemoveDepthStencilBuffer(hq_uint32 depthStencilBufferID) = 0;
	virtual void RemoveAllDepthStencilBuffer() = 0;
};
#endif
