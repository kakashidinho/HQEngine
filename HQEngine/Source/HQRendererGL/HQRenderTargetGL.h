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
#if !defined APPLE && !defined IOS && !defined ANDROID
#include <GL/glew.h>
#elif defined __APPLE__
#import "Apple.h"
#else
#include "AndroidGLES.h"
#endif

struct GLdepthStencilFormat
{
	GLenum depthFormat;
	GLenum stencilFormat;
};

/*----------frame buffer object implementation version---------------*/
class HQRenderTargetManagerFBO : public HQBaseRenderTargetManager
{
private:
	GLuint framebuffer;
#ifdef IOS
	GLuint defaultFBO;
#endif
	GLenum *buffers;
	
	void DetachSlot(hq_uint32 i);
public:
	HQRenderTargetManagerFBO(
#ifdef IOS
						   GLuint defaultFBO,
#endif
						   hq_uint32 maxActiveRenderTarget,
						   HQBaseTextureManager *pTexMan,
						   HQLogStream *logFileStream , bool flushLog);
	
	~HQRenderTargetManagerFBO();
	
	/*----------helper methods-------------*/
	static GLint GetGLInternalFormat(HQRenderTargetFormat format);
	static void GetGLImageFormat(GLint internalFormat , GLenum &format , GLenum &type);
	static GLdepthStencilFormat GetGLFormat(HQDepthStencilFormat format);
	
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
	
	void ActiveDefaultFrameBuffer();

	HQReturnVal RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList);

	HQReturnVal RemoveRenderTarget(unsigned int renderTargetID);
	void RemoveAllRenderTarget();

	HQReturnVal RemoveDepthStencilBuffer(unsigned int bufferID);
	void RemoveAllDepthStencilBuffer();
};


#endif
