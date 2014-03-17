/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _RENDER_TARGET_MANAGER_D3D_H_
#define _RENDER_TARGET_MANAGER_D3D_H_

#include "../BaseImpl/HQRenderTargetManagerBaseImpl.h"
#include "glHeaders.h"

struct GLdepthStencilFormat
{
	GLenum depthFormat;
	GLenum stencilFormat;
};

/*------------base class--------*/
class HQBaseRenderTargetManagerGL : public HQBaseRenderTargetManager, public HQResetable
{
public:
	HQBaseRenderTargetManagerGL(hq_uint32 maxActiveRenderTargets, 
							  HQBaseTextureManager *pTexMan,
							  HQLogStream* logFileStream , const char *logPrefix ,
							  bool flushLog)
		:HQBaseRenderTargetManager(
			maxActiveRenderTargets, pTexMan, 
			logFileStream, logPrefix, flushLog
		)
	{
	}
};

/*----------frame buffer object implementation version---------------*/
class HQRenderTargetManagerFBO : public HQBaseRenderTargetManagerGL
{
private:
#ifdef IOS
	GLuint defaultFBO;
#endif
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
	static void GetGLImageFormat(HQRenderTargetFormat hqformat, GLint& internalFormat , GLenum &format , GLenum &type);
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
	///
	///see HQBaseRenderTargetManager base class
	///
	HQReturnVal ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& group);
	///
	///see HQBaseRenderTargetManager base class
	///
	HQReturnVal CreateRenderTargetGroupImpl(const HQRenderTargetDesc *renderTargetDescs , 
									hq_uint32 depthStencilBufferID ,
									hq_uint32 numRenderTargets,//number of render targers
									HQBaseRenderTargetGroup **ppRenderTargetGroupOut
									) ;
	
	void ActiveDefaultFrameBuffer();

	//device lost handling methods
	virtual void OnLost();
	virtual void OnReset();
};


#endif
