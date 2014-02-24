/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQDeviceGL.h"


#ifndef IOS
#	define defaultFBO 0
#	if !defined APPLE

#ifdef glGenFramebuffers
#undef glGenFramebuffers
#endif
#ifdef glBindFramebuffer
#undef glBindFramebuffer
#endif
#ifdef glGenRenderbuffers
#undef glGenRenderbuffers
#endif
#ifdef glBindRenderbuffer
#undef glBindRenderbuffer
#endif
#ifdef glGenerateMipmap
#undef glGenerateMipmap
#endif
#ifdef glRenderbufferStorage
#undef glRenderbufferStorage
#endif
#ifndef GLES
#ifdef glRenderbufferStorageMultisample
#undef glRenderbufferStorageMultisample
#endif
#endif
#ifdef glDeleteFramebuffers
#undef glDeleteFramebuffers
#endif
#ifdef glDeleteRenderbuffers
#undef glDeleteRenderbuffers
#endif
#ifdef glCheckFramebufferStatus
#undef glCheckFramebufferStatus
#endif
#ifdef glFramebufferRenderbuffer
#undef glFramebufferRenderbuffer
#endif
#ifdef glFramebufferTexture2D
#undef glFramebufferTexture2D
#endif

#		ifdef ANDROID
#			define glGenFramebuffers  android_glGenFramebuffers
#			define glBindFramebuffer  android_glBindFramebuffer
#			define glGenRenderbuffers  android_glGenRenderbuffers
#			define glBindRenderbuffer  android_glBindRenderbuffer
#			define glGenerateMipmap  android_glGenerateMipmap
#			define glRenderbufferStorage android_glRenderbufferStorage
#			define glDeleteFramebuffers android_glDeleteFramebuffers
#			define glDeleteRenderbuffers android_glDeleteRenderbuffers
#			define glCheckFramebufferStatus  android_glCheckFramebufferStatus
#			define glFramebufferRenderbuffer  android_glFramebufferRenderbuffer
#			define glFramebufferTexture2D  android_glFramebufferTexture2D
#		else
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers;
PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer;
PFNGLGENERATEMIPMAPPROC glGenerateMipmap;
PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage;
PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC glRenderbufferStorageMultisample;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus;
PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
#		endif
#	endif
#endif

/*-------------------------*/

struct HQDepthStencilBufferGL : public HQBaseCustomRenderBuffer
{
	HQDepthStencilBufferGL(hq_uint32 _width ,hq_uint32 _height ,
							HQMultiSampleType _multiSampleType,
							GLdepthStencilFormat format)
			: HQBaseCustomRenderBuffer(_width , _height ,
									  _multiSampleType)
	{
		this->depthStencilName[0] = this->depthStencilName[1] = 0;
		this->format = format;
	}
	~HQDepthStencilBufferGL()
	{
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())
		{
#endif
			if (this->format.stencilFormat != 0 && this->format.depthFormat !=0)
				glDeleteRenderbuffers(2 , this->depthStencilName);
			else if (this->format.depthFormat != 0)
				glDeleteRenderbuffers(1 , this->depthStencilName);
			else if (this->format.stencilFormat != 0)
				glDeleteRenderbuffers(1 , &this->depthStencilName[1]);
#if defined DEVICE_LOST_POSSIBLE
		}
#endif
	}
	void *GetData()
	{
		return depthStencilName;
	}

	HQReturnVal Init()
	{
		glGetError();//reset error flag
		if (this->format.stencilFormat != 0 && this->format.depthFormat !=0)
			glGenRenderbuffers(2 , this->depthStencilName);
		else if (this->format.depthFormat != 0)
		{
			glGenRenderbuffers(1 , this->depthStencilName);
			if (this->format.depthFormat == GL_DEPTH24_STENCIL8_EXT)//packed depth stencil buffer
				this->depthStencilName[1] = this->depthStencilName[0];
		}
		else if (this->format.stencilFormat != 0)
			glGenRenderbuffers(1 , &this->depthStencilName[1]);

		//create depth buffer
		if (this->format.depthFormat)
		{
			glBindRenderbuffer(GL_RENDERBUFFER , this->depthStencilName[0]);
			if ((hq_uint32 )this->multiSampleType > 0)
				glRenderbufferStorageMultisample(GL_RENDERBUFFER , (GLsizei)this->multiSampleType, this->format.depthFormat , this->width , this->height);
			else
				glRenderbufferStorage(GL_RENDERBUFFER, this->format.depthFormat , this->width , this->height);
		}
		//create stencil buffer
		if (this->format.stencilFormat)
		{
			glBindRenderbuffer(GL_RENDERBUFFER , this->depthStencilName[1]);
			if ((hq_uint32 )this->multiSampleType > 0)
				glRenderbufferStorageMultisample(GL_RENDERBUFFER , (GLsizei)this->multiSampleType, this->format.stencilFormat , this->width , this->height);
			else
				glRenderbufferStorage(GL_RENDERBUFFER , this->format.stencilFormat , this->width , this->height);
		}
		GLenum err = glGetError();
		if (GL_OUT_OF_MEMORY == err)
			return HQ_FAILED_MEM_ALLOC;
		else if (err != GL_NO_ERROR)
			return HQ_FAILED;

		return HQ_OK;
	}

	GLdepthStencilFormat format;
	GLuint depthStencilName[2];//depth buffer name and stencil buffer name
};

struct HQRenderTargetTextureGL : public HQBaseRenderTargetTexture
{
	HQRenderTargetTextureGL(hq_uint32 _width ,hq_uint32 _height ,
							HQMultiSampleType _multiSampleType,
							GLint internalFormat , hq_uint32 _numMipmaps,
							hq_uint32 _textureID , HQSharedPtr<HQTexture> _pTex)
		: HQBaseRenderTargetTexture(_width , _height ,
							_multiSampleType, _numMipmaps,
							_textureID , _pTex)
	{
		this->internalFormat = internalFormat;

	}
	~HQRenderTargetTextureGL()
	{
	}

	void *GetData()
	{
		return this->pTexture->pData;
	}

	/*---------------not support multisample texture yet------*/
	HQReturnVal Init()
	{
		HQTextureManagerGL *pTextureMan = (HQTextureManagerGL *)g_pOGLDev->GetTextureManager();
		GLenum format, type;//get texture's pixel format and pixel data type
		HQRenderTargetManagerFBO::GetGLImageFormat(this->internalFormat , format , type);
#ifndef GLES
		GLclampf priority = 1.0f;
		glPrioritizeTextures(1 , (GLuint*)this->pTexture->pData , &priority);
#endif
		glGetError();//reset error flag

		GLuint *pTextureGL = (GLuint*)this->pTexture->pData;
		switch(pTexture->type)
		{
		case HQ_TEXTURE_2D:
			{
				glBindTexture(GL_TEXTURE_2D , *pTextureGL);
				int w = this->width;
				int h = this->height;
				for(hq_uint32 i = 0; i < this->numMipmaps ; ++i)
				{
					glTexImage2D(GL_TEXTURE_2D , i , this->internalFormat , w , h,
						0 , format , type , NULL);

					if (w > 1)
						w >>= 1;
					if (h > 1)
						h >>= 1;
				}

				//re-bind old texture
				glBindTexture(GL_TEXTURE_2D, pTextureMan->GetActiveTextureUnitInfo().GetTexture2DGL());
			}
			break;
		case HQ_TEXTURE_CUBE:

			glBindTexture(GL_TEXTURE_CUBE_MAP , *pTextureGL);

			for (int face = 0; face < 6 ; ++face)
			{
				int w = this->width;
				for(hq_uint32 i = 0; i < this->numMipmaps ; ++i)
				{
					glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face , i , this->internalFormat , w , w,
						0 , format , type , NULL);

					if (w > 1)
						w >>= 1;
				}
			}
			//re-bind old texture
			glBindTexture(GL_TEXTURE_CUBE_MAP, pTextureMan->GetActiveTextureUnitInfo().GetTextureCubeGL());
			break;
		}

		if (glGetError() != GL_NO_ERROR)
			return HQ_FAILED;
		return HQ_OK;
	}

	GLint internalFormat;
};

GLint HQRenderTargetManagerFBO::GetGLInternalFormat(HQRenderTargetFormat format)
{
	switch(format)
	{
	case HQ_RTFMT_R_FLOAT32:
		if (GLEW_VERSION_3_0)
			return GL_R32F;
		if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
				return GL_R32F;
			return GL_LUMINANCE32F_ARB;
		}
		break;
	case HQ_RTFMT_R_FLOAT16:
		if (GLEW_VERSION_3_0)
			return GL_R16F;
		if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
				return GL_R16F;
			return GL_LUMINANCE16F_ARB;
		}
		break;
	case HQ_RTFMT_RGBA_32:
		return GL_RGBA8;
	case HQ_RTFMT_R_UINT8:
		if (GLEW_VERSION_3_0)
			return GL_R8UI;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R8UI;
			return GL_LUMINANCE8UI_EXT;
		}
	case HQ_RTFMT_A_UINT8:
		if (GLEW_EXT_texture_integer)
		{
			return GL_LUMINANCE8UI_EXT;
		}
	case HQ_RTFMT_RGBA_FLOAT64:
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_float)
			return GL_RGBA16F;
	case HQ_RTFMT_RG_FLOAT32:
		if (GLEW_VERSION_3_0)
			return GL_RG16F;
		if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
				return GL_RG16F;
		}
	case HQ_RTFMT_RGBA_FLOAT128:
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_float)
			return GL_RGBA32F;
	case HQ_RTFMT_RG_FLOAT64:
		if (GLEW_VERSION_3_0)
			return GL_RG32F;
		if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
				return GL_RG32F;
		}
	}
	return 0;
}

void HQRenderTargetManagerFBO::GetGLImageFormat(GLint internalFormat , GLenum &format , GLenum &type)
{
	switch (internalFormat){
		case GL_LUMINANCE32F_ARB: case GL_LUMINANCE16F_ARB:
			format = GL_LUMINANCE;
			type = GL_FLOAT;
			break;
		case GL_LUMINANCE8UI_EXT:
			format = GL_LUMINANCE_INTEGER_EXT ;
			type = GL_UNSIGNED_BYTE;
			break;
		case GL_R8UI:
			format = GL_RED_INTEGER ;
			type = GL_UNSIGNED_BYTE;
			break;
		case GL_R32F: case GL_R16F:
			format = GL_RED;
			type = GL_FLOAT;
			break;
		case GL_RG32F: case GL_RG16F:
			format = GL_RG;
			type = GL_FLOAT;
			break;
		case GL_RGBA32F: case GL_RGBA16F:
			format = GL_RGBA;
			type = GL_FLOAT;
			break;
		default:
			format = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
	}
}

GLdepthStencilFormat HQRenderTargetManagerFBO::GetGLFormat(HQDepthStencilFormat format)
{
	GLdepthStencilFormat dsFormat;
	dsFormat.depthFormat = 0;
	dsFormat.stencilFormat = 0;
	switch(format)
	{
	case HQ_DSFMT_DEPTH_16:
		dsFormat.depthFormat = GL_DEPTH_COMPONENT16;
		dsFormat.stencilFormat = 0;
		break;
	case HQ_DSFMT_DEPTH_24:
		dsFormat.depthFormat = GL_DEPTH_COMPONENT24;
		dsFormat.stencilFormat = 0;
		break;
	case HQ_DSFMT_DEPTH_32:
		dsFormat.depthFormat = GL_DEPTH_COMPONENT32;
		dsFormat.stencilFormat = 0;
		break;
	case HQ_DSFMT_STENCIL_8:
		dsFormat.depthFormat = 0;
		dsFormat.stencilFormat = GL_STENCIL_INDEX8;
		break;
	case HQ_DSFMT_DEPTH_24_STENCIL_8:
		if (GLEW_EXT_packed_depth_stencil)//depth and stencil channel can be packed in 1 buffer
		{
			dsFormat.depthFormat = GL_DEPTH24_STENCIL8_EXT;
			dsFormat.stencilFormat = 0;
		}
		else
		{
			dsFormat.depthFormat = GL_DEPTH_COMPONENT24;
			dsFormat.stencilFormat = GL_STENCIL_INDEX8;
		}
		break;
	}
	return dsFormat;
}

HQRenderTargetManagerFBO::HQRenderTargetManagerFBO(
#ifdef IOS
											   GLuint defaultFBO,
#endif
											   hq_uint32 maxActiveRenderTargets,
											   HQBaseTextureManager *pTexMan,
											   HQLogStream *logFileStream ,  bool flushLog)
						: HQBaseRenderTargetManager(maxActiveRenderTargets ,
										 pTexMan , logFileStream ,
										 "GL Render Target Manager :" ,
										 flushLog)
{

#if !defined APPLE && !defined IOS && !defined ANDROID
	/*-----------init openGL functions------------*/
	glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC )gl_GetProcAddress("glGenFramebuffers");
	if (glGenFramebuffers == NULL)
		glGenFramebuffers = glGenFramebuffersEXT;

	glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC )gl_GetProcAddress("glBindFramebuffer");
	if(glBindFramebuffer == NULL)
		glBindFramebuffer = glBindFramebufferEXT;

	glGenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC )gl_GetProcAddress("glGenRenderbuffers");
	if (glGenRenderbuffers == NULL)
		glGenRenderbuffers = glGenRenderbuffersEXT;

	glBindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC )gl_GetProcAddress("glBindRenderbuffer");
	if(glBindRenderbuffer == NULL)
		glBindRenderbuffer = glBindRenderbufferEXT;

	glGenerateMipmap = (PFNGLGENERATEMIPMAPPROC )gl_GetProcAddress("glGenerateMipmap");
	if(glGenerateMipmap == NULL)
		glGenerateMipmap = glGenerateMipmapEXT;

	glRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC)gl_GetProcAddress("glRenderbufferStorage");
	if (glRenderbufferStorage == NULL)
		glRenderbufferStorage = glRenderbufferStorageEXT;

	glRenderbufferStorageMultisample = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC )gl_GetProcAddress("glRenderbufferStorageMultisample");
	if(glRenderbufferStorageMultisample == NULL)
		glRenderbufferStorageMultisample = glRenderbufferStorageMultisampleEXT;

	glDeleteFramebuffers =(PFNGLDELETEFRAMEBUFFERSPROC )gl_GetProcAddress("glDeleteFramebuffers");
	if (glDeleteFramebuffers == NULL)
		glDeleteFramebuffers = glDeleteFramebuffersEXT;

	glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC )gl_GetProcAddress("glDeleteRenderbuffers");
	if (glDeleteRenderbuffers == NULL)
		glDeleteRenderbuffers = glDeleteRenderbuffersEXT;

	glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC )gl_GetProcAddress("glCheckFramebufferStatus");
	if (glCheckFramebufferStatus == NULL)
		glCheckFramebufferStatus = glCheckFramebufferStatusEXT;

	glFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC )gl_GetProcAddress("glFramebufferRenderbuffer");
	if (glFramebufferRenderbuffer == NULL)
		glFramebufferRenderbuffer = glFramebufferRenderbufferEXT;

	glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC )gl_GetProcAddress("glFramebufferTexture2D");
	if (glFramebufferTexture2D == NULL)
		glFramebufferTexture2D = glFramebufferTexture2DEXT;
#elif defined IOS
	this->defaultFBO = defaultFBO;
#endif
	//default frame buffer's size
	this->renderTargetWidth = g_pOGLDev->GetWidth();
	this->renderTargetHeight = g_pOGLDev->GetHeight();

	this->buffers = new GLenum[this->maxActiveRenderTargets];
	for(hq_uint32 i = 0; i < this->maxActiveRenderTargets ; ++i)
		this->buffers[i] = GL_COLOR_ATTACHMENT0 + i;

	this->framebuffer = 0;
	glGenFramebuffers(1 , &this->framebuffer);
#ifndef GLES
	if (g_pOGLDev->GetDeviceCaps().maxDrawBuffers == 1)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glBindFramebuffer(GL_FRAMEBUFFER, defaultFBO);
	}
#endif
	Log("Init done!");
}

HQRenderTargetManagerFBO::~HQRenderTargetManagerFBO()
{
	this->RemoveAllRenderTarget();
	this->RemoveAllDepthStencilBuffer();

#if defined DEVICE_LOST_POSSIBLE
	if (!g_pOGLDev->IsDeviceLost())
#endif
	{
		if (this->framebuffer != 0)
			glDeleteFramebuffers(1 , &framebuffer);
	}

	SafeDeleteArray(buffers);
	Log("Released!");
}


HQReturnVal HQRenderTargetManagerFBO::CreateRenderTargetTexture(hq_uint32 width, hq_uint32 height, bool hasMipmaps,
											   HQRenderTargetFormat format, HQMultiSampleType multisampleType,
											   HQTextureType textureType,
											   hq_uint32 *pRenderTargetID_Out,
											   hq_uint32 *pTextureID_Out)
{
#ifdef GLES
	if(!GLEW_OES_texture_non_power_of_two)//texture size must be power of two
	{
		hq_uint32 exp;
		if (!HQBaseTextureManager::IsPowerOfTwo( width  , &exp ))
			width = 0x1 << exp;
		if (!HQBaseTextureManager::IsPowerOfTwo( height  , &exp ))
			height = 0x1 << exp;
	}
#endif
	const Caps &caps = g_pOGLDev->GetDeviceCaps();
	switch (textureType)
	{
		case HQ_TEXTURE_2D:
			if (width > caps.maxTextureSize || height > caps.maxTextureSize) {
				Log("CreateRenderTargetTexture() failed : %u x %u is too large!" , width , height);
				return HQ_FAILED;
			}
			break;
		case HQ_TEXTURE_CUBE:
			if (width > caps.maxCubeTextureSize) {
				Log("CreateRenderTargetTexture() failed : %u x %u is too large for a cube texture!" , width , width);
				return HQ_FAILED;
			}
			break;
	}

	char str[256];
	if (!g_pOGLDev->IsRTTFormatSupported(format , textureType , hasMipmaps))
	{
		HQBaseRenderTargetManager::GetString(format , str);
		if(!hasMipmaps)
			Log("CreateRenderTargetTexture() failed : creating render target texture with format = %s is not supported!" , str);
		else
			Log("CreateRenderTargetTexture() failed : creating render target texture with format = %s and full mipmap levels is not supported!" , str);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}

	if (!g_pOGLDev->IsRTTMultisampleTypeSupported(format , multisampleType , textureType))
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

	GLint internalFormat = HQRenderTargetManagerFBO::GetGLInternalFormat(format);

	hq_uint32 numMipmaps = 1;
	if (hasMipmaps)
		numMipmaps = HQBaseTextureManager::CalculateFullNumMipmaps(width , height);

	HQRenderTargetTextureGL *pNewRenderTarget =
		new HQRenderTargetTextureGL(width , height ,
									multisampleType ,internalFormat ,
									numMipmaps , textureID , pNewTex
									);
	if (pNewRenderTarget == NULL)
	{
		pTextureManager->RemoveTexture(textureID);
		return HQ_FAILED_MEM_ALLOC;
	}

	HQReturnVal re = pNewRenderTarget->Init();
	if (HQFailed(re))
	{
		delete pNewRenderTarget;
		pTextureManager->RemoveTexture(textureID);
		return re;
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


HQReturnVal HQRenderTargetManagerFBO::CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										hq_uint32 *pDepthStencilBufferID_Out)
{

	char str[256];
	if(!g_pOGLDev->IsDSFormatSupported(format))
	{
		HQBaseRenderTargetManager::GetString(format , str);
		Log("CreateDepthStencilBuffer() failed : %s is not supported!" , str);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	GLdepthStencilFormat dsFormat = HQRenderTargetManagerFBO::GetGLFormat(format);
	if (!g_pOGLDev->IsDSMultisampleTypeSupported(format , multisampleType))
	{
		HQBaseRenderTargetManager::GetString(multisampleType , str);
		Log("CreateDepthStencilBuffer() failed : %s is not supported!" , str);
		return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT;
	}


	HQDepthStencilBufferGL *pNewBuffer =
		new HQDepthStencilBufferGL(
				width , height ,
				multisampleType ,
				dsFormat);

	if (pNewBuffer == NULL)
		return HQ_FAILED_MEM_ALLOC;
	HQReturnVal re = pNewBuffer->Init();
	if (HQFailed(re))
	{
		delete pNewBuffer;
		return re;
	}
	if (!this->depthStencilBuffers.AddItem(pNewBuffer , pDepthStencilBufferID_Out))
	{
		delete pNewBuffer;
		return  HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerFBO::ActiveRenderTarget(const HQRenderTargetDesc &renderTargetDesc , 
														hq_uint32 depthStencilBufferID  )
{
	HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDesc.renderTargetID);
	HQActiveRenderTarget & activeRenderTarget =  this->activeRenderTargets[0];

#if defined _DEBUG || defined DEBUG
	if (pRenderTarget == NULL)
	{
		this->ActiveDefaultFrameBuffer();

		return HQ_FAILED;
	}
#endif
	
	if(this->currentUseDefaultBuffer)
		glBindFramebuffer(GL_FRAMEBUFFER , this->framebuffer);
	this->renderTargetWidth = pRenderTarget->width;
	this->renderTargetHeight = pRenderTarget->height;


	if (pRenderTarget == activeRenderTarget.pRenderTarget)//no change in this slot
	{
		if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		{
			if (renderTargetDesc.cubeFace != activeRenderTarget.cubeFace)//different cube face
			{
				glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 ,
					GL_TEXTURE_CUBE_MAP_POSITIVE_X + renderTargetDesc.cubeFace ,
					*(GLuint*)pRenderTarget->GetData() , 0);
			}
		}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
	}//if (pRenderTarget == activeRenderTarget.pRenderTarget)
	else//different render target
	{
		if (pRenderTarget->IsTexture())
		{
			GLuint *pGLtex = (GLuint *)pRenderTarget->GetData();
			if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
				glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 ,
					GL_TEXTURE_CUBE_MAP_POSITIVE_X + renderTargetDesc.cubeFace ,
					*pGLtex , 0);
			else
				glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0  ,
					GL_TEXTURE_2D , *pGLtex , 0);
		}//if (pRenderTarget->IsTexture())

		activeRenderTarget.pRenderTarget = pRenderTarget;
	}//else of if (pRenderTarget == activeRenderTarget.pRenderTarget)

	activeRenderTarget.cubeFace = renderTargetDesc.cubeFace;

	//detach render target at slot <numRenderTargets> and higher
	for (hq_uint32 i = 1 ; i < this->numActiveRenderTargets ; ++i)
	{
		this->DetachSlot(i);
	}

	this->numActiveRenderTargets = 1;

	/*----active depth stencil buffer------------*/
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);

	if (pDepthStencilBuffer != this->pActiveDepthStencilBuffer)
	{

		if (pDepthStencilBuffer == NULL)
		{
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , 0);
		}
		else
		{
			GLuint *depthStencilName = (GLuint *)pDepthStencilBuffer->GetData();
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , depthStencilName[0]);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , depthStencilName[1]);
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	this->currentUseDefaultBuffer = false;
#if	defined DEBUG || defined _DEBUG
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		this->Log("Warning : framebuffer incomplete!");
	}
#endif
#ifndef GLES
	if (g_pOGLDev->GetDeviceCaps().maxDrawBuffers > 1)
		glDrawBuffers(1 , buffers);
#endif

	g_pOGLDev->SetViewPort(g_pOGLDev->GetViewPort());//reset viewport

	return HQ_OK;
}
HQReturnVal HQRenderTargetManagerFBO::ActiveRenderTargets(const HQRenderTargetDesc *renderTargetDescs,
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

	bool firstSlotInvalid = true;//indicate whether first render target in array is invalid or not

	for (hq_uint32 i = 0 ; i < numRenderTargets ; ++i)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDescs[i].renderTargetID);
		HQActiveRenderTarget & activeRenderTarget =  this->activeRenderTargets[i];

		if (pRenderTarget == NULL)
		{
			if (i == 0)//first slot is invalid
				break;
			this->buffers[i] = GL_NONE;//disable fragment output
			this->DetachSlot(i);
			continue;
		}
		//now we know that first render target is valid , if not , this line can't be reached
		if (firstSlotInvalid)
		{
			if(this->currentUseDefaultBuffer)
				glBindFramebuffer(GL_FRAMEBUFFER , this->framebuffer);
			firstSlotInvalid = false;//first render target is valid

			this->renderTargetWidth = pRenderTarget->width;
			this->renderTargetHeight = pRenderTarget->height;
		}


		if (pRenderTarget == activeRenderTarget.pRenderTarget)//no change in this slot
		{
			if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
			{
				if (renderTargetDescs[i].cubeFace != activeRenderTarget.cubeFace)//different cube face
				{
					glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
						GL_TEXTURE_CUBE_MAP_POSITIVE_X + renderTargetDescs[i].cubeFace ,
						*(GLuint*)pRenderTarget->GetData() , 0);
				}
			}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		}//if (pRenderTarget == activeRenderTarget.pRenderTarget)
		else//different render target
		{
			if (pRenderTarget->IsTexture())
			{
				GLuint *pGLtex = (GLuint *)pRenderTarget->GetData();
				if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
					glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
						GL_TEXTURE_CUBE_MAP_POSITIVE_X + renderTargetDescs[i].cubeFace ,
						*pGLtex , 0);
				else
					glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
						GL_TEXTURE_2D , *pGLtex , 0);
			}//if (pRenderTarget->IsTexture())

			this->buffers[i] = GL_COLOR_ATTACHMENT0 + i ;//enable fragment output to this render target

			activeRenderTarget.pRenderTarget = pRenderTarget;
		}//else of if (pRenderTarget == activeRenderTarget.pRenderTarget)

		activeRenderTarget.cubeFace = renderTargetDescs[i].cubeFace;
	}//for (i)

	if (firstSlotInvalid)//first render target is invalid
	{
		this->ActiveDefaultFrameBuffer();

		return HQ_FAILED;
	}
	//detach render target at slot <numRenderTargets> and higher
	for (hq_uint32 i = numRenderTargets ; i < this->numActiveRenderTargets ; ++i)
	{
		this->DetachSlot(i);
	}

	this->numActiveRenderTargets = numRenderTargets;

	/*----active depth stencil buffer------------*/
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);

	if (pDepthStencilBuffer != this->pActiveDepthStencilBuffer)
	{

		if (pDepthStencilBuffer == NULL)
		{
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , 0);
		}
		else
		{
			GLuint *depthStencilName = (GLuint *)pDepthStencilBuffer->GetData();
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , depthStencilName[0]);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , depthStencilName[1]);
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	this->currentUseDefaultBuffer = false;
#if	defined DEBUG || defined _DEBUG
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		this->Log("Warning : framebuffer incomplete!");
	}
#endif
#ifndef GLES
	if (g_pOGLDev->GetDeviceCaps().maxDrawBuffers > 1)
		glDrawBuffers(this->numActiveRenderTargets , buffers);
#endif

	g_pOGLDev->SetViewPort(g_pOGLDev->GetViewPort());//reset viewport

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerFBO::RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList)
{
	bool firstSlotInvalid = true;//indicate whether first render target in array is invalid or not

	for (hq_uint32 i = 0 ; i < savedList.numActiveRenderTargets ; ++i)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = savedList[i].pRenderTarget;
		HQActiveRenderTarget & activeRenderTarget =  this->activeRenderTargets[i];

		if (pRenderTarget == NULL)
		{
			if (i == 0)//first slot is invalid
				break;
			this->buffers[i] = GL_NONE;//disable fragment output
			this->DetachSlot(i);
			continue;
		}
		//now we know that first render target is valid , if not , this line can't be reached
		if (firstSlotInvalid)
		{
			if(this->currentUseDefaultBuffer)
				glBindFramebuffer(GL_FRAMEBUFFER , this->framebuffer);
			firstSlotInvalid = false;//first render target is valid

			this->renderTargetWidth = pRenderTarget->width;
			this->renderTargetHeight = pRenderTarget->height;
		}


		if (pRenderTarget == activeRenderTarget.pRenderTarget)//no change in this slot
		{
			if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
			{
				if (savedList[i].cubeFace != activeRenderTarget.cubeFace)//different cube face
				{
					glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
						GL_TEXTURE_CUBE_MAP_POSITIVE_X + savedList[i].cubeFace ,
						*(GLuint*)pRenderTarget->GetData() , 0);
				}
			}//if (pRenderTarget->IsTexture() && pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
		}//if (pRenderTarget == activeRenderTarget.pRenderTarget)
		else//different render target
		{
			if (pRenderTarget->IsTexture())
			{
				GLuint *pGLtex = (GLuint *)pRenderTarget->GetData();
				if(pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
					glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
						GL_TEXTURE_CUBE_MAP_POSITIVE_X + savedList[i].cubeFace ,
						*pGLtex , 0);
				else
					glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
						GL_TEXTURE_2D , *pGLtex , 0);
			}//if (pRenderTarget->IsTexture())

			this->buffers[i] = GL_COLOR_ATTACHMENT0 + i ;//enable fragment output to this render target

			activeRenderTarget.pRenderTarget = pRenderTarget;
		}//else of if (pRenderTarget == activeRenderTarget.pRenderTarget)

		activeRenderTarget.cubeFace = savedList[i].cubeFace;
	}//for (i)

	if (firstSlotInvalid)//first render target is invalid
	{
		this->ActiveDefaultFrameBuffer();

		return HQ_OK;
	}
	//detach render target at slot <numRenderTargets> and higher
	for (hq_uint32 i = savedList.numActiveRenderTargets ; i < this->numActiveRenderTargets ; ++i)
	{
		this->DetachSlot(i);
	}

	this->numActiveRenderTargets = savedList.numActiveRenderTargets;

	/*----active depth stencil buffer------------*/
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = savedList.pActiveDepthStencilBuffer;

	if (pDepthStencilBuffer != this->pActiveDepthStencilBuffer)
	{

		if (pDepthStencilBuffer == NULL)
		{
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , 0);
		}
		else
		{
			GLuint *depthStencilName = (GLuint *)pDepthStencilBuffer->GetData();
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , depthStencilName[0]);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , depthStencilName[1]);
		}

		this->pActiveDepthStencilBuffer = pDepthStencilBuffer;
	}
	this->currentUseDefaultBuffer = false;
#if	defined DEBUG || defined _DEBUG
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		this->Log("Warning : framebuffer incomplete!");
	}
#endif
#ifndef GLES
	if (g_pOGLDev->GetDeviceCaps().maxDrawBuffers > 1)
		glDrawBuffers(this->numActiveRenderTargets , buffers);
#endif

	g_pOGLDev->SetViewPort(g_pOGLDev->GetViewPort());//reset viewport

	return HQ_OK;
}

void HQRenderTargetManagerFBO::DetachSlot(hq_uint32 i)
{
	HQActiveRenderTarget & activeRenderTarget =  this->activeRenderTargets[i];
	if (activeRenderTarget.pRenderTarget != NULL)
	{
		//detach old render target
		if (activeRenderTarget.pRenderTarget->IsTexture())
		{
			if(activeRenderTarget.pRenderTarget->GetTexture()->type == HQ_TEXTURE_CUBE)
				glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
									   GL_TEXTURE_CUBE_MAP_POSITIVE_X + activeRenderTarget.cubeFace ,
									   0 , 0);
			else {
				glFramebufferTexture2D(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
									   GL_TEXTURE_2D , 0 , 0);
			}

		}

		activeRenderTarget.pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
	}
}

void HQRenderTargetManagerFBO::ActiveDefaultFrameBuffer()
{
	if (this->currentUseDefaultBuffer)
		return;

	glBindFramebuffer(GL_FRAMEBUFFER , defaultFBO);

	this->currentUseDefaultBuffer = true;
	this->numActiveRenderTargets = 0;

	this->renderTargetWidth = g_pOGLDev->GetWidth();
	this->renderTargetHeight = g_pOGLDev->GetHeight();

	g_pOGLDev->SetViewPort(g_pOGLDev->GetViewPort());//reset viewport
}


HQReturnVal HQRenderTargetManagerFBO::GenerateMipmaps(hq_uint32 renderTargetTextureID)
{
	HQBaseCustomRenderBuffer* pRenderTarget = this->renderTargets.GetItemRawPointer(renderTargetTextureID);
#if defined _DEBUG || defined DEBUG
	if (pRenderTarget == NULL || !pRenderTarget->IsTexture())
		return HQ_FAILED_INVALID_ID;
#endif

	HQTextureGL* pTexture = (HQTextureGL*)pRenderTarget->GetTexture().GetRawPointer();
	GLuint *pGLtex = (GLuint *)pTexture->pData;
	GLuint currentBoundTex ;//current bound texture
	
	HQTextureManagerGL* glTextureManager = (HQTextureManagerGL*)this->pTextureManager;

	switch(pTexture->type)
	{
	case HQ_TEXTURE_2D:
		currentBoundTex = glTextureManager->GetActiveTextureUnitInfo().GetTexture2DGL();
		break;
	case HQ_TEXTURE_CUBE:
		currentBoundTex = glTextureManager->GetActiveTextureUnitInfo().GetTextureCubeGL();
		break;
	}
	if (*pGLtex != currentBoundTex)
		glBindTexture(pTexture->textureTarget , *pGLtex);

	glGenerateMipmap(pTexture->textureTarget);

	if (*pGLtex != currentBoundTex)
		glBindTexture(pTexture->textureTarget , currentBoundTex);//re-bind old texture

	return HQ_OK;
}


HQReturnVal HQRenderTargetManagerFBO::RemoveRenderTarget(unsigned int renderTargetID)
{
	HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetID);
	
	if (pRenderTarget == NULL)
		return HQ_FAILED_INVALID_ID;

#if defined DEVICE_LOST_POSSIBLE
	bool call_GL_API = (!g_pOGLDev->IsDeviceLost());//must not call opengl when device is in "lost" state
#else
	bool call_GL_API = true;
#endif

	if (this->currentUseDefaultBuffer && call_GL_API)
		glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer);//bind custom frame buffer so that render target will be detached from this fbo first before it's removed
	
	for (unsigned int i = 0 ; i < this->maxActiveRenderTargets ; ++i)
	{
		if (this->activeRenderTargets[i].pRenderTarget == pRenderTarget)
		{
			if (call_GL_API)
				this->DetachSlot(i);
			else
				this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
		}
	}

	HQReturnVal re = HQBaseRenderTargetManager::RemoveRenderTarget(renderTargetID);

	if (this->currentUseDefaultBuffer && call_GL_API)
		glBindFramebuffer(GL_FRAMEBUFFER, defaultFBO);
	
	return re;
}
void HQRenderTargetManagerFBO::RemoveAllRenderTarget()
{
#if defined DEVICE_LOST_POSSIBLE
	bool call_GL_API = (!g_pOGLDev->IsDeviceLost());//must not call opengl when device is in "lost" state
#else
	bool call_GL_API = true;
#endif

	if (call_GL_API)
		glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer);//bind frame buffer so that render targets will be detached from this fbo first before they're removed
	
	for (unsigned int i = 0 ; i < this->maxActiveRenderTargets ; ++i)
	{
		if (call_GL_API)
			this->DetachSlot(i);
		else
			this->activeRenderTargets[i].pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
	}
	
	HQBaseRenderTargetManager::RemoveAllRenderTarget();
	
	if(call_GL_API)
		glBindFramebuffer(GL_FRAMEBUFFER, defaultFBO);
	
}

HQReturnVal HQRenderTargetManagerFBO::RemoveDepthStencilBuffer(unsigned int bufferID)
{
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(bufferID);
	
	if (pDepthStencilBuffer == NULL)
		return HQ_FAILED_INVALID_ID;

#if defined DEVICE_LOST_POSSIBLE
	bool call_GL_API = (!g_pOGLDev->IsDeviceLost());//must not call opengl when device is in "lost" state
#else
	bool call_GL_API = true;
#endif

	if (this->currentUseDefaultBuffer && call_GL_API)
		glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer);//bind custom frame buffer so that depth stencil buffer will be detached from this fbo first before it's removed
	
	//is this buffer currently the main depth stencil buffer of custom frame buffer?
	if (this->pActiveDepthStencilBuffer == pDepthStencilBuffer)
	{
		this->pActiveDepthStencilBuffer = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
		if (call_GL_API)
		{
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , 0);
		}
	}
	HQReturnVal re =  HQBaseRenderTargetManager::RemoveDepthStencilBuffer(bufferID);
	if (this->currentUseDefaultBuffer && call_GL_API)
		glBindFramebuffer(GL_FRAMEBUFFER, defaultFBO);
	return re;
}
void HQRenderTargetManagerFBO::RemoveAllDepthStencilBuffer()
{
#if defined DEVICE_LOST_POSSIBLE
	if (!g_pOGLDev->IsDeviceLost())
	{
#endif
		glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer);//bind custom frame buffer so that depth stencil buffer will be detached from this fbo first before it's removed
		
		glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER , 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER , GL_STENCIL_ATTACHMENT , GL_RENDERBUFFER , 0);

		if (this->currentUseDefaultBuffer)
			glBindFramebuffer(GL_FRAMEBUFFER, defaultFBO);
#if defined DEVICE_LOST_POSSIBLE
	}
#endif	

	this->pActiveDepthStencilBuffer = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
	HQBaseRenderTargetManager::RemoveAllDepthStencilBuffer();

}
