/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQDeviceGL.h"


#ifndef HQ_IPHONE_PLATFORM
#	define defaultFBO 0
#	if !defined HQ_MAC_PLATFORM

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
#ifndef HQ_OPENGLES
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

#		ifdef HQ_ANDROID_PLATFORM
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

//wrapper function of glFramebufferTexture2D. for debugging purpose
inline void glFramebufferTexture2D_wrapper(GLenum target,  GLenum attachment,  GLenum textarget,  GLuint texture,  GLint level)
{
	//now attach new image
	glFramebufferTexture2D( target, attachment, textarget, texture, level);
}

/*-------------------------*/

struct HQDepthStencilBufferGL : public HQBaseDepthStencilBufferView, public HQResetable
{
	HQDepthStencilBufferGL(hq_uint32 _width ,hq_uint32 _height ,
							HQMultiSampleType _multiSampleType,
							GLdepthStencilFormat format)
							: HQBaseDepthStencilBufferView(_width, _height,
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

	void OnLost(){
	}

	void OnReset(){
		//TO DO: recreate buffer
	}

	GLdepthStencilFormat format;
	GLuint depthStencilName[2];//depth buffer name and stencil buffer name
};

/*--------------------------------*/

struct HQRenderTargetTextureGL : public HQBaseRenderTargetTexture, public HQResetable
{
	HQRenderTargetTextureGL(hq_uint32 _width ,hq_uint32 _height ,
							hq_uint32 arraySize,
							HQMultiSampleType _multiSampleType,
							HQRenderTargetFormat hqFormat , hq_uint32 _numMipmaps,
							HQSharedPtr<HQBaseTexture> _pTex)
		: HQBaseRenderTargetTexture(_width , _height ,
							_multiSampleType, _numMipmaps,
							_pTex)
	{
		this->hqFormat = hqFormat;
		switch (pTexture->type)
		{
		case HQ_TEXTURE_2D_ARRAY:
			this->arraySize = arraySize;
			break;
		default:
			//ignore arraySize
			this->arraySize = 1;
		}

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
		HQTextureGL* pTextureGL = static_cast<HQTextureGL*>(this->pTexture.GetRawPointer());
		if (pTextureGL->textureDesc == NULL)//make sure texture resource hasn't been created. It is in the case of UAV texture
		{
			/*--------------create texture-------------------*/
			//Note: this should be done in texture manager side in future
			HQTextureManagerGL *pTextureMan = (HQTextureManagerGL *)g_pOGLDev->GetTextureManager();
			GLint internalFormat;
			GLenum format, type;//get texture's pixel format and pixel data type
			HQRenderTargetManagerFBO::GetGLImageFormat(this->hqFormat, internalFormat, format, type);
			GLuint *pTextureGLHandle = (GLuint*)pTextureGL->pData;
#if 0 && !defined HQ_OPENGLES
			GLclampf priority = 1.0f;
			glPrioritizeTextures(1, pTextureGLHandle, &priority);
#endif
			glGetError();//reset error flag

			switch (pTexture->type)
			{
			case HQ_TEXTURE_2D:
			{
				glBindTexture(GL_TEXTURE_2D, *pTextureGLHandle);
				int w = this->width;
				int h = this->height;
				for (hq_uint32 i = 0; i < this->numMipmaps; ++i)
				{
					glTexImage2D(GL_TEXTURE_2D, i, internalFormat, w, h,
						0, format, type, NULL);

					if (w > 1)
						w >>= 1;
					if (h > 1)
						h >>= 1;
				}

				//re-bind old texture
				glBindTexture(GL_TEXTURE_2D, pTextureMan->GetActiveTextureUnitInfo().GetTexture2DGL());
				//since we created a texture outside texture managet. we need to tell texture manager about the size of this texture
				pTextureMan->DefineTexture2DSize(pTextureGL, this->width, this->height);

			}
				break;
			case HQ_TEXTURE_CUBE:

				glBindTexture(GL_TEXTURE_CUBE_MAP, *pTextureGLHandle);

				for (int face = 0; face < 6; ++face)
				{
					int w = this->width;
					for (hq_uint32 i = 0; i < this->numMipmaps; ++i)
					{
						glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, i, internalFormat, w, w,
							0, format, type, NULL);

						if (w > 1)
							w >>= 1;
					}
				}
				//re-bind old texture
				glBindTexture(GL_TEXTURE_CUBE_MAP, pTextureMan->GetActiveTextureUnitInfo().GetTextureCubeGL());
				//since we created a texture outside texture managet. we need to tell texture manager about the size of this texture
				pTextureMan->DefineTexture2DSize(pTextureGL, this->width, this->width);
				break;
			case HQ_TEXTURE_2D_ARRAY:
#ifndef HQ_OPENGLES//TO DO: add to gles 3 later
				glBindTexture(GL_TEXTURE_2D_ARRAY, *pTextureGLHandle);

				glTexStorage3D(GL_TEXTURE_2D_ARRAY, this->numMipmaps, internalFormat, this->width, this->height, this->arraySize);

				//re-bind old texture
				glBindTexture(GL_TEXTURE_2D_ARRAY, pTextureMan->GetActiveTextureUnitInfo().GetTexture2DArrayGL());
				//since we created a texture outside texture manager. we need to tell texture manager about the size of this texture
				pTextureMan->DefineTexture2DSize(pTextureGL, this->width, this->width);
#endif //#ifndef HQ_OPENGLES
				break;
			}


			if (glGetError() != GL_NO_ERROR)
				return HQ_FAILED;
		}//if (pTextureGL->textureDesc == NULL)
		return HQ_OK;
	}

	void OnLost(){
	}

	void OnReset(){
		//TO DO: recreate render target and texture
	}

	HQRenderTargetFormat hqFormat;
	hquint32 arraySize;
};

/*----------------------------------*/
struct HQRenderTargetGroupGL: public HQBaseRenderTargetGroup, public HQResetable
{
	HQRenderTargetGroupGL(hquint32 numRenderTargets, GLuint _defaultFBO)
		: HQBaseRenderTargetGroup(numRenderTargets),
		  draw_buffers (HQ_NEW GLenum[numRenderTargets]),
		  framebuffer (0)
	{
		glGenFramebuffers(1 , &this->framebuffer);
#ifndef HQ_OPENGLES
		if (g_pOGLDev->GetDeviceCaps().maxDrawBuffers == 1)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);//specify that we will draw to color attachment 0 of frame buffer
			glBindFramebuffer(GL_FRAMEBUFFER, _defaultFBO);
		}
#endif
	}
	~HQRenderTargetGroupGL(){
		delete[] draw_buffers;

#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())
#endif
		{
			if (this->framebuffer != 0)
				glDeleteFramebuffers(1 , &framebuffer);
		}
	}

	void OnLost(){
	}
	void OnReset(){
		//TO DO: recreate frame buffer
	}

	GLenum *draw_buffers;//to disable/enable drawing slot
	GLuint framebuffer;//opengl framebuffer object 
};

/*--------------HQRenderTargetManagerFBO----------------*/

void HQRenderTargetManagerFBO::GetGLImageFormat(HQRenderTargetFormat hqformat, GLint&  internalFormat , GLenum &format , GLenum &type)
{
	internalFormat = format = type = 0;
	switch(hqformat)
	{
	case HQ_RTFMT_R_FLOAT32:
		if (GLEW_VERSION_3_0)
		{
			internalFormat = GL_R32F;
			format = GL_RED; type = GL_FLOAT;
		}
		else if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
			{
				internalFormat = GL_R32F;
				format = GL_RED; type = GL_FLOAT;
			}
			else
			{
				internalFormat = GL_LUMINANCE32F_ARB;	
				format = GL_LUMINANCE; type = GL_FLOAT;
			}
		}
		else if (GLEW_OES_texture_float)
		{
			if (GLEW_EXT_texture_rg)
				internalFormat = format = GL_RED_EXT;
			else
				internalFormat = format = GL_LUMINANCE;
			type = GL_FLOAT; 
		}
		break;
	case HQ_RTFMT_R_FLOAT16:
		if (GLEW_VERSION_3_0)
		{
			internalFormat = GL_R16F;	
			format = GL_RED; type = GL_FLOAT;
		}
		else if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
			{
				internalFormat = GL_R16F;
				format = GL_RED; type = GL_FLOAT;
			}
			else
			{
				internalFormat = GL_LUMINANCE16F_ARB;
				format = GL_LUMINANCE; type = GL_FLOAT;
			}
		}
		else if (GLEW_OES_texture_half_float)
		{
			if (GLEW_EXT_texture_rg)
				internalFormat = format = GL_RED_EXT;
			else
				internalFormat = format = GL_LUMINANCE;
			type = GL_HALF_FLOAT_OES; 
		}
		break;
	case HQ_RTFMT_RGBA_32:
		internalFormat = GL_RGBA8;
		format = GL_RGBA; type = GL_UNSIGNED_BYTE;
		break;
	case HQ_RTFMT_R_UINT8:
		if (GLEW_VERSION_3_0)
		{
			internalFormat = GL_R8UI;
			format = GL_RED_INTEGER ; type = GL_UNSIGNED_BYTE;
		}
		else if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
			{
				internalFormat = GL_R8UI;
				format = GL_RED_INTEGER ; type = GL_UNSIGNED_BYTE;
			}
			else
			{
				internalFormat = GL_LUMINANCE8UI_EXT;
				format = GL_LUMINANCE_INTEGER_EXT ; type = GL_UNSIGNED_BYTE;
			}
		}
		break;
	case HQ_RTFMT_A_UINT8:
		if (GLEW_EXT_texture_integer && !GLEW_VERSION_4_1)
		{
			internalFormat = GL_LUMINANCE8UI_EXT;//this is deprecated
			format = GL_LUMINANCE_INTEGER_EXT ; type = GL_UNSIGNED_BYTE;
		}
		else{
			internalFormat = GL_RGBA8;
			format = GL_RGBA; type = GL_UNSIGNED_BYTE;
		}
		break;
	case HQ_RTFMT_RGBA_FLOAT64:
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_float)
		{
			internalFormat = GL_RGBA16F;
			format = GL_RGBA; type = GL_FLOAT;
		}
		else if (GLEW_OES_texture_half_float)
		{
			internalFormat = format = GL_RGBA;
			type = GL_HALF_FLOAT_OES; 
		}
		break;
	case HQ_RTFMT_RG_FLOAT32:
		if (GLEW_VERSION_3_0)
		{
			internalFormat = GL_RG16F;
			format = GL_RG; type = GL_FLOAT;
		}
		else if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
			{
				internalFormat = GL_RG16F;
				format = GL_RG; type = GL_FLOAT;
			}
		}
		else if (GLEW_OES_texture_half_float && GLEW_EXT_texture_rg)
		{
			internalFormat = format = GL_RG_EXT;
			type = GL_HALF_FLOAT_OES; 
		}
		break;
	case HQ_RTFMT_RGBA_FLOAT128:
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_float)
		{
			internalFormat = GL_RGBA32F;
			format = GL_RGBA; type = GL_FLOAT;
		}
		else if (GLEW_OES_texture_float)
		{
			internalFormat = format = GL_RGBA;
			type = GL_FLOAT; 
		}
		break;
	case HQ_RTFMT_RG_FLOAT64:
		if (GLEW_VERSION_3_0)
		{
			internalFormat = GL_RG32F;
			format = GL_RG; type = GL_FLOAT;
		}
		else if (GLEW_ARB_texture_float)
		{
			if (GLEW_ARB_texture_rg)
			{
				internalFormat = GL_RG32F;
				format = GL_RG; type = GL_FLOAT;
			}
		}
		else if (GLEW_OES_texture_half_float && GLEW_EXT_texture_rg)
		{
			internalFormat = format = GL_RG_EXT;
			type = GL_FLOAT; 
		}
		break;
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
#ifdef HQ_IPHONE_PLATFORM
											   GLuint defaultFBO,
#endif
											   hq_uint32 maxActiveRenderTargets,
											   HQBaseTextureManager *pTexMan,
											   HQLogStream *logFileStream ,  bool flushLog)
						: HQBaseRenderTargetManagerGL(maxActiveRenderTargets ,
										 pTexMan , logFileStream ,
										 "GL Render Target Manager :" ,
										 flushLog)
{

#if !defined HQ_MAC_PLATFORM && !defined HQ_IPHONE_PLATFORM && !defined HQ_ANDROID_PLATFORM
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
#elif defined HQ_IPHONE_PLATFORM
	this->defaultFBO = defaultFBO;
#endif
	//default frame buffer's size
	this->renderTargetWidth = g_pOGLDev->GetWidth();
	this->renderTargetHeight = g_pOGLDev->GetHeight();
	
	Log("Init done!");
}

HQRenderTargetManagerFBO::~HQRenderTargetManagerFBO()
{
	this->RemoveAllRenderTarget();
	this->RemoveAllDepthStencilBuffer();

	Log("Released!");
}


HQReturnVal HQRenderTargetManagerFBO::CreateRenderTargetTexture(hq_uint32 width, hq_uint32 height,
												hq_uint32 arraySize, bool hasMipmaps,
											   HQRenderTargetFormat format, HQMultiSampleType multisampleType,
											   HQTextureType textureType,
											   HQRenderTargetView **pRenderTargetID_Out,
											   HQTexture **pTextureID_Out)
{
#ifdef HQ_OPENGLES
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
	bool isTextureUAV = false;
	HQTextureUAVFormat uavFormat;
	switch (textureType)
	{
		case HQ_TEXTURE_2D_UAV:
			isTextureUAV = true;
			uavFormat = HQBaseTextureManager::GetTextureUAVFormat(format);

		case HQ_TEXTURE_2D:
			if (width > caps.maxTextureSize || height > caps.maxTextureSize) {
				Log("CreateRenderTargetTexture() failed : %u x %u is too large!" , width , height);
				return HQ_FAILED;
			}
			break;
		case HQ_TEXTURE_2D_ARRAY:
			if (width > caps.maxTextureSize || height > caps.maxTextureSize || arraySize > caps.maxTextureArraySize)
			{
				Log("CreateRenderTargetTexture() failed : %u x %u x%u is too large!", width, height, arraySize);
				return HQ_FAILED;
			}
			break;
		case HQ_TEXTURE_CUBE:
			if (width > caps.maxCubeTextureSize) {
				Log("CreateRenderTargetTexture() failed : %u x %u is too large for a cube texture!" , width , width);
				return HQ_FAILED;
			}
			break;
		default:
			Log("CreateRenderTargetTexture() failed : unsuported texture type=%u!", (hquint32)textureType);
			return HQ_FAILED;
	}

	char str[256];
	if (!g_pOGLDev->IsRTTFormatSupported(format , textureType , hasMipmaps)
		|| (isTextureUAV && !g_pOGLDev->IsUAVTextureFormatSupported(uavFormat, textureType, hasMipmaps)))
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
	HQTexture* textureID = 0;
	HQSharedPtr<HQBaseTexture> pNewTex = this->pTextureManager->AddEmptyTexture(textureType , &textureID);
	if (pNewTex == NULL)
		return HQ_FAILED_MEM_ALLOC;

	if (textureType == HQ_TEXTURE_CUBE)
		height = width;

	hq_uint32 numMipmaps = 1;
	if (hasMipmaps)
		numMipmaps = HQBaseTextureManager::CalculateFullNumMipmaps(width , height);

	//unoredered access supported texture
	if (isTextureUAV)
	{
		if (multisampleType != HQ_MST_NONE)
		{
			Log("warning : create render target UAV texture with multisample is not supported, texture will be created with no multisample");
			multisampleType = HQ_MST_NONE;
		}

		//create texture from texture manager side
		HQReturnVal re = HQ_FAILED_INVALID_PARAMETER;
		switch (textureType)
		{
		case HQ_TEXTURE_2D_UAV:
			re = this->pTextureManager->InitTextureUAV(pNewTex.GetRawPointer(), uavFormat, width, height, hasMipmaps);
			break;
		default:
			//TO DO
			break;
		}
		if (HQFailed(re))
		{
			pTextureManager->RemoveTexture(textureID);
			return re;
		}
	}

	HQRenderTargetTextureGL *pNewRenderTarget =
		new HQRenderTargetTextureGL(width , height , arraySize,
									multisampleType ,format ,
									numMipmaps , pNewTex
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
										HQDepthStencilBufferView **pDepthStencilBufferID_Out)
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

HQReturnVal HQRenderTargetManagerFBO::CreateRenderTargetGroupImpl(
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

	bool firstSlotInvalid = true;//indicate whether first render target in array is invalid or not
	bool differentSize = false;//render targets don't have same size

	HQRenderTargetGroupGL* newGroup = new HQRenderTargetGroupGL(numRenderTargets, defaultFBO);

	//save current bound frame buffer
	GLuint currentFBO;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint*)&currentFBO);

	for (hq_uint32 i = 0 ; i < numRenderTargets ; ++i)
	{
		HQSharedPtr<HQBaseRenderTargetView> pRenderTarget = this->renderTargets.GetItemPointer(renderTargetDescs[i].renderTargetID);

		if (pRenderTarget == NULL)
		{
			if (i == 0)//first slot is invalid
				break;
			newGroup->draw_buffers[i] = GL_NONE;//disable fragment output
			continue;
		}
		//now we know that first render target is valid , if not , this line can't be reached
		if (firstSlotInvalid)
		{
			firstSlotInvalid = false;//first render target is valid

			glBindFramebuffer(GL_FRAMEBUFFER, newGroup->framebuffer);
			newGroup->commonWidth = pRenderTarget->width;
			newGroup->commonHeight = pRenderTarget->height;
		}
		else if (pRenderTarget->width != newGroup->commonWidth || pRenderTarget->height != newGroup->commonHeight)
		{
			//this render target has different size
			differentSize = true;
			break;//stop
		}


		if (pRenderTarget->IsTexture())
		{
			HQRenderTargetTextureGL* pTextureTargetGL = static_cast<HQRenderTargetTextureGL*>(pRenderTarget.GetRawPointer());
			GLuint *pGLtex = (GLuint *)pTextureTargetGL->GetData();
			if (pTextureTargetGL->GetTexture()->type == HQ_TEXTURE_CUBE)
				glFramebufferTexture2D_wrapper(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
					GL_TEXTURE_CUBE_MAP_POSITIVE_X + renderTargetDescs[i].cubeFace ,
					*pGLtex , 0);
#ifndef HQ_OPENGLES //TO DO: add to gles 3 later
			else if (pTextureTargetGL->GetTexture()->type == HQ_TEXTURE_2D_ARRAY)
			{
				if (renderTargetDescs[i].arraySlice < pTextureTargetGL->arraySize)
					glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
					*pGLtex, 0, renderTargetDescs[i].arraySlice);
			}
#endif//#ifndef HQ_OPENGLES
			else
				glFramebufferTexture2D_wrapper(GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 + i ,
					GL_TEXTURE_2D , *pGLtex , 0);
		}//if (pRenderTarget->IsTexture())

		newGroup->draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i ;//enable fragment output to this render target

		newGroup->renderTargets[i].pRenderTarget = pRenderTarget;

		newGroup->renderTargets[i].cubeFace = renderTargetDescs[i].cubeFace;
	}//for (i)

	if (firstSlotInvalid)//first render target is invalid
	{
		delete newGroup;
		glBindFramebuffer(GL_FRAMEBUFFER, currentFBO);//restore old FBO

		this->Log("Error : CreateRenderTargetGroupImpl() failed because first render target is invalid!");

		return HQ_FAILED_INVALID_PARAMETER;
	}
	if (differentSize)
	{
		delete newGroup;
		glBindFramebuffer(GL_FRAMEBUFFER, currentFBO);//restore old FBO

		this->Log("Error : CreateRenderTargetGroupImpl() failed because render targets don't have same size!");
		return HQ_FAILED_DIFFERENT_RENDER_TARGETS_SIZE;
	}

	/*----depth stencil buffer------------*/
	HQSharedPtr<HQBaseDepthStencilBufferView> pDepthStencilBuffer = this->depthStencilBuffers.GetItemPointer(depthStencilBufferID);

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

	newGroup->pDepthStencilBuffer = pDepthStencilBuffer;

	//verify frame buffer object
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		delete newGroup;
		glBindFramebuffer(GL_FRAMEBUFFER, currentFBO);//restore old FBO

		this->Log("Error : CreateRenderTargetGroupImpl() failed because of incompatible render targets!");

		return HQ_FAILED;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, currentFBO);//restore old FBO

	*ppRenderTargetGroupOut = newGroup;

	return HQ_OK;
}

HQReturnVal HQRenderTargetManagerFBO::ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& base_group)
{
	HQRenderTargetGroupGL* group = static_cast <HQRenderTargetGroupGL*> (base_group.GetRawPointer());

	if (group == NULL)
	{
		//active default back buffer and depth stencil buffer
		this->ActiveDefaultFrameBuffer();
		return HQ_OK;
	}
	
	glBindFramebuffer(GL_FRAMEBUFFER, group->framebuffer);

	this->currentUseDefaultBuffer = false;
	this->renderTargetWidth = group->commonWidth;
	this->renderTargetHeight = group->commonHeight;

#ifndef HQ_OPENGLES
	if (g_pOGLDev->GetDeviceCaps().maxDrawBuffers > 1)
		glDrawBuffers(group->numRenderTargets , group->draw_buffers);
#endif

	g_pOGLDev->SetViewPort(g_pOGLDev->GetViewPort());//reset viewport

	return HQ_OK;
}

void HQRenderTargetManagerFBO::ActiveDefaultFrameBuffer()
{
	if (this->currentUseDefaultBuffer)
		return;

	glBindFramebuffer(GL_FRAMEBUFFER , defaultFBO);

	this->currentUseDefaultBuffer = true;

	this->renderTargetWidth = g_pOGLDev->GetWidth();
	this->renderTargetHeight = g_pOGLDev->GetHeight();

	g_pOGLDev->SetViewPort(g_pOGLDev->GetViewPort());//reset viewport
}


HQReturnVal HQRenderTargetManagerFBO::GenerateMipmaps(HQRenderTargetView* renderTargetTextureID)
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


void HQRenderTargetManagerFBO::OnLost(){
}

void HQRenderTargetManagerFBO::OnReset(){
	//reset render targets
	HQItemManager<HQBaseRenderTargetView>::Iterator ite1;
	this->renderTargets.GetIterator(ite1);

	while (!ite1.IsAtEnd())
	{
		if (ite1->IsTexture())
		{
			HQRenderTargetTextureGL * rtex = static_cast<HQRenderTargetTextureGL*> (ite1.GetItemPointerNonCheck().GetRawPointer());
			rtex->OnReset();
		}
		//TO DO handle other types of render target
		++ite1;
	}

	//reset depth stencil buffers
	HQItemManager<HQBaseDepthStencilBufferView>::Iterator ite2;
	this->depthStencilBuffers.GetIterator(ite2);

	while (!ite2.IsAtEnd())
	{
		HQDepthStencilBufferGL * ds = static_cast<HQDepthStencilBufferGL*> (ite2.GetItemPointerNonCheck().GetRawPointer());
		ds->OnReset();
		++ite2;
	}

	//reset render target groups
	HQItemManager<HQBaseRenderTargetGroup>::Iterator ite3;
	this->rtGroups.GetIterator(ite3);

	while (!ite3.IsAtEnd())
	{
		HQRenderTargetGroupGL * group = static_cast<HQRenderTargetGroupGL*> (ite3.GetItemPointerNonCheck().GetRawPointer());
		group->OnReset();
		++ite3;
	}
}