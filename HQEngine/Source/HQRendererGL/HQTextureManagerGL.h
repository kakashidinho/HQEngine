/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _MATERIALMAN_
#define _MATERIALMAN_
#include "glHeaders.h"
#include "../BaseImpl/HQTextureManagerBaseImpl.h"

inline GLenum gl_texture(UINT unit)
{
	return GL_TEXTURE0 + unit;
}


struct HQSamplerStateGL
{
	HQSamplerStateGL();
	GLint addressU;
	GLint addressV;
	GLint minMipFilter;
	GLint magFilter;
	GLfloat maxAnisotropy;
	HQColor borderColor;
};


struct HQTextureGL:public HQTexture
{
	HQTextureGL(HQTextureType type = HQ_TEXTURE_2D);
	~HQTextureGL();
	
	GLenum textureTarget;
	void *textureDesc;//width, height ...etc
	HQSharedPtr<HQSamplerStateGL> pSamplerState;
};

struct HQTextureUnitInfoGL
{
	HQSharedPtr<HQTexture> texture[3];//current bound textures (2d , cube , buffer)

	GLuint GetTexture2DGL() const {if (texture[HQ_TEXTURE_2D] != NULL) return *(GLuint*)texture[HQ_TEXTURE_2D]->pData ; return 0;}
	GLuint GetTextureCubeGL() const {if (texture[HQ_TEXTURE_CUBE] != NULL) return *(GLuint*)texture[HQ_TEXTURE_CUBE]->pData ; return 0;}
	GLuint GetTextureBufferGL() const {if (texture[HQ_TEXTURE_BUFFER] != NULL) return *(GLuint*)texture[HQ_TEXTURE_BUFFER]->pData ; return 0;}
};

class HQTextureManagerGL:public HQBaseTextureManager
{
public:
	HQTextureManagerGL(
					hq_uint32 maxTextureUnits,
					HQLogStream* logFileStream , 
					bool flushLog);
	~HQTextureManagerGL();

	inline const HQTextureUnitInfoGL & GetActiveTextureUnitInfo() const {return texUnits[activeTexture];}
#ifndef GLES	
	GLuint GetCurrentBoundTBuffer() {return this->currentBoundTBuffer;}
	void InvalidateCurrentBoundTBuffer() {this->currentBoundTBuffer = 0;}
#endif

	HQReturnVal GetTexture2DSize(hq_uint32 textureID, hquint32 &width, hquint32& height);
	HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType,HQTextureCompressionFormat type);
	
	inline void ActiveTextureUnit(hq_uint32 slot)
	{
		if (this->activeTexture != slot)
		{
			glActiveTexture(gl_texture(slot));
			this->activeTexture = slot;
		}
	}
private:
	UINT activeTexture;//current active texture unit
#ifndef GLES
	GLuint currentBoundTBuffer;
#endif
	HQTextureUnitInfoGL * texUnits;
	hq_uint32 maxTextureUnits;

	inline void BindTexture(hq_uint32 slot ,GLenum target ,  GLuint texture)
	{
		this->ActiveTextureUnit(slot);
		glBindTexture(target , texture);
	}
#ifndef GLES
	inline void BindTextureBuffer(GLuint buffer)
	{
		if (currentBoundTBuffer != buffer)
		{
			glBindBuffer(GL_TEXTURE_BUFFER , buffer);
			currentBoundTBuffer = buffer;
		}
	}
#endif

public:
	HQReturnVal SetTexture(hq_uint32 slot , hq_uint32 textureID);
	HQReturnVal SetTextureForPixelShader(hq_uint32 slot , hq_uint32 textureID);
	HQTexture * CreateNewTextureObject(HQTextureType type);
	HQReturnVal LoadTextureFromFile(HQTexture * pTex);
	HQReturnVal LoadCubeTextureFromFiles(const char *fileNames[6] , HQTexture * pTex);
	HQReturnVal CreateSingleColorTexture(HQTexture *pTex,HQColorui color);
	HQReturnVal CreateTexture(bool changeAlpha,hq_uint32 numMipmaps,HQTexture * pTex);
	HQReturnVal Create2DTexture(hq_uint32 numMipmaps,HQTexture * pTex);
	HQReturnVal CreateCubeTexture(hq_uint32 numMipmaps,HQTexture * pTex);
	HQReturnVal SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A);//set giá trị alpha của texel trong texture có giá trị RGB như tham số(hoặc R nến định dạng texture chỉ có kênh 8 bit greyscale) thành giá trị A.
	HQReturnVal SetTransparency(hq_float32 alpha);//set giá trị alpha lớn nhất của toàn bộ texel thành alpha


#ifndef GLES
	HQReturnVal CreateTextureBuffer(HQTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size , void *initData ,bool isDynamic);
	HQReturnVal MapTextureBuffer(hq_uint32 textureID , void **ppData);
	HQReturnVal UnmapTextureBuffer(hq_uint32 textureID) ;
#endif

	HQReturnVal RemoveTexture(hq_uint32 ID);
	void RemoveAllTexture();

#if defined DEVICE_LOST_POSSIBLE
	void OnLost();
#endif

	HQBaseRawPixelBuffer* CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height);
	HQReturnVal CreateTexture(HQTexture *pTex, const HQBaseRawPixelBuffer* color);
};
#endif
