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


struct HQTextureGL:public HQBaseTexture
{
	HQTextureGL(HQTextureType type = HQ_TEXTURE_2D);
	~HQTextureGL();
	
	GLenum textureTarget;
	void *textureDesc;//width, height ...etc
	HQSharedPtr<HQSamplerStateGL> pSamplerState;

	//implement HQTexture
	virtual hquint32 GetWidth() const;
	virtual hquint32 GetHeight() const;

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle();
};

struct HQTextureUnitInfoGL
{
	HQSharedPtr<HQBaseTexture> texture[3];//current bound textures (2d , cube , buffer)

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
#ifndef HQ_OPENGLES	
	GLuint GetCurrentBoundTBuffer() {return this->currentBoundTBuffer;}
	void InvalidateCurrentBoundTBuffer() {this->currentBoundTBuffer = 0;}
#endif

	void DefineTexture2DSize(HQBaseTexture* pTex, hquint32 width, hquint32 height);//define the size for texture. useful for textures created outside texture manager, such as those created by render target manager

	HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType,HQTextureCompressionFormat type);
	
	inline void ActiveTextureUnit(hq_uint32 slot)
	{
		if (this->activeTexture != slot)
		{
			glActiveTexture(gl_texture(slot));
			this->activeTexture = slot;
		}
	}

	inline void BindTexture(hq_uint32 slot, GLenum target, GLuint texture)
	{
		this->ActiveTextureUnit(slot);
		glBindTexture(target, texture);
	}
#ifndef HQ_OPENGLES
	inline void BindTextureBuffer(GLuint buffer)
	{
		if (currentBoundTBuffer != buffer)
		{
			glBindBuffer(GL_TEXTURE_BUFFER, buffer);
			currentBoundTBuffer = buffer;
		}
	}
#endif

	HQReturnVal SetTexture(hq_uint32 slot , HQTexture* textureID);
	HQReturnVal SetTextureForPixelShader(hq_uint32 slot, HQTexture* textureID);

	HQReturnVal SetTextureUAV(hq_uint32 slot, HQTexture* textureID, hq_uint32 mipLevel);

	HQBaseTexture * CreateNewTextureObject(HQTextureType type);
	HQReturnVal LoadTextureFromStream(HQDataReaderStream* dataStream, HQBaseTexture * pTex);
	HQReturnVal LoadCubeTextureFromStreams(HQDataReaderStream* dataStreams[6] , HQBaseTexture * pTex);
	HQReturnVal InitSingleColorTexture(HQBaseTexture *pTex,HQColorui color);
	HQReturnVal InitTexture(bool changeAlpha,hq_uint32 numMipmaps,HQBaseTexture * pTex);
	HQReturnVal Init2DTexture(hq_uint32 numMipmaps,HQBaseTexture * pTex);
	HQReturnVal InitCubeTexture(hq_uint32 numMipmaps,HQBaseTexture * pTex);
	HQReturnVal SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A);//set giá trị alpha của texel trong texture có giá trị RGB như tham số(hoặc R nến định dạng texture chỉ có kênh 8 bit greyscale) thành giá trị A.
	HQReturnVal SetTransparency(hq_float32 alpha);//set giá trị alpha lớn nhất của toàn bộ texel thành alpha
	
	HQReturnVal InitTextureBuffer(HQBaseTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size , void *initData ,bool isDynamic);

	HQReturnVal RemoveTexture(HQTexture* ID);
	void RemoveAllTexture();

#if defined DEVICE_LOST_POSSIBLE
	void OnLost();
#endif

	HQBaseRawPixelBuffer* CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height);
	HQReturnVal InitTexture(HQBaseTexture *pTex, const HQBaseRawPixelBuffer* color);

private:
	UINT activeTexture;//current active texture unit
#ifndef HQ_OPENGLES
	GLuint currentBoundTBuffer;
#endif
	HQTextureUnitInfoGL * texUnits;
	hq_uint32 maxTextureUnits;
};
#endif
