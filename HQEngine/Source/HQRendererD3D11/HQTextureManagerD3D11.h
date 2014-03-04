/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _MATERIALMAN_
#define _MATERIALMAN_
#include "../BaseImpl/HQTextureManagerBaseImpl.h"
#include <d3d11.h>


struct HQTextureResourceD3D11
{
	ID3D11Resource * pTexture;
	ID3D11ShaderResourceView *pResourceView;

	HQTextureResourceD3D11()
	{
		pTexture = 0;
		pResourceView = 0;
	}
	~HQTextureResourceD3D11()
	{
		SafeRelease(pResourceView);
		SafeRelease(pTexture);
	}

};

class HQTextureManagerD3D11:public HQBaseTextureManager
{
public:
	HQTextureManagerD3D11(ID3D11Device* pDev , ID3D11DeviceContext* pContext,HQLogStream* logFileStream , bool flushLog);
	~HQTextureManagerD3D11();

	HQReturnVal GetTexture2DSize(hq_uint32 textureID, hquint32 &width, hquint32& height);
	HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType, HQTextureCompressionFormat type);
private:
	ID3D11Device* pD3DDevice;
	ID3D11DeviceContext* pD3DContext;

	D3D11_TEXTURE2D_DESC t2DDesc;//for creating texture 2D
	HQSharedPtr<HQTexture> textureSlots[3][D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];

public:
	HQReturnVal CreateShaderResourceView(HQTexture * pTex);
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

	HQReturnVal CreateTextureBuffer(HQTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size  ,void *initData, bool isDynamic);
	HQReturnVal MapTextureBuffer(hq_uint32 textureID , void **ppData);
	HQReturnVal UnmapTextureBuffer(hq_uint32 textureID) ;

	HQBaseRawPixelBuffer* CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height);
	HQReturnVal CreateTexture(HQTexture *pTex, const HQBaseRawPixelBuffer* color);
};
#endif
