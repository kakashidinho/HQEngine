/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _TEXTUREMAN_
#define _TEXTUREMAN_
#include "../BaseImpl/HQBaseImplCommon.h"
#include "../BaseImpl/HQTextureManagerBaseImpl.h"
#include "d3d9.h"



class HQTextureManagerD3D9:public HQBaseTextureManager
{
public:
	HQTextureManagerD3D9(LPDIRECT3DDEVICE9 pDev, DWORD textureCaps , DWORD s3tc_dxtFlags,
						DWORD maxVertexShaderSamplers ,DWORD maxPixelShaderSamplers,
						HQLogStream* logFileStream , bool flushLog);
	~HQTextureManagerD3D9();

	void OnResetDevice();

	HQReturnVal GetTexture2DSize(hq_uint32 textureID, hquint32 &width, hquint32& height);
	HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType, HQTextureCompressionFormat type);
private:
	struct ShaderStageInfo{
		DWORD maxSamplers;
		DWORD samplerOffset;
		HQSharedPtr<HQTexture> * samplerSlots;

		~ShaderStageInfo() { SafeDeleteArray(samplerSlots); }
	};

	LPDIRECT3DDEVICE9 pD3DDevice;
	DWORD textureCaps;
	DWORD s3tc_dxtFlags;//compressed image support flags
	ShaderStageInfo shaderStage[2];//vertex shader and pixel shader

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
	

	HQBaseRawPixelBuffer* CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height);
	HQReturnVal CreateTexture(HQTexture *pTex, const HQBaseRawPixelBuffer* color);
};

#endif
