/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _MATERIALMAN_
#define _MATERIALMAN_
#include "../BaseImpl/HQTextureManagerBaseImpl.h"
#include "../BaseImpl/HQBaseImplCommon.h"

#include <d3d11.h>

#define HQ_D3D11_USE_TEX_UAV_IN_PIXEL_SHADER 0

#ifndef HQ_DEVICE_D3D11_CLASS_FORWARD_DECLARED
#define HQ_DEVICE_D3D11_CLASS_FORWARD_DECLARED
class HQDeviceD3D11;
#endif

//this is the type of HQBaseTexture::pData
struct HQTextureResourceD3D11
{
	ID3D11Resource * pTexture;
	ID3D11ShaderResourceView *pResourceView;

	HQTextureResourceD3D11()
	{
		pTexture = 0;
		pResourceView = 0;
	}
	virtual ~HQTextureResourceD3D11()
	{
		SafeRelease(pResourceView);
		SafeRelease(pTexture);
	}

};

struct HQUAVTextureResourceD3D11;


struct HQTextureD3D11 :public HQBaseTexture
{
	HQTextureD3D11(HQTextureType type = HQ_TEXTURE_2D);
	~HQTextureD3D11();

	virtual hquint32 GetWidth() const;
	virtual hquint32 GetHeight() const;

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle();

	typedef HQLinkedList<hquint32, HQPoolMemoryManager> SlotList; //list of texture slots that this texture is bound to
	SlotList boundSlots;
	SlotList uavBoundSlots;

	static HQSharedPtr<HQPoolMemoryManager> s_boundSlotsMemManager;//important! must be created before any creation of texture object
};

class HQTextureManagerD3D11:public HQBaseTextureManager
{
public:
	HQTextureManagerD3D11(ID3D11Device* pDev , ID3D11DeviceContext* pContext,HQLogStream* logFileStream , bool flushLog);
	~HQTextureManagerD3D11();

	HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType, HQTextureCompressionFormat type);

	HQReturnVal CreateShaderResourceView(HQBaseTexture * pTex);

	HQReturnVal SetTexture(hq_uint32 slot , HQTexture* textureID);
	HQReturnVal SetTextureForPixelShader(hq_uint32 slot, HQTexture* textureID);
	HQReturnVal SetTexture(HQShaderType shaderStage, hq_uint32 slot, HQTexture* textureID);

	HQReturnVal SetTextureUAV(hq_uint32 slot, HQTexture* textureID, hq_uint32 mipLevel, bool read);
	HQReturnVal SetTextureUAVForComputeShader(hq_uint32 slot, HQTexture* textureID, hq_uint32 mipLevel, bool read);
	
	HQBaseTexture * CreateNewTextureObject(HQTextureType type);
	HQReturnVal LoadTextureFromStream(HQDataReaderStream* dataStream, HQBaseTexture * pTex);
	HQReturnVal LoadCubeTextureFromStreams(HQDataReaderStream* dataStreams[6] , HQBaseTexture * pTex);
	HQReturnVal InitSingleColorTexture(HQBaseTexture *pTex,HQColorui color);
	HQReturnVal InitTexture(bool changeAlpha,hq_uint32 numMipmaps,HQBaseTexture * pTex);
	HQReturnVal Init2DTexture(hq_uint32 numMipmaps,HQBaseTexture * pTex);
	HQReturnVal InitCubeTexture(hq_uint32 numMipmaps,HQBaseTexture * pTex);
	HQReturnVal SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A);//set giá trị alpha của texel trong texture có giá trị RGB như tham số(hoặc R nến định dạng texture chỉ có kênh 8 bit greyscale) thành giá trị A.
	HQReturnVal SetTransparency(hq_float32 alpha);//set giá trị alpha lớn nhất của toàn bộ texel thành alpha

	HQReturnVal InitTextureBuffer(HQBaseTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size  ,void *initData, bool isDynamic);

	HQReturnVal InitTextureUAV(HQBaseTexture *pTex, HQTextureUAVFormat format, hquint32 width, hquint32 height, bool hasMipmap);
	HQReturnVal InitTextureUAVEx(HQBaseTexture *pTex, HQTextureUAVFormat format, hquint32 width, hquint32 height, bool hasMipmap, bool renderTarget);

	HQBaseRawPixelBuffer* CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height);
	HQReturnVal InitTexture(HQBaseTexture *pTex, const HQBaseRawPixelBuffer* color);

	void UnbindTextureFromAllTextureSlots(const HQSharedPtr<HQBaseTexture> &pTexture);//unbind the given texture from every texture slots

	void OnBufferBindToComputeShaderUAVSlot(hquint32 slot);//unbind any texture from compute shader's UAV slot {slot}

	void UnbindTextureFromAllUAVSlots(const HQSharedPtr<HQBaseTexture> &pTexture);//unbind texture from all UAV slots

	static DXGI_FORMAT GetD3DFormat(HQTextureUAVFormat format);
private:
	
	HQDeviceD3D11 * pMasterDevice;

	ID3D11Device* pD3DDevice;
	ID3D11DeviceContext* pD3DContext;

	D3D11_TEXTURE2D_DESC t2DDesc;//for creating texture 2D

	struct TextureSlot
	{
		HQTextureD3D11::SlotList::LinkedListNodeType *textureLink;//for fast removal
		HQSharedPtr<HQBaseTexture> pTexture;
	};
	TextureSlot textureSlots[4][D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];

	TextureSlot textureUAVSlots[2][D3D11_PS_CS_UAV_REGISTER_COUNT];

};
#endif
