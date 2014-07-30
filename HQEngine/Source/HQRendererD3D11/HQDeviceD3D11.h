/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _DEVICE_D3D_
#define _DEVICE_D3D_
#include "../BaseImpl/HQRenderDeviceBaseImpl.h"
#include <d3d11.h>
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include <d3dx11.h>
#endif
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"DXGI.lib")

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#if defined (_DEBUG)||defined(DEBUG)
#pragma comment(lib,"d3dx11d.lib")
#else
#pragma comment(lib,"d3dx11.lib")
#endif
#endif//#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#if defined (D3D_DEBUG_INFO)
#include <dxerr.h>
#pragma comment(lib,"dxErr.lib")
#endif

#include "HQDeviceEnumD3D11.h"
#include "HQTextureManagerD3D11.h"
#include "HQVertexStreamD3D11.h"
#include "HQShaderD3D11.h"
#include "HQRenderTargetD3D11.h"
#include "HQStateManagerD3D11.h"

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	include <DXGI1_2.h>
#	include <d3d11_1.h>
#	include <agile.h>
#endif


/*------------------------------*/
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
struct WindowInfo
{
	Platform::Agile<Windows::UI::Core::CoreWindow> window;
};
#else
struct WindowInfo
{
	HWND hwind;
	HWND hparent;
	LONG x,y;
	LONG_PTR styles; 
};
#endif

struct HQD3DFEATURE_CAPS
{
	UINT maxTextureSize;
	ULONGLONG maxPrimitives;
	UINT maxMRTs;
	UINT maxAnisotropy;
	UINT maxInputSlots;
	UINT maxVertexSamplers;
	UINT maxVertexTextures;
	UINT maxGeometrySamplers;
	UINT maxGeometryTextures;
	UINT maxPixelSamplers;
	UINT maxPixelTextures;
	UINT maxPixelUAVSlots;
	UINT maxComputeSamplers;
	UINT maxComputeTextures;
	UINT maxComputeUAVSlots;
	UINT maxTotalBoundTextures;//maximum number of textures that can be bound
	UINT maxTotalUAVSlots;
	UINT maxComputeGroupsX;
	UINT maxComputeGroupsY;
	UINT maxComputeGroupsZ;
	UINT shaderModel;
	UINT shaderModelMinor;
	bool colorWriteMask;
	bool alphaToCoverage; 
	bool occlusionQuery;
	bool condNpot;//conditional non power of two textures
};

//****************
//device
//*****************

class HQDeviceD3D11:public HQBaseRenderDevice 
{
private:
	~HQDeviceD3D11();
	void OnResizeBuffer();
	void CreateMainRenderTargetView();
	void CreateMainDepthStencilView();

	void InitFeatureCaps();

	void OnDraw();
	void OnDispatchCompute();

	void CommitCSUAVSlotsChanges();

	void CommitSRVSlotsChanges();

	void CommitIAStreamBufferChanges();

	WindowInfo winfo;
	hModule pDll; 
#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM
	IDXGIFactory * pFactory;
#else
	IDXGIFactory2 * pFactory;
#endif
	ID3D11Device* pDevice;
#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM
	ID3D11DeviceContext *pDevContext;

	IDXGISwapChain * pSwapChain;
#else
	ID3D11DeviceContext1 *pDevContext;

	IDXGISwapChain1 * pSwapChain;
#endif
	ID3D11RenderTargetView *pMainRenderTargetView;
	ID3D11Texture2D* pDepthStencilBuffer;
	ID3D11DepthStencilView *pMainDepthStencilView;
#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM
	DXGI_SWAP_CHAIN_DESC swapchainDesc;
#else
	DXGI_SWAP_CHAIN_DESC1 swapchainDesc;
#endif
	D3D_FEATURE_LEVEL featureLvl;
	HQD3DFEATURE_CAPS featureCaps;
	D3D11_VIEWPORT d3dViewPort[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	UINT numUsedViewports;
	D3D11_PRIMITIVE_TOPOLOGY d3dPrimitiveMode;
	D3D11_PRIMITIVE_TOPOLOGY d3dPrimitivelookupTable[HQ_PRI_NUM_PRIMITIVE_MODE];
	

	HQColor clearColor;
	HQColorui clearColorui;
	hq_float32 clearDepth;
	UINT8 clearStencil;

	HQDeviceEnumD3D11* pEnum;

	//-------------compute's deferred UAV slots--------------------
	ID3D11UnorderedAccessView *pDeferredCSUAVSlots[D3D11_PS_CS_UAV_REGISTER_COUNT];
	UINT CSUAVInitialCounts[D3D11_PS_CS_UAV_REGISTER_COUNT];
	hqint32 minUsedCSUAVSlot;//min used CS's UAV slot
	hqint32 maxUsedCSUAVSlot;//max used CS's UAV slot. -1 if no slot is used

	//deferred resources slots, only send to driver when drawing
	ID3D11ShaderResourceView * pVSSRVSlots[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
	ID3D11ShaderResourceView * pGSSRVSlots[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
	ID3D11ShaderResourceView * pPSSRVSlots[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
	ID3D11ShaderResourceView * pCSSRVSlots[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];

	hqint32 minUsedVSSRVSlot, maxUsedVSSRVSlot;
	hqint32 minUsedGSSRVSlot, maxUsedGSSRVSlot;
	hqint32 minUsedPSSRVSlot, maxUsedPSSRVSlot;
	hqint32 minUsedCSSRVSlot, maxUsedCSSRVSlot;

	//deferred vertex stream slots, only send to driver when drawing
	ID3D11Buffer* pIAStreamBufferSlots[MAX_VERTEX_ATTRIBS];
	UINT IAStreamStrides[MAX_VERTEX_ATTRIBS];
	UINT IAStreamOffsets[MAX_VERTEX_ATTRIBS];
	hqint32 minUsedIAStreamSlot;//min used vertex stream slot
	hqint32 maxUsedIAStreamSlot;//max used vertex stream slot. -1 if no slot is used
public:
	HQDeviceD3D11(hModule _pDll , bool flushLog);

	HQReturnVal Release();

	HQReturnVal Init(HQRenderDeviceInitInput input,const char* settingFileDir, HQLogStream* logFileStream, const char *additionalSettings);

	HQReturnVal BeginRender(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil, hquint32 numRTsToClear);
	HQReturnVal EndRender();
	HQReturnVal DisplayBackBuffer();

	HQMultiSampleType GetMultiSampleType() 
	{return (pEnum->selectedMulSamplesCount > 1)? (HQMultiSampleType)pEnum->selectedMulSamplesCount : HQ_MST_NONE;}

	HQReturnVal Clear(HQBool clearPixel, HQBool clearDepth, HQBool clearStencil, hquint32 numRTsToClear);
	void SetClearColorf(hq_float32 red,hq_float32 green,hq_float32 blue,hq_float32 alpha);//color range:0.0f->1.0f
	void SetClearColori(hq_uint32 red,hq_uint32 green,hq_uint32 blue,hq_uint32 alpha);//color range:0->255
	void SetClearDepthVal(hq_float32 val);
	void SetClearStencilVal(hq_uint32 val);

	void GetClearColor(HQColor &clearColorOut) const 
	{
		clearColorOut = clearColor;
	}

	hqfloat32 GetClearDepthVal() const 
	{
		return clearDepth;
	}

	hquint32 GetClearStencilVal() const
	{
		return clearStencil;
	}

	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions);
	HQReturnVal SetDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed);//thay đổi chế độ hiển thị màn hình
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	void OnOrientationChanged();
#endif
	HQReturnVal ChangeDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed, bool resizeWindow);

	///
	///note: windows store and windows phone: width and height must be the flipped size of window size when in portrait mode
	///
	HQReturnVal OnWindowSizeChanged(hq_uint32 width,hq_uint32 height);

	HQReturnVal SetViewport(const HQViewPort &viewport);
	HQReturnVal SetViewports(const HQViewPort * viewports, hquint32 numViewports);
	HQReturnVal ResetViewports();
	UINT GetNumUsedViewports() const { return this->numUsedViewports; }
	
	void EnableVSync(bool enable);
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	void OnWindowActive(bool active);
#endif

	void SetPrimitiveMode(HQPrimitiveMode primitiveMode) ;

	//put UAV binding to queue 
	HQReturnVal SetUAVForComputeShader(hquint32 slot, ID3D11UnorderedAccessView * UAV);

	HQReturnVal SetVSResource(hquint32 slot, ID3D11ShaderResourceView * pSRV);
	HQReturnVal SetGSResource(hquint32 slot, ID3D11ShaderResourceView * pSRV);
	HQReturnVal SetPSResource(hquint32 slot, ID3D11ShaderResourceView * pSRV);
	HQReturnVal SetCSResource(hquint32 slot, ID3D11ShaderResourceView * pSRV);
	HQReturnVal SetVertexBuffer(hquint32 slot, ID3D11Buffer *buffer, UINT stride, UINT offset);

	HQReturnVal Draw(hq_uint32 vertexCount , hq_uint32 firstVertex) ;
	HQReturnVal DrawPrimitive(hq_uint32 primitiveCount , hq_uint32 firstVertex) ;
	HQReturnVal DrawIndexed(hq_uint32 numVertices , hq_uint32 indexCount , hq_uint32 firstIndex );
	HQReturnVal DrawIndexedPrimitive(hq_uint32 numVertices, hq_uint32 primitiveCount, hq_uint32 firstIndex);

	HQReturnVal DrawInstancedIndirect(HQDrawIndirectArgsBuffer* buffer, hquint32 elementIndex);
	HQReturnVal DrawIndexedInstancedIndirect(HQDrawIndexedIndirectArgsBuffer* buffer, hquint32 elementIndex);

	HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ);
	HQReturnVal DispatchComputeIndirect(HQComputeIndirectArgsBuffer* buffer, hquint32 elementIndex);

	void TextureUAVBarrier() {}
	void BufferUAVBarrier() {}

	/*---------------------------------
	device capabilities
	----------------------------------*/
	HQColorLayout GetColoruiLayout() { return CL_RGBA;}
	hq_uint32 GetMaxVertexStream() {return MAX_VERTEX_ATTRIBS;}
	bool IsIndexDataTypeSupported(HQIndexDataType iDataType) {return true;}
	hq_uint32 GetMaxShaderSamplers();//truy vấn số texture sampler unit nhiều nhất có thể dùng trong shader
	hq_uint32 GetMaxShaderStageSamplers(HQShaderType shaderStage);//truy vấn số texture sampler nhiều nhất có thể dùng trong shader stage <shaderStage>
	hq_uint32 GetMaxShaderTextures();
	hq_uint32 GetMaxShaderStageTextures(HQShaderType shaderStage);
	hq_uint32 GetMaxShaderTextureUAVs();
	hq_uint32 GetMaxShaderStageTextureUAVs(HQShaderType shaderStage);
	hq_uint32 GetMaxShaderBufferUAVs();
	hq_uint32 GetMaxShaderStageBufferUAVs(HQShaderType shaderStage);

	void GetMaxComputeGroups(hquint32 &nGroupsX, hquint32 &nGroupsY, hquint32 &nGroupsZ);

	bool IsTwoSideStencilSupported() //is two sided stencil supported 
	{return true;};
	bool IsBlendStateExSupported() //is extended blend state supported
	{return true;}
	bool IsTextureBufferFormatSupported(HQTextureBufferFormat format);
	bool IsUAVTextureFormatSupported(HQTextureUAVFormat format, HQTextureType textureType, bool hasMipmap);
	bool IsNpotTextureFullySupported(HQTextureType textureType);
	bool IsNpotTextureSupported(HQTextureType textureType);
	/*
	truy vấn khả năng hỗ trợ shader.Ví dụ IsShaderSupport(HQ_VERTEX_SHADER,"2.0").
	Direct3D 9: format <major.minor> major và minor 1 chữa số.
	Direct3D 10/11 : format <major.minor>
	OpenGL : format <major.minor>.Thực chất là kiểm tra GLSL version
	*/
	bool IsShaderSupport(HQShaderType shaderType,const char* version);

	//check if render target texture can be created with format <format>
	//<hasMipmaps> - this texture has full range mipmap or not
	bool IsRTTFormatSupported(HQRenderTargetFormat format , HQTextureType textureType ,bool hasMipmaps);
	//check if depth stencil buffer can be created with format <format>
	bool IsDSFormatSupported(HQDepthStencilFormat format);
	//check if render target texture can be created with multi sample type <multisampleType>
	bool IsRTTMultisampleTypeSupported(HQRenderTargetFormat format , 
											   HQMultiSampleType multisampleType , 
											   HQTextureType textureType) ;
	//check if depth stencil buffer can be created with multi sample type <multisampleType>
	bool IsDSMultisampleTypeSupported(HQDepthStencilFormat format , 
											  HQMultiSampleType multisampleType);
	
	//return max number of render targets can be active at a time
	hq_uint32 GetMaxActiveRenderTargets(){
		return this->featureCaps.maxMRTs;
	}

	const HQD3DFEATURE_CAPS& GetCaps() const {return featureCaps;}
	D3D_FEATURE_LEVEL GetFeatureLevel() const {return featureLvl;}
	
	//check if mipmaps generation for render target texture is supported
	bool IsRTTMipmapGenerationSupported() 
	{
		return true;
	}

	void * GetRawHandle() { return pDevice; }

};
extern HQDeviceD3D11* g_pD3DDev;
#endif
