/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _DEVICE_D3D_
#define _DEVICE_D3D_
#include "../BaseImpl/HQRenderDeviceBaseImpl.h"
#include <d3d9.h>
#pragma comment(lib,"d3d9.lib")

#if defined (D3D_DEBUG_INFO)
#include <dxerr.h>
#pragma comment(lib,"dxErr.lib")
#endif

#include "HQDeviceEnumD3D9.h"
#include "HQTextureManagerD3D9.h"
#include "HQVertexStreamManagerD3D9.h"
#include "HQRenderTargetD3D9.h"
#include "HQStateManagerD3D9.h"
#include "HQShaderD3D9.h"

/*------------------------------*/
struct WindowInfo
{
	HWND hwind;
	HWND hparent;
	LONG x,y;
	LONG_PTR styles; 
};
//****************************

//****************
//device
//*****************

class HQDeviceD3D9:public HQBaseRenderDevice
{
private:
	~HQDeviceD3D9(); 

	WindowInfo winfo;
	HMODULE pDll; 

	LPDIRECT3D9 pD3D;
	LPDIRECT3DDEVICE9 pDevice;
	D3DPRESENT_PARAMETERS d3dp;
	D3DVIEWPORT9 d3dViewPort;
	D3DPRIMITIVETYPE primitiveMode;
	D3DPRIMITIVETYPE primitiveLookupTable[HQ_PRI_NUM_PRIMITIVE_MODE];
	D3DCOLOR clearColor;
	hq_float32 clearDepth;
	DWORD clearStencil;

	HQDeviceEnumD3D9* pEnum;
public:
	HQDeviceD3D9(HMODULE _pDll , bool flushLog);


	HQReturnVal Release();
	
	bool IsDeviceLost();

	HQReturnVal Init(HQRenderDeviceInitInput input,const char* settingFileDir,HQLogStream* logFileStream, const char *additionalSettings);

	HQReturnVal BeginRender(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil,HQBool clearWholeRenderTarget=HQ_FALSE);
	HQReturnVal EndRender();
	HQReturnVal DisplayBackBuffer();


	HQMultiSampleType GetMultiSampleType() {return (HQMultiSampleType)pEnum->selectedMulSampleType;}

	HQReturnVal Clear(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil,HQBool clearWholeRenderTarget=HQ_FALSE);
	void SetClearColorf(hq_float32 red,hq_float32 green,hq_float32 blue,hq_float32 alpha);//color range:0.0f->1.0f
	void SetClearColori(hq_uint32 red,hq_uint32 green,hq_uint32 blue,hq_uint32 alpha);//color range:0->255
	void SetClearDepthVal(hq_float32 val);
	void SetClearStencilVal(hq_uint32 val);

	void GetClearColor(HQColor &clearColorOut) const ;

	hqfloat32 GetClearDepthVal() const 
	{
		return clearDepth;
	}

	hquint32 GetClearStencilVal() const
	{
		return clearStencil;
	}

	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions)
	{
		pEnum->GetAllDisplayResolution(resolutionList, numResolutions);
	}
	HQReturnVal SetDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed);//thay đổi chế độ hiển thị màn hình
	HQReturnVal OnWindowSizeChanged(hq_uint32 width,hq_uint32 height);
	HQReturnVal ResizeBackBuffer(hq_uint32 width,hq_uint32 height, bool windowed, bool resizeWindow);
	HQReturnVal SetViewPort(const HQViewPort &viewport);
	void EnableVSync(bool enable);
	
	void SetPrimitiveMode(HQPrimitiveMode primitiveMode) ;
	HQReturnVal Draw(hq_uint32 vertexCount , hq_uint32 firstVertex) ;
	HQReturnVal DrawPrimitive(hq_uint32 primitiveCount , hq_uint32 firstVertex) ;
	HQReturnVal DrawIndexed(hq_uint32 numVertices , hq_uint32 indexCount , hq_uint32 firstIndex );
	HQReturnVal DrawIndexedPrimitive(hq_uint32 numVertices, hq_uint32 primitiveCount, hq_uint32 firstIndex);

	HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ) { return HQ_FAILED; }

	void TextureUAVBarrier() {}

	/*---------------------------------
	device capabilities
	----------------------------------*/
	const D3DCAPS9 * GetCaps() {return &pEnum->selectedDevice->dCaps;}
	HQColorLayout GetColoruiLayout() { return CL_BGRA;}
	hq_uint32 GetMaxVertexStream() {return pEnum->selectedDevice->dCaps.MaxStreams;}
	bool IsVertexAttribDataTypeSupported(HQVertexAttribDataType dataType);
	bool IsIndexDataTypeSupported(HQIndexDataType iDataType);
	hq_uint32 GetMaxShaderSamplers() {return pEnum->selectedDevice->numPixelShaderSamplers + pEnum->selectedDevice->numVertexShaderSamplers;}//truy vấn số texture sampler unit nhiều nhất có thể dùng trong shader
	hq_uint32 GetMaxShaderStageSamplers(HQShaderType shaderStage) ;//truy vấn số texture sampler nhiều nhất có thể dùng trong shader stage <shaderStage>

	hq_uint32 GetMaxShaderTextureUAVs() { return 0; }
	hq_uint32 GetMaxShaderStageTextureUAVs(HQShaderType shaderStage) { return 0; }

	void GetMaxComputeGroups(hquint32 &nGroupsX, hquint32 &nGroupsY, hquint32 &nGroupsZ) { nGroupsX = nGroupsY = nGroupsZ = 0; }

	bool IsTwoSideStencilSupported() //is two sided stencil supported 
	{return (pEnum->selectedDevice->dCaps.StencilCaps & D3DSTENCILCAPS_TWOSIDED) != 0;};
	bool IsBlendStateExSupported() //is extended blend state supported
	{return ((pEnum->selectedDevice->dCaps.PrimitiveMiscCaps & D3DPMISCCAPS_SEPARATEALPHABLEND) != 0 &&
			 (pEnum->selectedDevice->dCaps.PrimitiveMiscCaps & D3DPMISCCAPS_BLENDOP) != 0);}
	
	bool IsTextureBufferFormatSupported(HQTextureBufferFormat format) {return false;}
	bool IsUAVTextureFormatSupported(HQTextureUAVFormat format, HQTextureType textureType, bool hasMipmap) { return false; }
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
		return pEnum->selectedDevice->dCaps.NumSimultaneousRTs;
	}
	
	//check if mipmaps generation for render target texture is supported
	bool IsRTTMipmapGenerationSupported() 
	{
		return (pEnum->selectedDevice->dCaps.Caps2 & D3DCAPS2_CANAUTOGENMIPMAP) != 0;
	}
	
	bool IsFormatSupport(D3DFORMAT format , DWORD usage , D3DRESOURCETYPE resourceType);
	bool IsMultiSampleSupport(D3DFORMAT format, D3DMULTISAMPLE_TYPE multisampleType , DWORD *pQuality);

	void OnLostDevice();
	void OnResetDevice();

	void * GetRawHandle() { return pDevice; }
};
extern HQDeviceD3D9* g_pD3DDev;
#endif
