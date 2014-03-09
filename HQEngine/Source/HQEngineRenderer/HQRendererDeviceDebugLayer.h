/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_RENDERER_DEBUG_LAYER_H
#define HQ_RENDERER_DEBUG_LAYER_H

#include "../HQRenderDevice.h"

class HQRenderDeviceDebugLayer : public HQRenderDevice
{
private:
	HQRenderDevice * m_pDevice;
public:

	~HQRenderDeviceDebugLayer() {}

	void SetDevice(HQRenderDevice * pDevice);

	HQReturnVal Release() {return m_pDevice->Release();}

	HQReturnVal Init(HQRenderDeviceInitInput input,const char* settingFileDir, HQLogStream* logFileStream, const char *additionalSettings);

	HQReturnVal BeginRender(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil,HQBool clearWholeRenderTarget=HQ_FALSE);
	HQReturnVal EndRender();
	HQReturnVal DisplayBackBuffer();

	HQMultiSampleType GetMultiSampleType() 
	{
		return m_pDevice->GetMultiSampleType();
	}

	HQReturnVal Clear(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil,HQBool clearWholeRenderTarget=HQ_FALSE);
	void SetClearColorf(hq_float32 red,hq_float32 green,hq_float32 blue,hq_float32 alpha);//color range:0.0f->1.0f
	void SetClearColori(hq_uint32 red,hq_uint32 green,hq_uint32 blue,hq_uint32 alpha);//color range:0->255
	void SetClearDepthVal(hq_float32 val);
	void SetClearStencilVal(hq_uint32 val);

	void GetClearColor(HQColor &clearColorOut) const 
	{
		m_pDevice->GetClearColor(clearColorOut);
	}

	hqfloat32 GetClearDepthVal() const 
	{
		return m_pDevice->GetClearDepthVal();
	}

	hquint32 GetClearStencilVal() const
	{
		return m_pDevice->GetClearStencilVal();
	}

#if !defined GLES && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	HQReturnVal SetDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed);//thay đổi chế độ hiển thị màn hình
	
#endif
	HQReturnVal OnWindowSizeChanged(hq_uint32 width,hq_uint32 height);

	HQReturnVal SetViewPort(const HQViewPort &viewport);
	const HQViewPort & GetViewPort() const {return m_pDevice->GetViewPort();}
	
	void EnableVSync(bool enable);
	void SetPrimitiveMode(HQPrimitiveMode primitiveMode) ;
	HQReturnVal Draw(hq_uint32 vertexCount , hq_uint32 firstVertex) ;
	HQReturnVal DrawPrimitive(hq_uint32 primitiveCount , hq_uint32 firstVertex) ;
	HQReturnVal DrawIndexed(hq_uint32 numVertices , hq_uint32 indexCount , hq_uint32 firstIndex );
	HQReturnVal DrawIndexedPrimitive(hq_uint32 numVertices , hq_uint32 primitiveCount , hq_uint32 firstIndex );

	/*---------------------------------
	device capabilities
	----------------------------------*/
#if !defined GLES && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions)
	{
		m_pDevice->GetAllDisplayResolution(resolutionList , numResolutions);
	}
#endif

	HQColorLayout GetColoruiLayout() 
	{ 
		return m_pDevice->GetColoruiLayout() ;
	}
	hq_uint32 GetMaxVertexStream() 
	{
		return m_pDevice->GetMaxVertexStream();
	}
	bool IsIndexDataTypeSupported(HQIndexDataType iDataType) 
	{
		return m_pDevice->IsIndexDataTypeSupported(iDataType) ;
	}
	hq_uint32 GetMaxShaderSamplers() 
	{
		return m_pDevice->GetMaxShaderSamplers() ;
	}
	hq_uint32 GetMaxShaderStageSamplers(HQShaderType shaderStage) 
	{
		return m_pDevice->GetMaxShaderStageSamplers(shaderStage);
	}
	hq_uint32 GetMaxShaderTextures() 
	{
		return m_pDevice->GetMaxShaderTextures() ;
	}
	hq_uint32 GetMaxShaderStageTextures(HQShaderType shaderStage) 
	{
		return m_pDevice->GetMaxShaderStageTextures(shaderStage);
	}

	bool IsTwoSideStencilSupported() //is two sided stencil supported 
	{
		return m_pDevice->IsTwoSideStencilSupported();
	};
	bool IsBlendStateExSupported() //is extended blend state supported
	{
		return m_pDevice->IsBlendStateExSupported();
	}
	bool IsTextureBufferFormatSupported(HQTextureBufferFormat format) 
	{
		return m_pDevice->IsTextureBufferFormatSupported(format);
	}
	bool IsNpotTextureFullySupported(HQTextureType textureType)
	{
		return m_pDevice->IsNpotTextureFullySupported(textureType);
	}
	bool IsNpotTextureSupported(HQTextureType textureType)
	{
		return m_pDevice->IsNpotTextureSupported(textureType);
	}
	/*
	truy vấn khả năng hỗ trợ shader.Ví dụ IsShaderSupport(HQ_VERTEX_SHADER,"2.0").
	Direct3D 9: format <major.minor> major và minor 1 chữa số.
	Direct3D 10/11 : format <major.minor>
	OpenGL : format <major.minor>.Thực chất là kiểm tra GLSL version
	*/
	bool IsShaderSupport(HQShaderType shaderType,const char* version)
	{
		return m_pDevice->IsShaderSupport(shaderType , version);
	}

	//check if render target texture can be created with format <format>
	//<hasMipmaps> - this texture has full range mipmap or not
	bool IsRTTFormatSupported(HQRenderTargetFormat format , HQTextureType textureType ,bool hasMipmaps)
	{
		return m_pDevice->IsRTTFormatSupported(format, textureType , hasMipmaps);
	}
	//check if depth stencil buffer can be created with format <format>
	bool IsDSFormatSupported(HQDepthStencilFormat format)
	{
		return m_pDevice->IsDSFormatSupported(format);
	}
	//check if render target texture can be created with multi sample type <multisampleType>
	bool IsRTTMultisampleTypeSupported(HQRenderTargetFormat format , 
											   HQMultiSampleType multisampleType , 
											   HQTextureType textureType) 
	{
		return m_pDevice->IsRTTMultisampleTypeSupported(format , multisampleType , textureType);
	}
	//check if depth stencil buffer can be created with multi sample type <multisampleType>
	bool IsDSMultisampleTypeSupported(HQDepthStencilFormat format , 
											  HQMultiSampleType multisampleType)
	{
		return m_pDevice->IsDSMultisampleTypeSupported(format , multisampleType);
	}
	
	//return max number of render targets can be active as a time
	hq_uint32 GetMaxActiveRenderTargets(){
		return m_pDevice->GetMaxActiveRenderTargets();
	}
	
	//check if mipmaps generation for render target texture is supported
	bool IsRTTMipmapGenerationSupported() 
	{
		return m_pDevice->IsRTTMipmapGenerationSupported();
	}

	/*--------------------------------*/
	bool IsRunning(){return m_pDevice->IsRunning();};
	bool IsDeviceLost() {return m_pDevice->IsDeviceLost();}
	
	hq_uint32 GetWidth(){return m_pDevice->GetWidth();};
	hq_uint32 GetHeight(){return m_pDevice->GetHeight();};
	bool IsWindowed() {return m_pDevice->IsWindowed();};
	bool IsVSyncEnabled() {return m_pDevice->IsVSyncEnabled();};

	
	/*------------------------*/
	hq_uint32 GetMaxVertexAttribs() {return m_pDevice->GetMaxVertexAttribs();}//common value
	bool IsVertexAttribDataTypeSupported(HQVertexAttribDataType dataType)
	{
		return m_pDevice->IsVertexAttribDataTypeSupported(dataType);
	}
	/*------------------------*/
	void GetScreenCoord(const HQMatrix4 &viewProj , const HQVector4& vPos , HQPoint<hqint32> &pointOut)//truy vấn tọa độ của 1 điểm trong hệ tọa độ màn hình từ 1 điểm có tọa độ vPos trong hệ tọa độ thế giới
	{
		m_pDevice->GetScreenCoord(viewProj , vPos , pointOut);
	}

	void GetRay(const HQMatrix4 &view ,const HQMatrix4 &proj , hq_float32 zNear,
				const HQPoint<hqint32>& point, HQRay3D & rayOut)
	{
		m_pDevice->GetRay(view , proj , zNear , point , rayOut);
	}
	
	const char * GetDeviceDesc()
	{
		return m_pDevice->GetDeviceDesc();
	}
};

#endif
