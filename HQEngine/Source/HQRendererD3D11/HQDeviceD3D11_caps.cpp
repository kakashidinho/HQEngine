/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQDeviceD3D11.h"


//**********************
//shader support
//**********************
bool HQDeviceD3D11::IsShaderSupport(HQShaderType shaderType,const char* version)
{
	hq_float32 fversion;

	int re=sscanf(version,"%f",&fversion);
	if(re!=1)
		return false;
	if(fversion >= 2.0f && fversion <= this->featureCaps.shaderModel )
		return true;
	return false;
}

/*---------------------------*/
void HQDeviceD3D11::InitFeatureCaps()
{
	switch (this->featureLvl)
	{
	case D3D_FEATURE_LEVEL_11_0:
		featureCaps.maxAnisotropy = 16;
		featureCaps.maxTextureSize = 16384;
		featureCaps.maxPrimitives = (0x1ull << 32) - 1;
		featureCaps.maxMRTs = 8;
		featureCaps.maxInputSlots = 32;
		featureCaps.maxVertexTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxGeometryTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxPixelTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxVertexSamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.maxGeometrySamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.maxPixelSamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.shaderModel = 5;
		featureCaps.shaderModelMinor = 0;
		featureCaps.colorWriteMask = true;
		featureCaps.alphaToCoverage = true;//TO DO: what to do with this feature
		featureCaps.occlusionQuery = false;//TO DO
		featureCaps.condNpot = false;
		break;
	case D3D_FEATURE_LEVEL_10_0:
		featureCaps.maxAnisotropy = 16;
		featureCaps.maxTextureSize = 8192;
		featureCaps.maxPrimitives = (0x1ull << 32) - 1;
		featureCaps.maxMRTs = 8;
		featureCaps.maxInputSlots = 16;
		featureCaps.maxVertexTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxGeometryTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxPixelTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxVertexSamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.maxGeometrySamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.maxPixelSamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.shaderModel = 4;
		featureCaps.shaderModelMinor = 0;
		featureCaps.colorWriteMask = true;
		featureCaps.alphaToCoverage = true;//TO DO: what to do with this feature
		featureCaps.occlusionQuery = false;//TO DO
		featureCaps.condNpot = false;
		break;
	case D3D_FEATURE_LEVEL_9_3:
		featureCaps.maxAnisotropy = 16;
		featureCaps.maxTextureSize = 4096;
		featureCaps.maxPrimitives = 1048575;
#if defined HQ_WIN_PHONE_PLATFORM
		featureCaps.maxMRTs = 1;
#else
		featureCaps.maxMRTs = 4;
#endif
		featureCaps.maxInputSlots = 16;
		featureCaps.maxVertexTextures = 0;
		featureCaps.maxGeometryTextures = 0;
		featureCaps.maxPixelTextures = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		featureCaps.maxVertexSamplers = 0;
		featureCaps.maxGeometrySamplers = 0;
		featureCaps.maxPixelSamplers = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
		featureCaps.shaderModel = 2;
		featureCaps.shaderModelMinor = 0;
		featureCaps.colorWriteMask = true;
		featureCaps.alphaToCoverage = false;//TO DO: what to do with this feature
		featureCaps.occlusionQuery = false;//TO DO
		featureCaps.condNpot = true;
		break;
	}
}

/*---------------------------*/
hq_uint32 HQDeviceD3D11::GetMaxShaderSamplers() //truy vấn số texture sampler unit nhiều nhất có thể dùng trong shader
{
	return featureCaps.maxVertexSamplers + featureCaps.maxGeometrySamplers + featureCaps.maxPixelSamplers;
}
hq_uint32 HQDeviceD3D11::GetMaxShaderStageSamplers(HQShaderType shaderStage) //truy vấn số texture sampler nhiều nhất có thể dùng trong shader stage <shaderStage>
{
	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		return featureCaps.maxVertexSamplers;
	case HQ_GEOMETRY_SHADER:
		return featureCaps.maxGeometrySamplers;
	case HQ_PIXEL_SHADER:
		return featureCaps.maxPixelSamplers;
	default:
		//TO DO
		return 0;
	}
}

hq_uint32 HQDeviceD3D11::GetMaxShaderTextures() //truy vấn số texture nhiều nhất có thể dùng trong shader
{
	return featureCaps.maxVertexTextures + featureCaps.maxGeometryTextures + featureCaps.maxPixelTextures;
}
hq_uint32 HQDeviceD3D11::GetMaxShaderStageTextures(HQShaderType shaderStage) //truy vấn số texture nhiều nhất có thể dùng trong shader stage <shaderStage>
{
	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		return featureCaps.maxVertexTextures;
	case HQ_GEOMETRY_SHADER:
		return featureCaps.maxGeometryTextures;
	case HQ_PIXEL_SHADER:
		return featureCaps.maxPixelTextures;
	default:
		//TO DO
		return 0;
	}
}

bool HQDeviceD3D11::IsTextureBufferFormatSupported(HQTextureBufferFormat format)
{
	return this->featureLvl > D3D_FEATURE_LEVEL_9_3;
}

bool HQDeviceD3D11::IsNpotTextureFullySupported(HQTextureType textureType)
{
	return this->featureCaps.condNpot == false;
}
bool HQDeviceD3D11::IsNpotTextureSupported(HQTextureType textureType)
{
	return true;
}

/*---------------------------*/

bool HQDeviceD3D11::IsRTTFormatSupported(HQRenderTargetFormat format , HQTextureType textureType ,bool hasMipmaps)
{
	UINT masks = D3D11_FORMAT_SUPPORT_RENDER_TARGET;

	if (hasMipmaps)
		masks |=  D3D11_FORMAT_SUPPORT_MIP;
	switch (textureType)
	{
	case HQ_TEXTURE_2D:
		masks |= D3D11_FORMAT_SUPPORT_TEXTURE2D;
		break;
	case HQ_TEXTURE_CUBE:
		masks |= D3D11_FORMAT_SUPPORT_TEXTURECUBE;
		break;
	}
	DXGI_FORMAT D3Dformat = HQRenderTargetManagerD3D11::GetD3DFormat(format);
	if (D3Dformat == DXGI_FORMAT_FORCE_UINT || 
		FAILED(this->pDevice->CheckFormatSupport(D3Dformat , &masks))
		)
	{
		return false;
	}

	return true;
}
bool HQDeviceD3D11::IsDSFormatSupported(HQDepthStencilFormat format)
{
	UINT masks = D3D11_FORMAT_SUPPORT_DEPTH_STENCIL;
	
	DXGI_FORMAT D3Dformat = HQRenderTargetManagerD3D11::GetD3DFormat(format);
	if (D3Dformat == DXGI_FORMAT_FORCE_UINT || 
		FAILED(this->pDevice->CheckFormatSupport(D3Dformat , &masks))
		)
	{
		return false;
	}

	return true;
}
bool HQDeviceD3D11::IsRTTMultisampleTypeSupported(HQRenderTargetFormat format , HQMultiSampleType multisampleType , HQTextureType textureType)
{
	UINT sampleCount = (multisampleType > 0)? multisampleType : 1;
	UINT quality;
	DXGI_FORMAT D3Dformat = HQRenderTargetManagerD3D11::GetD3DFormat(format);
	if (D3Dformat == DXGI_FORMAT_FORCE_UINT)
	{
		return false;
	}

	pDevice->CheckMultisampleQualityLevels(D3Dformat , sampleCount , &quality);
	if (quality == 0)
		return false;

	return true;
}

bool HQDeviceD3D11::IsDSMultisampleTypeSupported(HQDepthStencilFormat format , HQMultiSampleType multisampleType)
{
	UINT sampleCount = (multisampleType > 0)? multisampleType : 1;
	UINT quality;
	DXGI_FORMAT D3Dformat = HQRenderTargetManagerD3D11::GetD3DFormat(format);
	if (D3Dformat == DXGI_FORMAT_FORCE_UINT)
	{
		return false;
	}

	pDevice->CheckMultisampleQualityLevels(D3Dformat , sampleCount , &quality);
	if (quality == 0)
		return false;
	return true;
}
