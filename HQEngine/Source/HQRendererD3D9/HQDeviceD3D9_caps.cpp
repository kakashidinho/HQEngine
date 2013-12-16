#include "HQDeviceD3D9PCH.h"
#include "HQDeviceD3D9.h"
#include <string.h>


/*----------------------*/
bool HQDeviceD3D9::IsIndexDataTypeSupported(HQIndexDataType iDataType)
{
	switch(iDataType)
	{
	case HQ_IDT_UINT:
		return pEnum->selectedDevice->dCaps.MaxVertexIndex > 0x0000ffff;
		break;
	default:
		return true;
	}
}

//**********************
//shader support
//**********************
bool HQDeviceD3D9::IsShaderSupport(HQShaderType shaderType,const char* version)
{
	if(strlen(version)!= 3)
		return false;
	int major,minor;

	int re=sscanf(version,"%d.%d",&major,&minor);
	if(re!=2)
		return false;
	switch(shaderType)
	{
	case HQ_VERTEX_SHADER:
		return (pEnum->selectedDevice->dCaps.VertexShaderVersion >= D3DVS_VERSION(major,minor));
	case HQ_PIXEL_SHADER:
		return (pEnum->selectedDevice->dCaps.PixelShaderVersion >= D3DPS_VERSION(major,minor));
	default:
		return false;
	}
}

/*--------------------------*/

hq_uint32 HQDeviceD3D9::GetMaxShaderStageSamplers(HQShaderType shaderStage)
{
	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		return pEnum->selectedDevice->numVertexShaderSamplers;
	case HQ_PIXEL_SHADER:
		return pEnum->selectedDevice->numPixelShaderSamplers;
	default:
		return 0;
	}
}

/*---------------------------*/
bool HQDeviceD3D9::IsNpotTextureFullySupported(HQTextureType textureType)
{
	switch (textureType)
	{
	case HQ_TEXTURE_2D:
		return ((this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_POW2) == 0 && 
				(this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_NONPOW2CONDITIONAL) == 0);
	case HQ_TEXTURE_CUBE:
		return ((this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_CUBEMAP_POW2) == 0 && 
				(this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_NONPOW2CONDITIONAL) == 0);
	default:
		return false;
	}
}
bool HQDeviceD3D9::IsNpotTextureSupported(HQTextureType textureType)
{
	switch (textureType)
	{
	case HQ_TEXTURE_2D:
		return ((this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_POW2) == 0 || 
				(this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_NONPOW2CONDITIONAL) != 0);
	case HQ_TEXTURE_CUBE:
		return ((this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_CUBEMAP_POW2) == 0 || 
				(this->GetCaps()-> TextureCaps  & D3DPTEXTURECAPS_NONPOW2CONDITIONAL) != 0);
	default:
		return false;
	}
}

/*--------------------------*/

bool HQDeviceD3D9::IsVertexAttribDataTypeSupported(HQVertexAttribDataType dataType)
{
	switch (dataType)
	{
	case HQ_VADT_FLOAT :
	case HQ_VADT_FLOAT2 :
	case HQ_VADT_FLOAT3 :
	case HQ_VADT_FLOAT4 :
	case HQ_VADT_SHORT2 :
	case HQ_VADT_SHORT4 :
	case HQ_VADT_UBYTE4N:
		return true;
	case HQ_VADT_UBYTE4 :
		return (pEnum->selectedDevice->dCaps.DeclTypes & D3DDTCAPS_UBYTE4) != 0;
	case HQ_VADT_USHORT2N :
		return (pEnum->selectedDevice->dCaps.DeclTypes & D3DDTCAPS_USHORT2N) != 0;
	case HQ_VADT_USHORT4N :
		return (pEnum->selectedDevice->dCaps.DeclTypes & D3DDTCAPS_USHORT4N) != 0;
	}

	return false;
}

/*-------------------------*/

bool HQDeviceD3D9::IsRTTFormatSupported(HQRenderTargetFormat format , HQTextureType textureType ,bool hasMipmaps)
{
	D3DRESOURCETYPE resourceType ;
	DWORD usage = D3DUSAGE_RENDERTARGET;
	
	if (hasMipmaps)
		usage |= D3DUSAGE_AUTOGENMIPMAP;

	switch(textureType)
	{
	case HQ_TEXTURE_2D:
		resourceType = D3DRTYPE_TEXTURE;
		break;
	case HQ_TEXTURE_CUBE:
		resourceType = D3DRTYPE_CUBETEXTURE;
		break;
	}
	D3DFORMAT D3Dformat = HQRenderTargetManagerD3D9::GetD3DFormat(format);
	if (D3Dformat == D3DFMT_UNKNOWN || 
		!this->IsFormatSupport(D3Dformat , 
		usage ,resourceType)
		)
	{
		return false;
	}

	return true;
}
bool HQDeviceD3D9::IsDSFormatSupported(HQDepthStencilFormat format)
{
	D3DFORMAT D3Dformat = HQRenderTargetManagerD3D9::GetD3DFormat(format);
	if (D3Dformat == D3DFMT_UNKNOWN || 
		!this->IsFormatSupport(D3Dformat , 
		D3DUSAGE_DEPTHSTENCIL,D3DRTYPE_SURFACE)
		)
	{
		return false;
	}

	return true;
}
bool HQDeviceD3D9::IsRTTMultisampleTypeSupported(HQRenderTargetFormat format , HQMultiSampleType multisampleType , HQTextureType textureType)
{
	D3DFORMAT D3Dformat = HQRenderTargetManagerD3D9::GetD3DFormat(format);
	if (D3Dformat == D3DFMT_UNKNOWN)
		return false;
	if (multisampleType != HQ_MST_NONE)//direct3d 9 can't create multisample texture
		return false;
	return true;
}

bool HQDeviceD3D9::IsDSMultisampleTypeSupported(HQDepthStencilFormat format , HQMultiSampleType multisampleType)
{
	D3DFORMAT D3Dformat = HQRenderTargetManagerD3D9::GetD3DFormat(format);
	if (D3Dformat == D3DFMT_UNKNOWN || 
		this->IsMultiSampleSupport(D3Dformat , 
		(D3DMULTISAMPLE_TYPE) multisampleType , NULL))
		return false;
	return true;
}

/*-------------------------*/

bool HQDeviceD3D9::IsFormatSupport(D3DFORMAT format , DWORD usage , D3DRESOURCETYPE resourceType)
{
	if(FAILED(pD3D->CheckDeviceFormat(pEnum->selectedAdapter->adapter , pEnum->selectedDevice->devType , 
		pEnum->selectedMode->Format , usage , resourceType ,format)))
		return false;
	return true;
}

bool HQDeviceD3D9::IsMultiSampleSupport(D3DFORMAT format, D3DMULTISAMPLE_TYPE multisampleType , DWORD *pQuality)
{
	if(FAILED(pD3D->CheckDeviceMultiSampleType(pEnum->selectedAdapter->adapter , pEnum->selectedDevice->devType ,
		format , this->IsWindowed() , multisampleType , pQuality)))
		return false;
	return true;
}