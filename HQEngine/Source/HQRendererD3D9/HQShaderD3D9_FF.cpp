/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQShaderD3D9.h"
#include <math.h>
#include <float.h>


HQReturnVal HQShaderManagerD3D9::SetRenderState(hq_uint32 stateType , const hq_int32 *value)
{
	switch(stateType)
	{
	case HQ_LIGHT0: case HQ_LIGHT1: case HQ_LIGHT2:
	case HQ_LIGHT3: 
		{
			HQFFLight * light = (HQFFLight *)value; 
			D3DLIGHT9 d3dlight;
			d3dlight.Type = (D3DLIGHTTYPE) light->type;
			memcpy(&d3dlight.Ambient , &light->ambient , sizeof(HQColor));
			memcpy(&d3dlight.Diffuse , &light->diffuse , sizeof(HQColor));
			memcpy(&d3dlight.Specular , &light->specular , sizeof(HQColor));
			memcpy(&d3dlight.Position , &light->position , sizeof(HQFloat3));
			memcpy(&d3dlight.Direction , &light->direction , sizeof(HQFloat3));
			d3dlight.Attenuation0 = light->attenuation0;
			d3dlight.Attenuation1 = light->attenuation1;
			d3dlight.Attenuation2 = light->attenuation2;
			d3dlight.Range = sqrtf(FLT_MAX);

			pD3DDevice->SetLight(stateType , &d3dlight);
		}
		break;
	case HQ_LIGHT0_ENABLE : case HQ_LIGHT1_ENABLE: case HQ_LIGHT2_ENABLE: 
	case HQ_LIGHT3_ENABLE: 
		pD3DDevice->LightEnable(stateType - HQ_LIGHT0_ENABLE , *value);
		break;
	case HQ_MATERIAL:
		pD3DDevice->SetMaterial((const D3DMATERIAL9*)value);
		break;
	case HQ_TEXTURE_ENABLE:
		if (*value == HQ_TRUE)
		{
			pD3DDevice->SetTextureStageState(0 , D3DTSS_COLOROP , this->firstStageOp[0]);
			pD3DDevice->SetTextureStageState(0 , D3DTSS_ALPHAOP , this->firstStageOp[1]);
		}
		else
		{
			pD3DDevice->SetTextureStageState(0 , D3DTSS_COLOROP , D3DTOP_DISABLE);
			pD3DDevice->SetTextureStageState(0 , D3DTSS_ALPHAOP , D3DTOP_DISABLE);
		}
		break;
	default:
		pD3DDevice->SetRenderState((D3DRENDERSTATETYPE)stateType , *value);
	}

	return HQ_OK;
}
