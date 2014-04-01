/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQCommonVertexStreamManGL_inline.h"
#include "HQVertexStreamManagerPreShaderGL.h"
#include "HQDeviceGL.h"

GLenum g_clientStateType[4] = 
{
	GL_VERTEX_ARRAY,
	GL_COLOR_ARRAY,
	GL_NORMAL_ARRAY,
	GL_TEXTURE_COORD_ARRAY
};

inline void Common_SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride , void* start)
{
	switch (vAttribInfo.attribIndex)
	{
	case 0://position
		glVertexPointer(vAttribInfo.size,vAttribInfo.dataType , stride ,(hqubyte8*)start + (hquint32)vAttribInfo.offset);
		break;
	case 1://color
		glColorPointer(vAttribInfo.size,vAttribInfo.dataType , stride ,(hqubyte*)start + (hquint32)vAttribInfo.offset);
		break;
	case 2://normal
		glNormalPointer(vAttribInfo.dataType , stride , (hqubyte*)start + (hquint32)vAttribInfo.offset);
		break;
	case 3://texcoord
		glTexCoordPointer(vAttribInfo.size,vAttribInfo.dataType , stride , (hqubyte*)start + (hquint32)vAttribInfo.offset);
		break;
	}
}

/*-----------------------HQVertexStreamManagerNoShaderGL--------------------------------------*/
/*--------for HQVertexStreamManDelegateGL---------*/
inline void HQVertexStreamManagerNoShaderGL::EnableVertexAttribArray(GLuint index)
{
	glEnableClientState(g_clientStateType[index]);
}
inline void HQVertexStreamManagerNoShaderGL::DisableVertexAttribArray(GLuint index)
{
	glDisableClientState(g_clientStateType[index]);
}
inline void HQVertexStreamManagerNoShaderGL::SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer)
{
	Common_SetVertexAttribPointer(vAttribInfo , stride , NULL);
}

/*-------override base class-----------*/
bool HQVertexStreamManagerNoShaderGL::IsVertexAttribValid(const HQVertexAttribDesc &vAttribDesc)
{
	switch(vAttribDesc.usage)
	{
	case HQ_VAU_POSITION:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT2 :
		case HQ_VADT_FLOAT3 :
		case HQ_VADT_FLOAT4 :
		case HQ_VADT_SHORT2 :
		case HQ_VADT_SHORT4 :
			return true;
		default:
			this->Log("Error : fixed function vertex position only accepts FLOAT2 , FLOAT3 , FLOAT4 , SHORT2 , SHORT4 data types!");
			return false;
		}
	case HQ_VAU_COLOR:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT4 :
		case HQ_VADT_UBYTE4N:
			return true;
		default:
			this->Log("Error : fixed function vertex color only accepts FLOAT4 , UBYTE4N data types!");
			return false;
		}
	case HQ_VAU_NORMAL:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT3 :
			return true;
		default:
			this->Log("Error : fixed function vertex normal only accepts FLOAT3 data type!");
			return false;
		}
	case HQ_VAU_TEXCOORD0:
		switch (vAttribDesc.dataType)
		{
		case HQ_VADT_FLOAT2 :
		case HQ_VADT_FLOAT3 :
		case HQ_VADT_FLOAT4 :
		case HQ_VADT_SHORT2 :
		case HQ_VADT_SHORT4  :
			return true;
		default:
			this->Log("Error : fixed function vertex texcoord only accepts FLOAT2 , FLOAT3 , FLOAT4 , SHORT2 , SHORT4 data types!");
			return false;
		}
	}
	this->Log("Error : vertex attribute's usage is not supported in fixed function mode!");
	return false;
}

HQReturnVal HQVertexStreamManagerNoShaderGL::SetVertexBuffer(HQVertexBuffer* vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) 
{
	if (streamIndex >= this->maxVertexAttribs)
		return HQ_FAILED;
	HQSharedPtr<HQBufferGL> vBuffer = this->vertexBuffers.GetItemPointer(vertexBufferID).UpCast<HQBufferGL>();

	return HQVertexStreamManDelegateGL<HQVertexStreamManagerNoShaderGL>::SetVertexBuffer(this , vBuffer , this->streams[streamIndex], stride);
}

HQReturnVal HQVertexStreamManagerNoShaderGL::SetVertexInputLayout(HQVertexLayout* inputLayoutID)
{
	HQSharedPtr<HQVertexInputLayoutGL> pVLayout = this->inputLayouts.GetItemPointer(inputLayoutID);
	
	return HQVertexStreamManDelegateGL<HQVertexStreamManagerNoShaderGL>::SetVertexInputLayout(
							this,
							pVLayout,
							this->activeInputLayout,
							this->streams,
							this->maxVertexAttribs,
							this->vAttribInfoNodeCache);
}