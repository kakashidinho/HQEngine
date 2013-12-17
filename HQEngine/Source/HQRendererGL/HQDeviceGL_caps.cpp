/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQDeviceGL.h"

bool HQDeviceGL::IsTextureBufferFormatSupported(HQTextureBufferFormat format)
{
	if (!GLEW_VERSION_3_1 && !GLEW_ARB_texture_buffer_object && !GLEW_EXT_texture_buffer_object)
		return false;
	switch (format)
	{
	case HQ_TBFMT_R16_FLOAT :
	case HQ_TBFMT_R16G16B16A16_FLOAT  :
	case HQ_TBFMT_R32_FLOAT  :
	case HQ_TBFMT_R32G32B32_FLOAT:
	case HQ_TBFMT_R32G32B32A32_FLOAT  :
		return (GLEW_VERSION_3_0 || GLEW_ARB_texture_float);
	case HQ_TBFMT_R8_INT :
	case HQ_TBFMT_R8G8B8A8_INT :
	case HQ_TBFMT_R8_UINT  :
	case HQ_TBFMT_R8G8B8A8_UINT  :
	case HQ_TBFMT_R16_INT :
	case HQ_TBFMT_R16G16B16A16_INT :
	case HQ_TBFMT_R16_UINT :
	case HQ_TBFMT_R16G16B16A16_UINT :
	case HQ_TBFMT_R32_INT :
	case HQ_TBFMT_R32G32B32A32_INT :
	case HQ_TBFMT_R32_UINT :
	case HQ_TBFMT_R32G32B32A32_UINT :
		return (GLEW_VERSION_3_0 || GLEW_EXT_texture_integer);
	case HQ_TBFMT_R8_UNORM  :
	case HQ_TBFMT_R8G8B8A8_UNORM  :
	case HQ_TBFMT_R16_UNORM :
	case HQ_TBFMT_R16G16B16A16_UNORM :
		return true;
	}
	return false;
}
bool HQDeviceGL::IsNpotTextureFullySupported(HQTextureType textureType)
{
	//TO DO: GL_APPLE_texture_2D_limited_npot for GLES
	switch (textureType)
	{
	case HQ_TEXTURE_2D:
	case HQ_TEXTURE_CUBE:
#ifdef GLES
		return GLEW_OES_texture_non_power_of_two;
#else
		return true;
#endif
	default:
		return false;
	}
}
bool HQDeviceGL::IsNpotTextureSupported(HQTextureType textureType)
{
	switch (textureType)
	{
	case HQ_TEXTURE_2D:
	case HQ_TEXTURE_CUBE:
#ifdef GLES
		return GLEW_OES_texture_non_power_of_two;
#else
		return true;
#endif
	default:
		return false;
	}
}

//**********************
//shader support
//**********************
bool HQDeviceGL::IsShaderSupport(HQShaderType shaderType,const char* version)
{
	if (shaderType == HQ_GEOMETRY_SHADER &&
		!GLEW_EXT_geometry_shader4 && !GLEW_VERSION_3_2)
		return false;
	hq_float32 fversion , fversion2;

	int re=sscanf(version,"%f",&fversion);
	if(re!=1)
		return false;
	const GLubyte * glsl=glGetString(GL_SHADING_LANGUAGE_VERSION);

#ifdef GLES
	re=sscanf((const char*)glsl,"OpenGL ES GLSL ES %f",&fversion2);
#else
	re=sscanf((const char*)glsl,"%f",&fversion2);
#endif
	if(re!=1)
		return false;
	if(fversion2 < fversion)
		return false;
	return true;
}

hq_uint32 HQDeviceGL::GetMaxShaderStageSamplers(HQShaderType shaderStage)
{
	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		return pEnum->caps.nVertexShaderSamplers;
	case HQ_PIXEL_SHADER:
		return pEnum->caps.nFragmentShaderSamplers;
	case HQ_GEOMETRY_SHADER:
		if (GLEW_EXT_geometry_shader4 || GLEW_VERSION_3_2)
			return pEnum->caps.nGeometryShaderSamplers;
	default:
		return 0;
	}
}

/*-----------render target caps--------------------*/

bool HQDeviceGL::IsRTTFormatSupported(HQRenderTargetFormat format , HQTextureType textureType ,bool hasMipmaps)
{
	GLint internalFormat = HQRenderTargetManagerFBO::GetGLInternalFormat(format);
	if (internalFormat == 0)
		return false;

	return pEnum->caps.rttInternalFormat[format];
}
bool HQDeviceGL::IsDSFormatSupported(HQDepthStencilFormat format)
{
	GLdepthStencilFormat dsFormat = HQRenderTargetManagerFBO::GetGLFormat(format);
	if (dsFormat.depthFormat == 0 && dsFormat.stencilFormat == 0)
		return false;
	return pEnum->caps.dsFormat[format];
}
bool HQDeviceGL::IsRTTMultisampleTypeSupported(HQRenderTargetFormat format , HQMultiSampleType multisampleType , HQTextureType textureType)
{
	GLint internalFormat = HQRenderTargetManagerFBO::GetGLInternalFormat(format);
	if (internalFormat == 0 || g_pOGLDev->GetDeviceCaps().rttInternalFormat[format] == false)
		return false;
	if (multisampleType != HQ_MST_NONE)//multisample texture not supported yet
		return false;
	return true;
}

bool HQDeviceGL::IsDSMultisampleTypeSupported(HQDepthStencilFormat format , HQMultiSampleType multisampleType)
{
#if !defined APPLE && !defined GLES
	if (glRenderbufferStorageMultisample == NULL)
		return false;
#endif
#if defined GLES
#	if !GL_APPLE_framebuffer_multisample
	if (multisampleType != HQ_MST_NONE)
		return false;
#	endif
#endif
	GLint maxSamples;//get max samples
	glGetIntegerv(GL_MAX_SAMPLES , &maxSamples);

	if (maxSamples < (GLint)multisampleType)
		return false;

	return pEnum->caps.dsFormat[format];
}
