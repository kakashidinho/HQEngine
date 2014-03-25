/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "AndroidGLES.h"
#include <stdio.h>
#include <string.h>
#include "../HQPrimitiveDataType.h"

GLboolean GLEW_VERSION_1_1 = false;
GLboolean GLEW_VERSION_1_2 = false;
GLboolean GLEW_VERSION_1_3 = false;
GLboolean GLEW_VERSION_1_4 = false;
GLboolean GLEW_VERSION_1_5 = false;
GLboolean GLEW_VERSION_2_0 = false;
GLboolean GLEW_VERSION_2_1 = false;
GLboolean GLEW_VERSION_3_0 = false;
GLboolean GLEW_VERSION_3_1 = false;
GLboolean GLEW_VERSION_3_2 = false;
GLboolean GLEW_VERSION_3_3 = false;
GLboolean GLEW_VERSION_4_0 = false;
GLboolean GLEW_VERSION_4_1 = false;

GLboolean GLEW_ARB_multisample;
GLboolean GLEW_EXT_texture_filter_anisotropic;
GLboolean GLEW_NV_multisample_filter_hint;
GLboolean GLEW_EXT_texture_compression_s3tc;
GLboolean GLEW_EXT_geometry_shader4;
GLboolean GLEW_EXT_framebuffer_object;
GLboolean GLEW_ARB_draw_buffers;
GLboolean GLEW_ARB_texture_float;
GLboolean GLEW_EXT_packed_depth_stencil;
GLboolean GLEW_ARB_texture_buffer_object;
GLboolean GLEW_EXT_texture_buffer_object;
GLboolean GLEW_EXT_texture_integer;
GLboolean GLEW_ARB_texture_rg;
GLboolean GLEW_NV_gpu_shader4;
GLboolean GLEW_EXT_gpu_shader4;
GLboolean GLEW_ARB_uniform_buffer_object;
GLboolean GLEW_OES_texture_non_power_of_two;
GLboolean GLEW_OES_mapbuffer;
GLboolean GLEW_OES_compressed_ETC1_RGB8_texture;
GLboolean GLEW_IMG_texture_compression_pvrtc;
GLboolean GLEW_EXT_texture_rg;
GLboolean GLEW_OES_texture_half_float;
GLboolean GLEW_OES_texture_float;

PFNGLGENFRAMEBUFFERSOESPROC android_glGenFramebuffers;
PFNGLBINDFRAMEBUFFEROESPROC android_glBindFramebuffer;
PFNGLGENRENDERBUFFERSOESPROC android_glGenRenderbuffers;
PFNGLBINDRENDERBUFFEROESPROC android_glBindRenderbuffer;
PFNGLGENERATEMIPMAPOESPROC android_glGenerateMipmap;
PFNGLRENDERBUFFERSTORAGEOESPROC android_glRenderbufferStorage;
PFNGLDELETEFRAMEBUFFERSOESPROC android_glDeleteFramebuffers;
PFNGLDELETERENDERBUFFERSOESPROC android_glDeleteRenderbuffers;
PFNGLCHECKFRAMEBUFFERSTATUSOESPROC android_glCheckFramebufferStatus;
PFNGLFRAMEBUFFERRENDERBUFFEROESPROC android_glFramebufferRenderbuffer;
PFNGLFRAMEBUFFERTEXTURE2DOESPROC android_glFramebufferTexture2D;

#define GET_GL_FUNC(name) {if (GLEW_VERSION_2_0)  android_##name = &name; else android_##name = &(name##OES);}

GLboolean gluCheckExtension(const GLubyte* ext , const GLubyte *strExt)
{
	hq_ubyte8* loc = (hq_ubyte8*)strstr((char*)strExt ,(char*) ext);
	if(loc == NULL)
		return false;
	if (loc != strExt) {
		hq_ubyte8 pre = *(loc - 1);
		if (pre != ' ')
			return false;
	}		
	hq_ubyte8 post = *(loc + strlen((char*)ext));
	if(post != ' ' && post != '\0')
		return false;
	return true;
}

//this function does nothing
void nullGlUniformMatrixNonSquare( 	
	GLint location,
  	GLsizei count,
  	GLboolean transpose,
  	const GLfloat *value)
{
}

int glewInit()
{
	const GLubyte * strversion = glGetString(GL_VERSION);
	hq_float32 versionf  = 1.0f;

	sscanf((const char*)strversion, "OpenGL ES %f" , &versionf);
	
	GLEW_VERSION_1_1 = versionf >= 1.1f;
	GLEW_VERSION_1_2 = versionf >= 1.2f;
	GLEW_VERSION_1_3 = versionf >= 1.3f;
	GLEW_VERSION_1_4 = versionf >= 1.4f;
	GLEW_VERSION_1_5 = versionf >= 1.5f;
	GLEW_VERSION_2_0 = versionf >= 2.0f;
	GLEW_VERSION_2_1 = versionf >= 2.1f;
	GLEW_VERSION_3_0 = versionf >= 3.0f;
	GLEW_VERSION_3_1 = versionf >= 3.1f;
	GLEW_VERSION_3_2 = versionf >= 3.2f;
	GLEW_VERSION_3_3 = versionf >= 3.3f;
	GLEW_VERSION_4_0 = versionf >= 4.0f;
	GLEW_VERSION_4_1 = versionf >= 4.1f;
	
	const GLubyte * strExt = glGetString (GL_EXTENSIONS); 
	GLEW_ARB_multisample = gluCheckExtension ((const GLubyte*)"GL_ARB_multisample",strExt);
	GLEW_EXT_texture_filter_anisotropic = gluCheckExtension ((const GLubyte*)"GL_EXT_texture_filter_anisotropic",strExt);
	GLEW_NV_multisample_filter_hint  = gluCheckExtension ((const GLubyte*)"GL_NV_multisample_filter_hint",strExt);
	GLEW_EXT_texture_compression_s3tc = gluCheckExtension ((const GLubyte*)"GL_EXT_texture_compression_s3tc",strExt);
	GLEW_EXT_geometry_shader4 = gluCheckExtension ((const GLubyte*)"GL_EXT_geometry_shader4",strExt);
	GLEW_ARB_draw_buffers = gluCheckExtension ((const GLubyte*)"GL_ARB_draw_buffers",strExt);
	GLEW_ARB_texture_buffer_object = gluCheckExtension ((const GLubyte*)"GL_ARB_texture_buffer_object",strExt);
	GLEW_EXT_texture_buffer_object = gluCheckExtension ((const GLubyte*)"GL_EXT_texture_buffer_object",strExt);
	GLEW_EXT_texture_integer = gluCheckExtension ((const GLubyte*)"GL_EXT_texture_integer",strExt);
	GLEW_ARB_texture_rg = gluCheckExtension ((const GLubyte*)"GL_ARB_texture_rg",strExt);
	GLEW_NV_gpu_shader4 = gluCheckExtension ((const GLubyte*)"GL_NV_gpu_shader4",strExt);
	GLEW_EXT_gpu_shader4 = gluCheckExtension ((const GLubyte*)"GL_EXT_gpu_shader4",strExt);
	GLEW_ARB_uniform_buffer_object = gluCheckExtension ((const GLubyte*)"GL_ARB_uniform_buffer_object",strExt);
	
	if (GLEW_VERSION_2_0)
		GLEW_EXT_framebuffer_object = true;
	else
		GLEW_EXT_framebuffer_object = gluCheckExtension ((const GLubyte*)"GL_OES_framebuffer_object",strExt);
		
	GLEW_ARB_texture_float = gluCheckExtension ((const GLubyte*)"GL_ARB_texture_float",strExt);
	GLEW_EXT_packed_depth_stencil = gluCheckExtension ((const GLubyte*)"GL_OES_packed_depth_stencil",strExt);
	
	GLEW_OES_texture_non_power_of_two = (GLEW_VERSION_2_0 || gluCheckExtension ((const GLubyte*)"GL_APPLE_texture_2D_limited_npot",strExt)) && 
										gluCheckExtension ((const GLubyte*)"GL_OES_texture_npot",strExt);
	GLEW_OES_mapbuffer = gluCheckExtension ((const GLubyte*)"GL_OES_mapbuffer",strExt);
	GLEW_OES_compressed_ETC1_RGB8_texture = gluCheckExtension ((const GLubyte*)"GL_OES_compressed_ETC1_RGB8_texture",strExt);
	GLEW_IMG_texture_compression_pvrtc = gluCheckExtension ((const GLubyte*)"GL_IMG_texture_compression_pvrtc",strExt);
	GLEW_OES_texture_half_float = gluCheckExtension ((const GLubyte*)"GL_OES_texture_half_float",strExt);
	GLEW_OES_texture_float = gluCheckExtension ((const GLubyte*)"GL_OES_texture_float",strExt);
	GLEW_EXT_texture_rg = gluCheckExtension ((const GLubyte*)"GL_EXT_texture_rg",strExt);

	/*-----get framebuffer object related functions -----------*/
	GET_GL_FUNC(glGenFramebuffers);
	GET_GL_FUNC(glBindFramebuffer);
	GET_GL_FUNC(glGenRenderbuffers);
	GET_GL_FUNC(glBindRenderbuffer);
	GET_GL_FUNC(glGenerateMipmap);
	GET_GL_FUNC(glRenderbufferStorage);
	GET_GL_FUNC(glDeleteFramebuffers);
	GET_GL_FUNC(glDeleteRenderbuffers);
	GET_GL_FUNC(glCheckFramebufferStatus);
	GET_GL_FUNC(glFramebufferRenderbuffer);
	GET_GL_FUNC(glFramebufferTexture2D);
	
	return GLEW_OK;
}
