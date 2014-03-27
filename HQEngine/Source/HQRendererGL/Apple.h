/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifdef HQ_IPHONE_PLATFORM

#	ifndef HQ_OPENGLES
#		define HQ_OPENGLES
#	endif

#	include "HQDeviceEnumGL.h"

#	import <QuartzCore/CAEAGLLayer.h>
#	import <OpenGLES/EAGLDrawable.h>
#	import <OpenGLES/ES2/gl.h>
#	import <OpenGLES/ES2/glext.h>
#	import <OpenGLES/ES1/gl.h>
#	import <OpenGLES/ES1/glext.h>
#else
#	import <OpenGL/gl.h>
#	import <OpenGL/gl3.h>
#	import <OpenGL/glu.h>
#	import <OpenGL/glext.h>
#	import <OpenGL/gl3ext.h>
#endif

#define GLEW_OK 0

extern GLboolean GLEW_VERSION_1_1;
extern GLboolean GLEW_VERSION_1_2;
extern GLboolean GLEW_VERSION_1_3;
extern GLboolean GLEW_VERSION_1_4;
extern GLboolean GLEW_VERSION_1_5;
extern GLboolean GLEW_VERSION_2_0;
extern GLboolean GLEW_VERSION_2_1;
extern GLboolean GLEW_VERSION_3_0;
extern GLboolean GLEW_VERSION_3_1;
extern GLboolean GLEW_VERSION_3_2;
extern GLboolean GLEW_VERSION_3_3;
extern GLboolean GLEW_VERSION_4_0;
extern GLboolean GLEW_VERSION_4_1;
extern GLboolean GLEW_VERSION_4_2;

extern GLboolean GLEW_ARB_multisample;
extern GLboolean GLEW_EXT_texture_filter_anisotropic;
extern GLboolean GLEW_NV_multisample_filter_hint;
extern GLboolean GLEW_EXT_texture_compression_s3tc;
extern GLboolean GLEW_EXT_geometry_shader4;
extern GLboolean GLEW_ARB_framebuffer_object;
extern GLboolean GLEW_EXT_framebuffer_object;
extern GLboolean GLEW_ARB_draw_buffers;
extern GLboolean GLEW_ARB_texture_float;
extern GLboolean GLEW_EXT_packed_depth_stencil;
extern GLboolean GLEW_ARB_texture_buffer_object;
extern GLboolean GLEW_EXT_texture_buffer_object;
extern GLboolean GLEW_EXT_texture_integer;
extern GLboolean GLEW_ARB_texture_rg;
extern GLboolean GLEW_NV_gpu_shader4;
extern GLboolean GLEW_EXT_gpu_shader4;
extern GLboolean GLEW_ARB_uniform_buffer_object;

//opengl es extensions
extern GLboolean GLEW_OES_texture_non_power_of_two;//full support for none power of two texture
extern GLboolean GLEW_OES_compressed_ETC1_RGB8_texture;
extern GLboolean GLEW_IMG_texture_compression_pvrtc;
extern GLboolean GLEW_EXT_texture_rg;
extern GLboolean GLEW_OES_texture_half_float;
extern GLboolean GLEW_OES_texture_float;

#define GL_TEXTURE_BUFFER 0x8C2A
#define GL_RGBA16F 0x881A
#define GL_RGBA32F 0x8814
#define GL_RED_INTEGER 0x8D94
#ifndef GL_NUM_EXTENSIONS
#define GL_NUM_EXTENSIONS                 0x821D
#endif

#	ifndef GL_OES_texture_half_float
#		define GL_HALF_FLOAT_OES                0x8D61
#	endif

#	ifndef GL_EXT_texture_rg
#		define GL_RED_EXT                 0x1903
#		define GL_RG_EXT                  0x8227
#	endif

#ifndef GL_UNIFORM_BUFFER

#define GL_MAX_UNIFORM_BUFFER_BINDINGS 0x8A2F
#define GL_UNIFORM_BUFFER 0x8A11
#define GL_INVALID_INDEX 0xFFFFFFFF

typedef GLuint (* PFNGLGETUNIFORMBLOCKINDEXPROC) (GLuint program, const GLchar* uniformBlockName);
typedef void (* PFNGLUNIFORMBLOCKBINDINGPROC) (GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
typedef void (*PFNGLBINDBUFFERBASEPROC) (GLenum target, GLuint index, GLuint buffer);
extern PFNGLBINDBUFFERBASEPROC glBindBufferBase;
extern PFNGLGETUNIFORMBLOCKINDEXPROC glGetUniformBlockIndex;
extern PFNGLUNIFORMBLOCKBINDINGPROC glUniformBlockBinding;

#endif//#ifndef GL_UNIFORM_BUFFER

#ifdef HQ_IPHONE_PLATFORM 

#	ifdef SIMULATOR
#		define NUM_DS_BUFFERS 2
#	else
#		define NUM_DS_BUFFERS 1
#	endif

//ericsson texture compression format
#	ifndef GL_ETC1_RGB8_OES    
#		define GL_ETC1_RGB8_OES 0x8D64
#	endif
#	define GL_LUMINANCE32F_ARB 0x8818
#	define GL_LUMINANCE16F_ARB 0x881E
#	define GL_RED 0x1903
#	define GL_RG 0x8227
#	define GL_LUMINANCE8UI_EXT 0x8D80
#	define GL_LUMINANCE_INTEGER_EXT 0x8D9C
#	define GL_DEPTH24_STENCIL8_EXT GL_DEPTH24_STENCIL8_OES
#	define GL_DEPTH_COMPONENT24 GL_DEPTH_COMPONENT24_OES
#	define GL_DEPTH_COMPONENT32 0
#	define GL_RGBA8 GL_RGBA
#	define GL_LUMINANCE8 GL_LUMINANCE
#	define GL_ALPHA8 GL_ALPHA
#	define GL_LUMINANCE8_ALPHA8 GL_LUMINANCE_ALPHA
#	define GL_RGB8 GL_RGB
#	define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT -1
#	define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT -1
#	define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT -1
#	define GL_R8 0x8229
#	define GL_R16 0x822A
#	define GL_RG8 0x822B
#	define GL_RG16 0x822C
#	define GL_R16F 0x822D
#	define GL_R32F 0x822E
#	define GL_RG16F 0x822F
#	define GL_RG32F 0x8230
#	define GL_R8I 0x8231
#	define GL_R8UI 0x8232
#	define GL_R16I 0x8233
#	define GL_R16UI 0x8234
#	define GL_R32I 0x8235
#	define GL_R32UI 0x8236
#	define GL_RG8I 0x8237
#	define GL_RG8UI 0x8238
#	define GL_RG16I 0x8239
#	define GL_RG16UI 0x823A
#	define GL_RG32I 0x823B
#	define GL_RG32UI 0x823C

#	define GLEW_OES_mapbuffer 1
#	define GL_WRITE_ONLY GL_WRITE_ONLY_OES
#	define glMapBuffer glMapBufferOES
#	define glUnmapBuffer glUnmapBufferOES

#	define GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS_EXT 0
#	define GL_MAX_TEXTURE_UNITS_ARB GL_MAX_TEXTURE_UNITS

#	define GL_CLAMP_TO_BORDER GL_CLAMP_TO_EDGE
#	define GL_GEOMETRY_SHADER_EXT 0
#	if GL_APPLE_framebuffer_multisample
#		define GL_MAX_SAMPLES GL_MAX_SAMPLES_APPLE
#		define glRenderbufferStorageMultisample glRenderbufferStorageMultisampleAPPLE
#	else
#		define GL_MAX_SAMPLES 0
#		define glRenderbufferStorageMultisample(target , samples, format , width , height) \
			   glRenderbufferStorage(target , format , width , height)
#	endif

void nullGlUniformMatrixNonSquare( 	
	GLint location,
  	GLsizei count,
  	GLboolean transpose,
  	const GLfloat *value);

#define glUniformMatrix2x3fv nullGlUniformMatrixNonSquare
#define glUniformMatrix2x4fv nullGlUniformMatrixNonSquare
#define glUniformMatrix3x2fv nullGlUniformMatrixNonSquare
#define glUniformMatrix3x4fv nullGlUniformMatrixNonSquare
#define glUniformMatrix4x2fv nullGlUniformMatrixNonSquare
#define glUniformMatrix4x3fv nullGlUniformMatrixNonSquare

#endif


int glewInit();
void * gl_GetProcAddress (const char *name);

#ifdef HQ_IPHONE_PLATFORM

@interface HQIOSOpenGLContext : EAGLContext
{
@private
	GLuint frameBuffer, colorRenderBuffer , depthStencilRenderBuffer[NUM_DS_BUFFERS];
	CAEAGLLayer * eaglLayer;
}


- (id)initWithAPI:(EAGLRenderingAPI)api 
	withEAGLLayer: (CAEAGLLayer *)layer 
   andColorFormat: (FORMAT)colorFormat 
andDepthStencilFormat: (FORMAT) depthStencilFmt;

- (void)presentRenderbuffer;
- (GLuint) getFrameBuffer;

@end

#else
@interface HQAppleOpenGLContext : NSOpenGLContext
{
@private
	NSView *view;	
	uint *pViewWidth , *pViewHeight; 
}
- (id)initWithFormat:(NSOpenGLPixelFormat *)format andView :(NSView *) view 
		   andViewWidthPointer : (uint *) pWidth  andViewHeightPointer : (uint*) pHeight;
//- (void)changeResolution : (GLint) width : (GLint)height;

@end
#endif
