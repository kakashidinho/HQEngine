/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#import "Apple.h"
#import <mach-o/dyld.h>
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
GLboolean GLEW_VERSION_4_2 = false;

GLboolean GLEW_ARB_multisample;
GLboolean GLEW_EXT_texture_filter_anisotropic;
GLboolean GLEW_NV_multisample_filter_hint;
GLboolean GLEW_OES_texture_non_power_of_two;
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
#ifdef IOS
GLboolean GLEW_OES_compressed_ETC1_RGB8_texture;
GLboolean GLEW_IMG_texture_compression_pvrtc;
#endif


#ifndef GL_UNIFORM_BUFFER
PFNGLBINDBUFFERBASEPROC glBindBufferBase = NULL;
PFNGLGETUNIFORMBLOCKINDEXPROC glGetUniformBlockIndex = NULL;
PFNGLUNIFORMBLOCKBINDINGPROC glUniformBlockBinding = NULL;
#endif

#ifdef IOS
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
#endif

void * gl_GetProcAddress (const char *procName)

{
#ifdef IOS
	return NULL;
#else
    static CFBundleRef openGLBundle = NULL;
	
    if (openGLBundle == NULL)
    {
        openGLBundle = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.opengl"));
    }
	
    CFStringRef functionName = CFStringCreateWithCString(NULL, procName, kCFStringEncodingASCII);
	
    void* function = CFBundleGetFunctionPointerForName(openGLBundle, functionName);
	
    CFRelease(functionName);
	
    return function;
#endif
	
}

int glewInit()
{
	const GLubyte * strversion = glGetString(GL_VERSION);
	hq_float32 versionf  = 1.0f;
#ifdef IOS
	sscanf((const char*)strversion, "OpenGL ES %f" , &versionf);
#else
	sscanf((const char*)strversion, "%f" , &versionf);
#endif
	
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
    GLEW_VERSION_4_2 = versionf >= 4.2f;
	
	const GLubyte * strExt = glGetString (GL_EXTENSIONS); 
	GLEW_ARB_multisample = gluCheckExtension ((const GLubyte*)"GL_ARB_multisample",strExt);
	GLEW_EXT_texture_filter_anisotropic = gluCheckExtension ((const GLubyte*)"GL_EXT_texture_filter_anisotropic",strExt);
	GLEW_NV_multisample_filter_hint  = gluCheckExtension ((const GLubyte*)"GL_NV_multisample_filter_hint",strExt);
	GLEW_OES_texture_non_power_of_two = gluCheckExtension ((const GLubyte*)"GL_OES_texture_npot",strExt);
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
#ifdef IOS
	GLEW_EXT_framebuffer_object = true;
	GLEW_ARB_texture_float = gluCheckExtension ((const GLubyte*)"GL_OES_texture_float",strExt);
	GLEW_EXT_packed_depth_stencil = gluCheckExtension ((const GLubyte*)"GL_OES_packed_depth_stencil",strExt);
	GLEW_OES_compressed_ETC1_RGB8_texture = gluCheckExtension ((const GLubyte*)"GL_OES_compressed_ETC1_RGB8_texture",strExt);
	GLEW_IMG_texture_compression_pvrtc = gluCheckExtension ((const GLubyte*)"GL_IMG_texture_compression_pvrtc",strExt);
#else
	GLEW_EXT_framebuffer_object = gluCheckExtension ((const GLubyte*)"GL_EXT_framebuffer_object",strExt);
	GLEW_ARB_texture_float = gluCheckExtension ((const GLubyte*)"GL_ARB_texture_float",strExt);
	GLEW_EXT_packed_depth_stencil = gluCheckExtension ((const GLubyte*)"GL_EXT_packed_depth_stencil",strExt);
#endif
    
#ifndef GL_UNIFORM_BUFFER
    if (GLEW_VERSION_3_1 || GLEW_ARB_uniform_buffer_object)
    {
        glBindBufferBase = (PFNGLBINDBUFFERBASEPROC) gl_GetProcAddress("glBindBufferBase");
        glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)gl_GetProcAddress("glGetUniformBlockIndex");
        glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC)gl_GetProcAddress("glUniformBlockBinding");
    }
#endif
    
	return GLEW_OK;
}

#ifdef IOS

/*-----opengl context init helper class----*/
@interface HQIOSOpenGLContextInitHelper : NSObject
{
@public
	CAEAGLLayer *eaglLayer;
	const NSString* pixelFormat;
}

- (void) changeLayerProperties ; 

@end


@implementation HQIOSOpenGLContextInitHelper

- (void) changeLayerProperties
{
	eaglLayer.opaque = TRUE;
	eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys: [NSNumber numberWithBool:FALSE],
									kEAGLDrawablePropertyRetainedBacking,
									pixelFormat, kEAGLDrawablePropertyColorFormat, nil];
}

@end

/*------------------------------*/
@interface HQIOSOpenGLContextInitArgs : NSObject
{
@public FORMAT colorFormat, depthStencilFmt;
	    BOOL succeeded;
}


@end

@implementation HQIOSOpenGLContextInitArgs


@end



/*------opengl context class --------*/

@implementation HQIOSOpenGLContext


-(void) changeEAGLLayerPropertiesWithColorFormat: (FORMAT) colorFormat
{
	//change eagl layer's properties in main thread
	HQIOSOpenGLContextInitHelper *helper = [[HQIOSOpenGLContextInitHelper alloc] init];
	helper->eaglLayer = self->eaglLayer;
	helper->pixelFormat = helper::GetEAGLColorFormat(colorFormat);
	
	[helper performSelectorOnMainThread:@selector(changeLayerProperties) withObject:nil waitUntilDone:YES];
	
	[helper release];
}

- (BOOL) createRenderBuffersWithDepthStencilBufferFormat: (FORMAT) dsFormat
{
	GLenum depthStencilFmt = helper::GetEAGLDepthStencilFormat(dsFormat);
	
	//create render buffer
	
	[super renderbufferStorage:GL_RENDERBUFFER fromDrawable:self->eaglLayer];
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self->colorRenderBuffer);
	
	GLint width , height;
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height);
	
	
	GLenum status;
	
	
	if (depthStencilFmt) {
		glGenRenderbuffers(NUM_DS_BUFFERS, self->depthStencilRenderBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, self->depthStencilRenderBuffer[0]);
		if (depthStencilFmt == GL_DEPTH24_STENCIL8_OES)
		{
#ifdef SIMULATOR
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24_OES, width, height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self->depthStencilRenderBuffer[0]);
			
			status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			
			if (status == GL_FRAMEBUFFER_COMPLETE)
			{
				glBindRenderbuffer(GL_RENDERBUFFER, self->depthStencilRenderBuffer[1]);
				glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, width, height);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self->depthStencilRenderBuffer[1]);
			}
			
#else
			glRenderbufferStorage(GL_RENDERBUFFER, depthStencilFmt, width, height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self->depthStencilRenderBuffer[0]);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self->depthStencilRenderBuffer[0]);
#endif
		}
		else{
			glRenderbufferStorage(GL_RENDERBUFFER, depthStencilFmt, width, height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self->depthStencilRenderBuffer[0]);
		}
	}
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		return NO;
	}
	
	return YES;
	
}

- (void) createRenderBuffersOnMainThread: (id) args
{
	HQIOSOpenGLContextInitArgs* myArgs = (HQIOSOpenGLContextInitArgs*)args;
	
	[EAGLContext setCurrentContext:self];
	
	//create default frame buffer
	glGenFramebuffers(1, &self->frameBuffer);
	glGenRenderbuffers(1, &self->colorRenderBuffer);
	
	glBindFramebuffer(GL_FRAMEBUFFER, self->frameBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, self->colorRenderBuffer);
	
	//change eagllayer properties
	[self changeEAGLLayerPropertiesWithColorFormat: myArgs->colorFormat];
	
	
	//create color and depth stencil buffer
	myArgs->succeeded = [self createRenderBuffersWithDepthStencilBufferFormat:myArgs->depthStencilFmt];
	
	if (myArgs->succeeded == NO)
	{
		glDeleteFramebuffers(1, &self->frameBuffer);
		glDeleteRenderbuffers(1, &self->colorRenderBuffer);
		
		
		[EAGLContext setCurrentContext:nil];
		[super release];
	}
	
	[EAGLContext setCurrentContext:nil];
	
}

- (id)initWithAPI:(EAGLRenderingAPI)api 
	withEAGLLayer: (CAEAGLLayer *)layer 
   andColorFormat: (FORMAT)colorFormat 
andDepthStencilFormat: (FORMAT) depthStencilFmt
{
	if ((self = [super initWithAPI: api]) != nil)
	{
		
		self->frameBuffer = 0;
		self->colorRenderBuffer = 0;
		for (int i = 0; i < NUM_DS_BUFFERS ; ++i)
			self->depthStencilRenderBuffer[i] = 0;
		
		self->eaglLayer = layer;
		
		HQIOSOpenGLContextInitArgs* args = [[HQIOSOpenGLContextInitArgs alloc] init];
		args->colorFormat = colorFormat;
		args->depthStencilFmt = depthStencilFmt;
		args->succeeded = NO;
		
		//perform create frame buffer in main thread
		[self performSelectorOnMainThread:@selector(createRenderBuffersOnMainThread:) withObject:args waitUntilDone:TRUE];
		
		
		if(args->succeeded != YES)
		{
			[args release];
			return nil;
		}
		
		[args release];
		
		[EAGLContext setCurrentContext:self];
		
		return self;
	}
	
	return self;
}

#define PRESENT_ON_MAIN_THREAD 0

#if PRESENT_ON_MAIN_THREAD
- (void)presentRenderbuffer
{
	[EAGLContext setCurrentContext:nil];
	//present back buffer in main thread
	[self performSelectorOnMainThread:@selector(performPresenRenderbufferOnMainThread) withObject:nil waitUntilDone:TRUE];
	[EAGLContext setCurrentContext:self];//switch back to this thread
}
- (void) performPresenRenderbufferOnMainThread
{
	[EAGLContext setCurrentContext:self];
	glBindRenderbuffer(GL_RENDERBUFFER, self->colorRenderBuffer);
	[super presentRenderbuffer:self->colorRenderBuffer];
	[EAGLContext setCurrentContext:nil];
}

#else
- (void)presentRenderbuffer
{
	glBindRenderbuffer(GL_RENDERBUFFER, self->colorRenderBuffer);
	[super presentRenderbuffer:self->colorRenderBuffer];
}

#endif

- (GLuint) getFrameBuffer
{
	return self->frameBuffer;
}


- (oneway void) release
{
	glDeleteFramebuffers(1, &self->frameBuffer);
	
	glDeleteRenderbuffers(1, &self->colorRenderBuffer);
	
	glDeleteRenderbuffers(NUM_DS_BUFFERS, self->depthStencilRenderBuffer);
	
	[EAGLContext setCurrentContext:nil];
	
	[super release];
}

@end


#else

@implementation HQAppleOpenGLContext

-(id)initWithFormat:(NSOpenGLPixelFormat *)format andView :(NSView *) _view
	andViewWidthPointer : (uint *) pWidth  andViewHeightPointer : (uint*) pHeight
{
	self = [super initWithFormat : format shareContext : nil];
	if (self != nil)
	{
		self->view = _view;
		self->pViewWidth = pWidth;
		self->pViewHeight = pHeight;
		[[NSNotificationCenter defaultCenter] addObserver:self
											selector:@selector(surfaceNeedsUpdate:) 
											name:NSViewGlobalFrameDidChangeNotification 
											object:_view];
	}
	return self;
}

-(void) surfaceNeedsUpdate :(NSNotification*)notification
{
	NSRect rect = [self->view frame];
	*self->pViewWidth = rect.size.width;
	*self->pViewHeight = rect.size.height;
	[super update];
}

-(void) makeCurrentContext
{
	if ([super view] != self->view)
		[super setView : self->view];
	[super makeCurrentContext];
}

/*

- (void) changeResoluton :(GLint) width : (GLint)height
{
	CGLContextObj cglContext = (CGLContextObj) [super CGLContextObj];
	NSRect rect = [self->view frame];
	if (width == rect.size.width && height == rect.size.height)
		CGLDisable (cglContext, kCGLCESurfaceBackingSize);
	else {
		GLint dim[2] = {width, height}; 
		CGLSetParameter(cglContext, kCGLCPSurfaceBackingSize, dim); 
		CGLEnable (cglContext, kCGLCESurfaceBackingSize);
	}

		
}
*/

- (oneway void)release {
	[[NSNotificationCenter defaultCenter] removeObserver:self
										name:NSViewGlobalFrameDidChangeNotification
										object:self->view]; 
	[super release];
}

@end
#endif


