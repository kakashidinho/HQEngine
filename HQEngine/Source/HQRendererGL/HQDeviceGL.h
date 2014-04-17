/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _DEVICE_GL_
#define _DEVICE_GL_
#include "glHeaders.h"
#include "../BaseImpl/HQRenderDeviceBaseImpl.h"

/*----windows-----*/
#ifdef WIN32

struct WindowInfo
{
	HWND hwind;
	HWND hparent;
	LONG x,y;
	LONG_PTR styles;
};

#	pragma comment(lib,"opengl32.lib")
#	ifndef GLEW_STATIC
#		pragma comment(lib,"glew32.lib")
#	endif

#define gl_GetProcAddress wglGetProcAddress
/*------linux-----*/
#elif defined (HQ_LINUX_PLATFORM)

struct WindowInfo
{
    Window win;
    GLXWindow glxWin;
    GLXDrawable swapDrawable;
    Window parent;
    int x,y;
    Colormap cmap;
};


/*----Iphone OS----*/
#elif defined HQ_IPHONE_PLATFORM



/*----MacOS--------*/
#elif defined HQ_MAC_PLATFORM

#include <ApplicationServices/ApplicationServices.h>

#endif

/*-----------------*/



#include "HQDeviceEnumGL.h"
#include "HQTextureManagerGL.h"
#include "HQShaderGL.h"
#include "HQRenderTargetGL.h"
#include "HQStateManagerGL.h"
#include "HQVertexStreamManagerGL.h"

//****************
//device
//*****************


class HQDeviceGL:public HQBaseRenderDevice 
{
protected:
	~HQDeviceGL();
	int CreateContext(HQRenderDeviceInitInput input, const char* coreProfile);


#if defined DEVICE_LOST_POSSIBLE
	void OnLost();
	void OnReset();
#endif

	void OnFinishInitDevice(int shaderManagerType);
	void EnableVSyncNonSave(bool enable);

#ifdef WIN32
	HMODULE pDll;

	HDC hDC;//device context
	HGLRC hRC;//render context
    WindowInfo winfo;//window info
#elif defined (HQ_LINUX_PLATFORM)
    Display *dpy;
	WindowInfo winfo;
	GLXContext glc;
#elif defined HQ_IPHONE_PLATFORM
	HQIOSOpenGLContext *glc;
#elif defined HQ_MAC_PLATFORM
	HQAppleOpenGLContext * glc;
#elif defined HQ_ANDROID_PLATFORM
	HQAndroidOpenGLContext *glc;
	jobject jeglConfig;
#endif

	HQColor clearColor;
	hqfloat32 clearDepth;
	hquint32 clearStencil;

	GLenum primitiveMode;
	GLenum primitiveLookupTable[HQ_PRI_NUM_PRIMITIVE_MODE];

	bool usingCoreProfile;
	HQDeviceEnumGL *pEnum;
public:
#ifdef WIN32
	HQDeviceGL(HMODULE _pDll, bool flushLog);
#elif defined HQ_LINUX_PLATFORM
    HQDeviceGL(Display *dpy, bool flushLog);
#else
	HQDeviceGL(bool flushLog);
#endif

	HQReturnVal Release();

#if defined DEVICE_LOST_POSSIBLE
	bool IsDeviceLost();
#endif

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

	void GetClearColor(HQColor &clearColorOut) const 
	{
		clearColorOut = clearColor;
	}

	hqfloat32 GetClearDepthVal() const 
	{
		return clearDepth;
	}

	hquint32 GetClearStencilVal() const
	{
		return clearStencil;
	}

	bool IsUsingCoreProfile() const {
		return usingCoreProfile;
	}

	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions);
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

	HQReturnVal DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ);

	void TextureUAVBarrier();

	/*---------------------------------
	device capabilities
	----------------------------------*/
	inline const Caps& GetDeviceCaps() {return this->pEnum->caps;}
	HQColorLayout GetColoruiLayout() { return CL_RGBA;}
	hq_uint32 GetMaxVertexStream() {return pEnum->caps.maxVertexAttribs;}//bằng số vertex attributes
	hq_uint32 GetMaxVertexAttribs() {return pEnum->caps.maxVertexAttribs;}
	bool IsIndexDataTypeSupported(HQIndexDataType iDataType) {
#ifdef HQ_OPENGLES
		if (iDataType == HQ_IDT_UINT)//openGL es not support 32 bit index
			return false;
#endif
		return true;
	}
	hq_uint32 GetMaxShaderSamplers() {return pEnum->caps.nShaderSamplerUnits;}//truy vấn số texture sampler unit nhiều nhất có thể dùng trong shader
	hq_uint32 GetMaxShaderStageSamplers(HQShaderType shaderStage) ;//truy vấn số texture sampler nhiều nhất có thể dùng trong shader stage <shaderStage>

	hq_uint32 GetMaxShaderTextureUAVs();
	hq_uint32 GetMaxShaderStageTextureUAVs(HQShaderType shaderStage);

	void GetMaxComputeGroups(hquint32 &nGroupsX, hquint32 &nGroupsY, hquint32 &nGroupsZ);

	bool IsTwoSideStencilSupported() {return true;};//is two sided stencil supported 
	bool IsBlendStateExSupported() {return true;};//is extended blend state supported
	bool IsTextureBufferFormatSupported(HQTextureBufferFormat format);
	bool IsUAVTextureFormatSupported(HQTextureUAVFormat format, HQTextureType textureType, bool hasMipmap);
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
#ifdef HQ_OPENGLES
		return 1;
#else
		return pEnum->caps.maxDrawBuffers;
#endif
	}
	
	//check if mipmaps generation for render target texture is supported
	bool IsRTTMipmapGenerationSupported() 
	{
		return GLEW_VERSION_3_0 || GLEW_EXT_framebuffer_object;
	}

	void * GetRawHandle();
};


extern HQDeviceGL* g_pOGLDev;

/*----------------------*/

#endif
