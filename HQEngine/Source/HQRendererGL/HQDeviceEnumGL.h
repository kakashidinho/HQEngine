/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _DEV_ENUM_
#define _DEV_ENUM_
#include "../HQPlatformDef.h"
#include "../HQLinkedList.h"
#include "../HQRendererCoreType.h"
#ifdef ANDROID
#include <jni.h>
#endif

#define NUM_RTT_FORMAT 9
#define NUM_DS_FORMAT 5

#ifdef LINUX
#	if defined HQ_USE_XFREE86_VIDMODE
#include <X11/extensions/xf86vmode.h>
#	else
#include <X11/extensions/Xrandr.h>
#	endif
#endif

enum FORMAT
{
    SFMT_R8G8B8               = 20,
    SFMT_A8R8G8B8             = 21,
    SFMT_X8R8G8B8             = 22,
    SFMT_R5G6B5               = 23,
    SFMT_X1R5G5B5             = 24,
    SFMT_A1R5G5B5             = 25,
    SFMT_A4R4G4B4             = 26,
    SFMT_R3G3B2               = 27,
    SFMT_A8                   = 28,
    SFMT_A8R3G3B2             = 29,
    SFMT_X4R4G4B4             = 30,
    SFMT_A2B10G10R10          = 31,
    SFMT_A8B8G8R8             = 32,
    SFMT_X8B8G8R8             = 33,
    SFMT_D32                  = 71,
    SFMT_D15S1                = 73,
    SFMT_D24S8                = 75,
    SFMT_D24X8                = 77,
    SFMT_D24X4S4              = 79,
    SFMT_D16                  = 80,
    SFMT_S8		       = 82,
    SFMT_NODEPTHSTENCIL	       = 81
};

struct Resolution : public HQResolution
{
#ifdef WIN32 /*-------windows----*/
    DEVMODE w32DisplayMode;
#elif defined LINUX /*-----linux----------*/
#	if defined HQ_USE_XFREE86_VIDMODE
    XF86VidModeModeInfo * x11DisplayMode;
#	else
    int x11ScreenSizeIndex;
#	endif
    double x11RefreshRate;
    
#elif defined HQ_MAC_PLATFORM /*--------mac OSX------*/
    CGDisplayModeRef cgDisplayMode;
#endif
};

struct BufferInfo{
	FORMAT pixelFmt;//pixel buffer format
	FORMAT depthStencilFmt;//depth stenctil buffer format
	DWORD maxMulSampleLevel;//kiểu siêu lấy mẫu tốt nhất có thể đạt dc  (0,2X,4X,8X..v.v.v.)
};

struct Caps//device capabilities
{
	DWORD maxVertexAttribs;
	DWORD maxTextureSize;
	DWORD maxCubeTextureSize;
	DWORD maxAF;//max anisotropic level
	DWORD nShaderSamplerUnits;//number of shader sampler units
	DWORD nFragmentShaderSamplers;
	DWORD nGeometryShaderSamplers;
	DWORD nVertexShaderSamplers;
	DWORD nFFTextureUnits;
	bool rttInternalFormat[NUM_RTT_FORMAT];//list of supported render target texture 's internal format
	bool dsFormat[NUM_DS_FORMAT];
#ifndef GLES
	DWORD maxDrawBuffers;
	DWORD maxUniformBufferSlots;
#endif
#if defined HQ_MAC_PLATFORM
	bool hardwareAccel;
#endif
};

class HQDeviceEnumGL{

public:
#ifdef WIN32
	HQDeviceEnumGL(HMODULE pDll);

	void SetDC(HDC hDC) {this->hDC = hDC;}
#elif defined LINUX
	HQDeviceEnumGL(Display *dpy);
#elif defined ANDROID
	HQDeviceEnumGL(jobject jegl, jobject jdisplay, jint apiLevel);
#else
	HQDeviceEnumGL();
#endif
	~HQDeviceEnumGL();

	void CheckCapabilities();//check opengl capabilities

#ifdef HQ_MAC_PLATFORM
	void ParseSettingFile(const char* settingFile , hq_uint32 width , hq_uint32 height , bool windowed);
#else
	void ParseSettingFile(const char* settingFile);
#endif
	void SaveSettingFile(const char* settingFile);

#ifdef ANDROID
	jobject GetJEGLConfig();//create global reference
#elif !defined GLES
	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions);
	void EnumAllDisplayModes();
	bool ChangeSelectedDisplay(hq_uint32 width,hq_uint32 height , bool windowed);
#endif

	/*------attributes-------*/
	Caps caps;//device capabilities

#ifndef GLES
	Resolution *selectedResolution;
	BufferInfo* selectedBufferSetting;
	bool windowed;
#else
	FORMAT selectedPixelFormat;
#	ifdef ANDROID
	jint selectedApiLevel;//selected OpenGL ES version
#	endif
#endif
	int vsync;//vsync enable or not
	FORMAT selectedDepthStencilFmt;
	DWORD selectedMulSampleType;//multisample level

#ifdef WIN32
	DEVMODE currentScreenDisplayMode;
#elif defined LINUX
#	if defined HQ_USE_XFREE86_VIDMODE
	XF86VidModeModeInfo * currentScreenDisplayMode;
#	else
	int currentScreenSizeIndex;//id of current screen size
	Rotation currentScreenRotation;
	XRRScreenConfiguration *screenConfig;
#	endif
#elif defined HQ_MAC_PLATFORM
	CGDisplayModeRef currentScreenDisplayMode;
#endif
	
private:

#ifndef GLES
	HQLinkedList<Resolution> reslist;
	HQLinkedList<BufferInfo> bufferInfoList;

	Resolution customWindowedRes;//lưu tùy chọn dành cho chế độ windowed , không giới hạn kích thước width x height như chế độ fullscreen
#endif
	int unUseValue[4];//4 giá trị parse từ setting file không dùng,vì chúng dành cho dạng device khác(ví dụ Direct3D)
	//dialog box items'handles
#ifdef WIN32

	HMODULE pDll;
	HDC hDC;

#elif defined LINUX

	Display *dpy;
#	if defined HQ_USE_XFREE86_VIDMODE
	XF86VidModeModeInfo **modes;//list of display mode
	int modeNum;
#	endif

#elif defined HQ_MAC_PLATFORM

	CFArrayRef modeList;//list of display mode
#elif defined ANDROID

	jobject jegl;//java egl object
	jobject jdisplay;//java EGLDisplay object
	
	void GetPixelFormat(JNIEnv *jenv , jobject jeglConfig , 
							jint &red, jint& green, jint& blue, jint& alpha,
							jint &depth, jint& stencil);

#endif//#ifdef WIN32
#ifndef GLES
	bool CheckPixelFmt(FORMAT format);
	bool CheckDepthStencilFmt(BufferInfo& bufInfo);
	bool CheckMultisample(BufferInfo &bufInfo);
#endif

	void CheckRenderBufferFormatSupported();
};

namespace helper{
#ifdef HQ_IPHONE_PLATFORM
	const NSString* GetEAGLColorFormat(FORMAT fmt);
	uint GetEAGLDepthStencilFormat(FORMAT depthStencilFmt);
#else
	void FormatInfo(FORMAT fmt, hq_ubyte8 *pRGBBitCount,
					hq_ubyte8 *pRBits,
					hq_ubyte8 *pGBits,
					hq_ubyte8 *pBBits,
					hq_ubyte8 *pABits,
					hq_ubyte8 *pDBits,hq_ubyte8 *pSBits);//depth,stencil
#endif
};
#endif
