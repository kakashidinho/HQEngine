/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_RENDERER_
#define _HQ_RENDERER_


#include "HQRenderDevice.h"

#if !defined HQ_STATIC_ENGINE && (defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM)
#	define HQ_DYNAMIC_RENDERER_MANAGER
#endif

#ifndef HQ_RENDERER_API
#	ifndef HQ_DYNAMIC_RENDERER_MANAGER
#		define HQ_RENDERER_API
#	else
#		if defined WIN32 || defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
#			ifdef HQRENDERER_EXPORTS
#				define HQ_RENDERER_API __declspec(dllexport)
#			else
#				define HQ_RENDERER_API __declspec(dllimport)
#			endif
#		else
#				define HQ_RENDERER_API __attribute__ ((visibility("default")))
#		endif
#	endif
#endif

#if defined HQ_IPHONE_PLATFORM || defined HQ_ANDROID_PLATFORM || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	ifndef _STATIC_RENDERER_LIB_
#		define _STATIC_RENDERER_LIB_
#	endif 
#elif defined HQ_MAC_PLATFORM
#   define _NO_DYNAMIC_LOAD_RENDERER_LIB_
#elif defined HQ_LINUX_PLATFORM
#   define _NO_DYNAMIC_LOAD_RENDERER_LIB_
#endif


#ifdef _STATIC_RENDERER_LIB_
#   define _NO_DYNAMIC_LOAD_RENDERER_LIB_
#endif

///
///Render Device Factory
///

class HQ_RENDERER_API HQRenderer{

public:
	///
	///{debugLayer} is ignored in release build
	///
	HQRenderer(bool debugLayer = false);

	~HQRenderer();
	///create Direct3D9 device
	HQReturnVal CreateD3DDevice9(bool flushDebugLog=false);
	//HQReturnVal CreateD3DDevice10(bool flushDebugLog=false);//create Direct3D10 device

	///create Direct3D11 device
	HQReturnVal CreateD3DDevice11(bool flushDebugLog=false);

#ifdef LINUX
	///create openGL
	HQReturnVal CreateGLDevice(Display *dpy , bool flushDebugLog=false);
#else
	///create openGL or openGL ES 2.x/1.x device
	HQReturnVal CreateGLDevice(bool flushDebugLog=false);
#endif
	///release device and resource for creating that device
	void Release();

	LPHQRenderDevice GetDevice();
    
#ifndef _NO_DYNAMIC_LOAD_RENDERER_LIB_
	hModule GetModule(){return pDll;};
#endif
    
private:
	LPHQRenderDevice pDevice;
    
#ifndef _NO_DYNAMIC_LOAD_RENDERER_LIB_
	hModule pDll;
#endif
	int APIType ;
	bool debug;
};
typedef HQRenderer *LPHQRenderer;
#endif
