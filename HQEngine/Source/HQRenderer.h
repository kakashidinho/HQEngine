/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
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

#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	ifndef _STATIC_RENDERER_LIB_
#		define _STATIC_RENDERER_LIB_
#	endif 
#elif defined __APPLE__
#	ifndef APPLE
#		define APPLE __APPLE__
#	endif
#endif

class HQ_RENDERER_API HQRenderer{
private:
	LPHQRenderDevice pDevice;

#ifndef _STATIC_RENDERER_LIB_
	hModule pDll;
#endif
	int APIType ;
	bool debug;

public:
	///
	///{debugLayer} is ignored in release build
	///
	HQRenderer(bool debugLayer = false);

	~HQRenderer();
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#	if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM
	///create Direct3D9 device
	HQReturnVal CreateD3DDevice9(bool flushDebugLog=false);
	//HQReturnVal CreateD3DDevice10(bool flushDebugLog=false);//create Direct3D10 device
#	endif

	///create Direct3D11 device
	HQReturnVal CreateD3DDevice11(bool flushDebugLog=false);
#endif

#ifdef LINUX
	///create openGL
	HQReturnVal CreateGLDevice(Display *dpy , bool flushDebugLog=false);
#else
#	if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM//windows phone doesn't support opengl es
	///create openGL or openGL ES 2.x/1.x device
	HQReturnVal CreateGLDevice(bool flushDebugLog=false);
#	endif
#endif
	///release device and resource for creating that device
	void Release();

	LPHQRenderDevice GetDevice();
#ifndef _STATIC_RENDERER_LIB_
	hModule GetModule(){return pDll;};
#endif
};
typedef HQRenderer *LPHQRenderer;
#endif
