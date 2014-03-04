/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQPlatformDef.h"
#include "../HQEngineCustomHeap.h"
#include "../HQRenderer.h"
#include "HQRendererDeviceDebugLayer.h"
#if defined LINUX || defined __APPLE__
#include <dlfcn.h>
#endif


#if defined _DEBUG || defined DEBUG
HQRenderDeviceDebugLayer debugLayer;
#endif

#ifdef _STATIC_RENDERER_LIB_/*---function declare-----*/
extern "C" {
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
	extern HQReturnVal CreateDevice(hModule pDll,LPHQRenderDevice *ppDev,bool flushDebugLog , bool debugLayer);
#elif defined LINUX
	extern HQReturnVal CreateDevice(Display *display,LPHQRenderDevice *ppDev,bool flushDebugLog , bool debugLayer);
#else
	extern HQReturnVal CreateDevice(LPHQRenderDevice *ppDev,bool flushDebugLog , bool debugLayer);
#endif
	
//**************************************
//safe release device
//**************************************
	extern HQReturnVal ReleaseDevice(LPHQRenderDevice * ppDev);
}

#else /*---function pointer-*/
//====================================================
#ifdef WIN32
typedef HQReturnVal (*CreateDev)(HMODULE,LPHQRenderDevice *,bool , bool );
#elif defined LINUX
typedef HQReturnVal (*CreateDev)(Display*,LPHQRenderDevice *,bool , bool );
#else
typedef HQReturnVal (*CreateDev)(LPHQRenderDevice *,bool , bool);
#endif

typedef HQReturnVal (*pReleaseDevice)(LPHQRenderDevice * );
//====================================================

#endif


#define D3D9_API 0
#define D3D10_API 1
#define GL_API 2
#define D3D11_API 3
HQRenderer::HQRenderer(bool debugLayer) 
#if defined _DEBUG || defined DEBUG
: debug(debugLayer)
#endif
{
#ifndef _STATIC_RENDERER_LIB_
	pDll=NULL;
#endif
	pDevice=NULL;
	APIType = 0xffffffff;
}
HQRenderer::~HQRenderer(){
	Release();
}

HQRenderDevice * HQRenderer::GetDevice()
{
#if defined _DEBUG || defined DEBUG
	if (debug)
	{
		if (pDevice == NULL)
			return NULL;
		debugLayer.SetDevice( pDevice );
		return &debugLayer;
	}
#endif
	return pDevice;
}

void HQRenderer::Release()
{
#ifndef _STATIC_RENDERER_LIB_
#	ifdef WIN32
    pReleaseDevice ReleaseDevice = (pReleaseDevice)GetProcAddress(pDll,"ReleaseDevice");
#	elif defined (LINUX) || defined __APPLE__
    pReleaseDevice ReleaseDevice = (pReleaseDevice)dlsym(pDll,"ReleaseDevice");
#	endif
#endif
	if(pDevice)
	{
        ReleaseDevice(&pDevice);
	}
#ifndef _STATIC_RENDERER_LIB_
	if(pDll)
	{
#	ifdef WIN32
		FreeLibrary(pDll);
#	elif defined (LINUX) || defined __APPLE__
		dlclose(pDll);
#	endif
		pDll=NULL;
	}
#endif
}
#if defined WIN32

#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM
HQReturnVal HQRenderer::CreateD3DDevice9(bool flushDebugLog){
#ifndef _STATIC_RENDERER_LIB_
	if(pDll != NULL && APIType != D3D9_API)
	{
		FreeLibrary(pDll);
		pDll = NULL;
	}
	if(pDll == NULL)
	{
#	if defined DEBUG || defined _DEBUG
		pDll=LoadLibrary(L"HQRendererD3D9_D.dll");
#	else
		pDll=LoadLibrary(L"HQRendererD3D9.dll");
#	endif
		if(!pDll)
			return HQ_FAILED_LOAD_LIBRARY;
		APIType = D3D9_API;
	}

	CreateDev CreateDevice=0;
	CreateDevice=(CreateDev)GetProcAddress(pDll,"CreateDevice");

	if(!CreateDevice)
		return HQ_FAILED_LOAD_LIBRARY;
#else
	hModule pDll = GetModuleHandle(NULL);
#endif
	HQReturnVal re=CreateDevice(pDll,&pDevice,flushDebugLog , debug);
	if (re == HQ_DEVICE_ALREADY_EXISTS)
		return re;
	if(HQFailed(re))
	{
		pDevice=0;
		return re;
	}
	return HQ_OK;
}

#if 0

HQReturnVal HQRenderer::CreateD3DDevice10(bool flushDebugLog){
#ifndef _STATIC_RENDERER_LIB_
	if(pDll != NULL && APIType != D3D10_API)
	{
		FreeLibrary(pDll);
		pDll = NULL;
	}
	if(pDll == NULL)
	{
#	if defined DEBUG || defined _DEBUG
		pDll=LoadLibrary(L"HQRendererD3D10_D.dll");
#	else
		pDll=LoadLibrary(L"HQRendererD3D10.dll");
#	endif

		if(!pDll)
			return HQ_FAILED_LOAD_LIBRARY;
		APIType = D3D10_API;
	}

	CreateDev CreateDevice=0;
	CreateDevice=(CreateDev)GetProcAddress(pDll,"CreateDevice");

	if(!CreateDevice)
		return HQ_FAILED_LOAD_LIBRARY;
#else
	hModule pDll = GetModuleHandle(NULL);
#endif
	HQReturnVal re=CreateDevice(pDll,&pDevice,flushDebugLog , debug);
	if (re == HQ_DEVICE_ALREADY_EXISTS)
		return re;
	if(HQFailed(re))
	{
		pDevice=0;
		return re;
	}


	return HQ_OK;
}
#endif

#endif //#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM

HQReturnVal HQRenderer::CreateD3DDevice11(bool flushDebugLog){
#ifndef _STATIC_RENDERER_LIB_
	if(pDll != NULL && APIType != D3D11_API)
	{
		FreeLibrary(pDll);
		pDll = NULL;
	}
	if(pDll == NULL)
	{
#	if defined DEBUG || defined _DEBUG
		pDll=LoadLibrary(L"HQRendererD3D11_D.dll");
#	else
		pDll=LoadLibrary(L"HQRendererD3D11.dll");
#	endif

		if(!pDll)
			return HQ_FAILED_LOAD_LIBRARY;
		APIType = D3D11_API;
	}

	CreateDev CreateDevice=0;
	CreateDevice=(CreateDev)GetProcAddress(pDll,"CreateDevice");

	if(!CreateDevice)
		return HQ_FAILED_LOAD_LIBRARY;
#elif defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	hModule pDll = GetModuleHandle(NULL);
#else
	hModule pDll = NULL;
#endif
	HQReturnVal re=CreateDevice(pDll,&pDevice,flushDebugLog , debug);
	if (re == HQ_DEVICE_ALREADY_EXISTS)
		return re;
	if(HQFailed(re))
	{
		pDevice=0;
		return re;
	}
	return HQ_OK;
}
#endif

#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM//windows phone doesn't support opengl es
#ifdef LINUX
HQReturnVal HQRenderer::CreateGLDevice(Display *dpy , bool flushDebugLog)
#else
HQReturnVal HQRenderer::CreateGLDevice(bool flushDebugLog)
#endif
{
#ifndef _STATIC_RENDERER_LIB_
	if(pDll != NULL && APIType != GL_API)
	{
#	ifdef WIN32
		FreeLibrary(pDll);
#	elif defined (LINUX) || defined __APPLE__
		dlclose(pDll);
#	endif
		pDll = NULL;
	}
	if(pDll == NULL)
	{
#	ifdef WIN32

#		if defined DEBUG || defined _DEBUG
		pDll=LoadLibrary(L"HQRendererGL_D.dll");
#		else
		pDll=LoadLibrary(L"HQRendererGL.dll");
#		endif

#	elif defined (LINUX)
#		if defined DEBUG || defined _DEBUG
        pDll=dlopen("./libHQEngineRendererDevice_D.so",RTLD_LAZY);
#       else
		pDll=dlopen("./libHQEngineRendererDevice.so",RTLD_LAZY);
#       endif

#	elif defined __APPLE__
#		if defined DEBUG || defined _DEBUG
		pDll=dlopen("./libHQEngineRendererDevice_D.dylib",RTLD_LAZY);
#       else
		pDll=dlopen("./libHQEngineRendererDevice.dylib",RTLD_LAZY);
#       endif
#	endif
		if(!pDll)
		{
#ifndef WIN32
			printf("%s" , dlerror());
#endif
			return HQ_FAILED_LOAD_LIBRARY;
		}
		APIType = GL_API;
	}

	CreateDev CreateDevice=NULL;
#	ifdef WIN32
	CreateDevice=(CreateDev)GetProcAddress(pDll,"CreateDevice");
#	elif defined (LINUX) || defined __APPLE__
    CreateDevice=(CreateDev )dlsym(pDll,"CreateDevice");
#	endif

	if(CreateDevice==NULL)
		return HQ_FAILED_LOAD_LIBRARY;
#elif defined WIN32
	hModule pDll = GetModuleHandle(NULL);
#endif

#ifdef WIN32
	HQReturnVal re=CreateDevice(pDll,&pDevice,flushDebugLog , debug);
#elif defined LINUX
    HQReturnVal re=CreateDevice(dpy,&pDevice,flushDebugLog , debug);
#else
    HQReturnVal re=CreateDevice(&pDevice,flushDebugLog , debug);
#endif
	if (re == HQ_DEVICE_ALREADY_EXISTS)
		return re;
	if(HQFailed(re))
	{
		pDevice=NULL;
		return re;
	}
	return HQ_OK;
}

#endif//#if !defined HQ_WIN_PHONE_PLATFORM && !defined HQ_WIN_STORE_PLATFORM
