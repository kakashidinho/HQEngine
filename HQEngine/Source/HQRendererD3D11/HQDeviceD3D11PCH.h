// HQDeviceD3D11PCH.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include "targetver.h"
#endif

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

// TODO: reference additional headers your program requires here
#include <iostream>
#include <fstream>

#ifndef SafeRelease
#	if defined _DEBUG || defined DEBUG

#include <Unknwn.h>

inline void SafeReleaseD(IUnknown * p)
{
	if(p){
		ULONG refCount = p->Release();
		if (refCount > 0)
		{
			char debugStr[5];
			sprintf(debugStr , "%u\n" , refCount);
			OutputDebugStringA(debugStr);
		}
	}
}

#		define SafeRelease(p) {SafeReleaseD(p) ; p = 0; }
#	else
#		define SafeRelease(p) {if(p){p->Release();p=0;}}
#	endif
#endif

#include "../HQPlatformDef.h"
#include "../HQEngineCustomHeap.h"
#include "../HQItemManager.h"
#include "../HQLoggableObject.h"
#include "../HQLinkedList.h"
#include "../HQRendererCoreType.h"
#include "../HQRendererCoreTypeInline.h"
#include "../HQReturnVal.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "HQDeviceD3D11.h"

#ifndef SafeDelete
#define SafeDelete(p){if(p){delete p;p=0;}}
#endif
#ifndef SafeDeleteArray
#define SafeDeleteArray(p){if(p){delete[] p;p=0;}}
#endif