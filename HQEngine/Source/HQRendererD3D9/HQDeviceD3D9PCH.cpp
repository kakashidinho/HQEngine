// stdafx.cpp : source file that includes just the standard includes
// HQEngine_RendererD3D.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "HQDeviceD3D9PCH.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

#if defined _DEBUG || defined DEBUG
void SafeReleaseD(IUnknown * p)
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
#endif