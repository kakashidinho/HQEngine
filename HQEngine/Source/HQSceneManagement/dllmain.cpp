/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

// dllmain.cpp : Defines the entry point for the DLL application.
#include "HQScenePCH.h"

#define FOR_LVTN 0 //lvtn version

#if FOR_LVTN 

#	ifdef WIN32
#		if defined _DEBUG || defined DEBUG
#			pragma comment(lib,"HQUtilMathD.lib")
#		else
#			pragma comment(lib,"HQUtilMath.lib")
#		endif
#	endif

#else

#	ifdef WIN32
#		ifdef _STATIC_CRT
#			if defined _DEBUG || defined DEBUG
#				pragma comment(lib,"../../VS/Output/StaticDebug static CRT/HQUtilMathD.lib")
#			else
#				pragma comment(lib,"../../VS/Output/StaticRelease static CRT/HQUtilMath.lib")
#			endif

#		else
#			if defined _DEBUG || defined DEBUG
#				pragma comment(lib,"../../VS/Output/StaticDebug/HQUtilMathD.lib")
#			else
#				pragma comment(lib,"../../VS/Output/StaticRelease/HQUtilMath.lib")
#			endif
#		endif
#	endif

#endif


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

