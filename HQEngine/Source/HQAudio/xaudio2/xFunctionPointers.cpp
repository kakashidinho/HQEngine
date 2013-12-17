/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "xFunctionPointers.h"

HMODULE XAudio2Lib = NULL;
HMODULE X3DAudioLib = NULL;

fpX3DAudioInitialize_t fpX3DAudioInitialize = NULL;
fpX3DAudioCalculate_t fpX3DAudioCalculate = NULL;

bool InitXAudioFunctions()
{
#if defined _DEBUG || defined DEBUG
	XAudio2Lib = LoadLibraryA("XAudioD2_7.dll");
	if (XAudio2Lib == NULL)//try again with nondebug dll
		XAudio2Lib = LoadLibraryA("XAudio2_7.dll");
	X3DAudioLib = LoadLibraryA("X3DAudioD1_7.dll");
	if (X3DAudioLib == NULL)//try again with nondebug dll
		X3DAudioLib = LoadLibraryA("X3DAudio1_7.dll");
#else
	XAudio2Lib = LoadLibraryA("XAudio2_7.dll");
	X3DAudioLib = LoadLibraryA("X3DAudio1_7.dll");
#endif
	if (XAudio2Lib == NULL || X3DAudioLib == NULL)
		return false;

	fpX3DAudioInitialize = (fpX3DAudioInitialize_t )GetProcAddress(X3DAudioLib, "X3DAudioInitialize");
	fpX3DAudioCalculate = (fpX3DAudioCalculate_t )GetProcAddress(X3DAudioLib, "X3DAudioCalculate");

	if (fpX3DAudioInitialize == NULL || fpX3DAudioCalculate == NULL)
	{
		ReleaseXAudioFunctions();
		return false;
	}

	return true;
}

void ReleaseXAudioFunctions()
{
	if (X3DAudioLib)
	{
		FreeLibrary(X3DAudioLib);
		X3DAudioLib = NULL;
	}
	if (XAudio2Lib)
	{
		FreeLibrary(XAudio2Lib);
		XAudio2Lib = NULL;
	}
}
