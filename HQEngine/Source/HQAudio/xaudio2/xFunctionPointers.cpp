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