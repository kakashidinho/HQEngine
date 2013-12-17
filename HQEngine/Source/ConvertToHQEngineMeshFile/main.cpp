/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/
#if defined (_DEBUG) || defined(DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "main.h"

#if defined DEBUG || defined _DEBUG
#	pragma comment(lib,"../../VS/Output/StaticDebug/HQUtilMathD.lib")
#else
#	pragma comment(lib,"../../VS/Output/StaticRelease/HQUtilMath.lib")
#endif

int main(int argc, char **argv)
{
#if defined (_DEBUG) || defined(DEBUG)
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	//_crtBreakAlloc = 245;

#endif

	if (argc != 3)
		return -1;

	//check if this is .x file
	bool isXFile = false;
	const char *ext = strrchr(argv[1], '.');
	if (ext != NULL)
	{
		ext += 1;
		if (!strcmp(ext,"X") || !strcmp(ext,"x"))
			isXFile = true;
	}

	if (isXFile)
	{
		//create dummy window and dummy d3d device
		CreateWindowHandle();
		CreateDirect3dDevice();
		
		ConvertXToHQMeshFile(argv[2], argv[1]);
		
		CleanUpDirectX();
	}
	else
	{
		ConvertToHQMeshFile(argv[2], argv[1]);
	}

	return 0;
}

char* GetAnimationFileName(const char* destMeshFile)
{
	const char *ext = strrchr(destMeshFile, '.');
	unsigned long leng = ext - destMeshFile + 1;
	char *animFileName = new char [leng + 1 + strlen("hqanimation")];
	strncpy(animFileName, destMeshFile, leng);
	strcpy(animFileName + leng, "hqanimation");

	return animFileName;
}

