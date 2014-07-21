/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#if defined (_DEBUG) || defined(DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "main.h"

#if defined DEBUG || defined _DEBUG
#	pragma comment(lib,"../../HQEngine/VS2008/Output/StaticDebug/HQUtilMathD.lib")
#else
#	pragma comment(lib,"../../HQEngine/VS2008/Output/StaticRelease/HQUtilMath.lib")
#endif

int main(int argc, char **argv)
{
#if defined (_DEBUG) || defined(DEBUG)
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	//_crtBreakAlloc = 245;

#endif

	if (argc < 3)
	{
		printf("usage: <input> [options] <output>\n");
		return -1;
	}

	//parse options
	int flags = 0;
	for (int i = 2; i < argc - 1; ++i){
		if (strcmp(argv[i], "-flat") == 0)
			flags |= FLAG_FLAT_FACES;//flat faces model
		else if (strcmp(argv[i], "-32bidx") == 0)
			flags |= FLAG_FORCE_32BIT_INDICES;//use 32 bit index data
		else if (!strcmp(argv[i], "-info"))
			flags |= FLAG_OUTPUT_ADDITIONAL_INFO;//output more info
	}

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
		
		ConvertXToHQMeshFile(argv[argc - 1], argv[1], flags);
		
		CleanUpDirectX();
	}
	else
	{
		ConvertToHQMeshFile(argv[argc - 1], argv[1], flags);
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

char* GetMoreInfoFileName(const char* destMeshFile)
{
	unsigned long leng = strlen(destMeshFile) + 1;
	char *infoFileName = new char [leng + strlen(".hqmeshinfo")];
	strncpy(infoFileName, destMeshFile, leng - 1);
	strcpy(infoFileName + leng - 1, ".hqmeshinfo");

	return infoFileName;
}

void WriteMoreInfo(const char* destMeshFile, const MeshAdditionalInfo &info)
{
	char * infoFileName = GetMoreInfoFileName(destMeshFile);

	FILE *f = fopen(infoFileName, "w");

	if (f)
	{
		fprintf (f, "%f %f %f\n%f %f %f\n", info.bboxMin.x, info.bboxMin.y, info.bboxMin.z, 
			info.bboxMax.x, info.bboxMax.y, info.bboxMax.z);
		fprintf(f, "%f\n", info.meshSurfArea);
		fclose(f);
	}

	delete[] infoFileName;
}

