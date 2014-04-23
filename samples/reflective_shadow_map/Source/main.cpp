/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "RenderLoop.h"




//main function
int HQEngineMain(int argc, char **argv)
{
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#	if defined (_DEBUG) || defined(DEBUG)
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	//_crtBreakAlloc = 2889;

#	endif
#endif
	
#if defined WIN32
#	if defined USE_D3D9 
	const char renderAPI[] = "D3D9";//"D3D9" or "GL"
#	elif defined USE_D3D11
	const char renderAPI[] = "D3D11";//"D3D9" or "GL"
#	else
	const char renderAPI[] = "GL";//"D3D9" or "GL"
#	endif
#else
	const char renderAPI[] = "GL";//"D3D9" or "GL"
#endif
	
	//create log stream
	HQLogStream *logStream = HQCreateFileLogStream("log.txt");

	//create engine instance and its render device
#ifdef USE_GL3
	const char additionalOptions[] = "Core-GL3.0";
#else
	const char additionalOptions[] = "";
#endif
	
	//create rendering loop
	RenderLoop* loop = new RenderLoop(renderAPI, logStream, additionalOptions);

	//start rendering loop
	loop->Run();

	//clean up
	delete loop;

	logStream->Close();

	return 0;
}