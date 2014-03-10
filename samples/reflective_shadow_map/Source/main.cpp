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
	_crtBreakAlloc = 6835;

#	endif
#endif

	const char renderAPI[] = "GL";//"D3D9" or "GL"

	//create log stream
	HQLogStream *logStream = HQCreateFileLogStream("log.txt");

	//create engine instance and its render device
	HQEngineApp::WindowInitParams params = HQEngineApp::WindowInitParams::Construct(
			NULL,
			renderAPI,
			NULL,
			"GLSL-only",
			logStream,
			true
		);
	HQEngineApp::GetInstance()->CreateInstanceAndWindow(&params, false, NULL);
	HQEngineApp::GetInstance()->AddFileSearchPath("../Data");
#if HQ_RENDER_RESOLUTION_CHANGEABLE 
	HQEngineApp::GetInstance()->GetRenderDevice()->SetDisplayMode(600, 600, true);
	HQViewPort viewport = {0, 0, 600, 600};
	HQEngineApp::GetInstance()->GetRenderDevice()->SetViewPort(viewport);
#endif


	//create rendering loop
	RenderLoop* loop = new RenderLoop(renderAPI);

	//prepare engine 
	HQEngineApp::GetInstance()->SetRenderDelegate(*loop);

	HQEngineApp::GetInstance()->ShowWindow();

	//start rendering loop
	HQEngineApp::GetInstance()->Run();

	//clean up
	delete loop;
	HQEngineApp::Release();

	logStream->Close();

	return 0;
}