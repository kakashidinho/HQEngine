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
	//_crtBreakAlloc = 312;

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