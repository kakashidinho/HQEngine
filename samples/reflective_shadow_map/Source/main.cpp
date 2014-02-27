
#include "RenderLoop.h"




//main function
int HQEngineMain(int argc, char **argv)
{
	//create log stream
	HQLogStream *logStream = HQCreateFileLogStream("log.txt");

	//create engine instance and its render device
	HQEngineApp::WindowInitParams params = HQEngineApp::WindowInitParams::Construct(
			NULL,
			"D3D9",
			NULL,
			NULL,
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
	RenderLoop* loop = new RenderLoop();

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