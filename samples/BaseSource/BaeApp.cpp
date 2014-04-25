#include "BaseApp.h"

#include <string>
#include <iostream>
#include <fstream>

struct HQPointerClose{
	template <class T>
	static inline void Release(T *&ptr) { if (ptr != NULL) ptr->Close(); ptr = NULL; }
};

HQBaseSharedPtr<HQLogStream, HQPointerClose> defaultLogStream;

BaseApp::BaseApp(const char* rendererAPI,
	HQLogStream *logStream,
	const char * additionalAPISettings,
	hquint32 width, hquint32 height)
	: m_camera(NULL)
{
	Init(rendererAPI, logStream, additionalAPISettings, width, height);
}

BaseApp::BaseApp(hquint32 width, hquint32 height)
	: m_camera(NULL)
{
	//read renderer API config from "API.txt"
	std::string rendererAPI = "D3D11";
	std::ifstream stream("API.txt");
	if (stream.good())
	{
		stream >> rendererAPI;
	}

	stream.close();

	//create default log stream
	defaultLogStream = HQCreateFileLogStream("log.txt");

	Init(rendererAPI.c_str(), defaultLogStream.GetRawPointer(), NULL, width, height);
}

void BaseApp::Init(const char* rendererAPI,
				HQLogStream *logStream,
				const char * additionalAPISettings,
				hquint32 width, hquint32 height)
{
	strcpy(this->m_renderAPI_name, rendererAPI);
	if (strcmp(m_renderAPI_name, "GL") == 0)
	{
		this->m_renderAPI_type = HQ_RA_OGL;
	}
	else if (strcmp(m_renderAPI_name, "D3D11") == 0)
	{
		this->m_renderAPI_type = HQ_RA_D3D;
	}
	else
	{
		this->m_renderAPI_type = HQ_RA_D3D;
	}

	//create window and render device
	HQEngineApp::WindowInitParams params = HQEngineApp::WindowInitParams::Construct(
		NULL,
		rendererAPI,
		NULL,
		additionalAPISettings,
		logStream,
		true
		);
	HQEngineApp::CreateInstanceAndWindow(&params, false, NULL);

	//register renderering delegate
	HQEngineApp::GetInstance()->SetRenderDelegate(*this);

	m_pRDevice = HQEngineApp::GetInstance()->GetRenderDevice();

	m_pRDevice->SetDisplayMode(width, height, true);
	m_pRDevice->SetFullViewPort();

	//prepare GUI engine
	m_guiPlatform = new MyGUI::HQEnginePlatform();

	m_guiPlatform->initialise();
	m_guiPlatform->getDataManagerPtr()->addResourceLocation("../Data", true);
	m_guiPlatform->getDataManagerPtr()->addResourceLocation("../../Data", true);

	m_myGUI = new MyGUI::Gui();
	m_myGUI->initialise();

	MyGUI::LayoutManager::getInstance().loadLayout("sample.layout");
	MyGUI::PointerManager::getInstance().setVisible(false);

	m_fpsTextBox = m_myGUI->findWidget<MyGUI::TextBox>("FPS-Info");

	//create scene container
	m_scene = new HQSceneNode("scene_root"); 

}
BaseApp::~BaseApp()
{
	m_myGUI->shutdown();
	delete m_myGUI;
	m_guiPlatform->shutdown();
	delete m_guiPlatform;

	delete m_camera;

	delete m_scene;

	HQEngineApp::Release();
}

void BaseApp::Run()
{
	//show window
	HQEngineApp::GetInstance()->ShowWindow();

	//start loop
	HQEngineApp::GetInstance()->Run();
}

void BaseApp::Render(HQTime dt)
{
	{
		//update fps
		char fps_text[256];
		sprintf(fps_text, "fps:%.3f", HQEngineApp::GetInstance()->GetFPS());
		m_fpsTextBox->setCaption(fps_text);
	}
	//call update implementation
	this->Update(dt);

	//start rendering
	m_pRDevice->BeginRender(HQ_TRUE, HQ_TRUE, HQ_FALSE);

	//call rendering implementation
	this->RenderImpl(dt);

	//draw GUI
	m_guiPlatform->getRenderManagerPtr()->drawOneFrame();

	m_pRDevice->EndRender();
}