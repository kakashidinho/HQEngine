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
	: m_camera(NULL), m_leftMousePressed(false)
{
	Init(rendererAPI, logStream, additionalAPISettings, width, height);
}

BaseApp::BaseApp(const char * additionalAPISettings, hquint32 width, hquint32 height)
: m_camera(NULL), m_leftMousePressed(false)
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

	Init(rendererAPI.c_str(), defaultLogStream.GetRawPointer(), additionalAPISettings, width, height);
}

BaseApp::BaseApp(HQLogStream* logStream, const char * additionalAPISettings, hquint32 width, hquint32 height)
: m_camera(NULL), m_leftMousePressed(false)
{
	//read renderer API config from "API.txt"
	std::string rendererAPI = "D3D11";
	std::ifstream stream("API.txt");
	if (stream.good())
	{
		stream >> rendererAPI;
	}

	stream.close();

	Init(rendererAPI.c_str(), logStream, additionalAPISettings, width, height);
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
	//register listeners
	HQEngineApp::GetInstance()->SetKeyListener(*this);
	HQEngineApp::GetInstance()->SetMouseListener(*this);

	m_pRDevice = HQEngineApp::GetInstance()->GetRenderDevice();
	m_resManager = HQEngineApp::GetInstance()->GetResourceManager();
	m_effectManager = HQEngineApp::GetInstance()->GetEffectManager();

	m_pRDevice->SetDisplayMode(width, height, true);
	m_pRDevice->SetFullViewPort();

	//prepare GUI engine
	m_guiPlatform = new MyGUI::HQEnginePlatform();

	m_guiPlatform->initialise();
	m_guiPlatform->getDataManagerPtr()->addResourceLocation("Data/MyGUI", false);
	m_guiPlatform->getDataManagerPtr()->addResourceLocation("../Data/MyGUI", false);
	m_guiPlatform->getDataManagerPtr()->addResourceLocation("../../Data/MyGUI", false);

	m_myGUI = new MyGUI::Gui();
	m_myGUI->initialise();

	MyGUI::LayoutManager::getInstance().loadLayout("sample.layout");
	MyGUI::PointerManager::getInstance().setVisible(false);

	m_fpsTextBox = m_myGUI->findWidget<MyGUI::TextBox>("FPS-Info");

	//create scene container
	m_scene = new HQSceneNode("scene_root"); 

	m_sceneController = new HQSceneNode("scene_controller");

	//create world container
	m_world = new HQSceneNode("world_root");

	m_world->AddChild(m_sceneController);
	m_sceneController->AddChild(m_scene);

	//by default, camera will stay still every frame
	m_cameraTransition.Set(0, 0, 0);

}
BaseApp::~BaseApp()
{
	m_myGUI->shutdown();
	delete m_myGUI;
	m_guiPlatform->shutdown();
	delete m_guiPlatform;

	delete m_camera;

	delete m_scene;

	delete m_sceneController;

	delete m_world;

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
	//move camera
	if (m_camera != NULL)
	{
		m_camera->MoveLeftRight( m_cameraTransition.x * dt);
		m_camera->Translate(0, m_cameraTransition.y * dt, 0);
		m_camera->MoveBackForward(m_cameraTransition.z * dt);
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

void BaseApp::KeyReleased(HQKeyCodeType keyCode)
{
	switch (keyCode)
	{
	case HQKeyCode::ESCAPE:
		HQEngineApp::GetInstance()->Stop();
		break;
	}
}

void BaseApp::MousePressed(HQMouseKeyCodeType button, const HQPointi &point)
{
	switch (button)
	{
	case HQKeyCode::LBUTTON:
		m_leftMousePressed = true;
		m_prevMousePos = point;
		break;
	}
}
void BaseApp::MouseReleased(HQMouseKeyCodeType button, const HQPointi &point)
{
	switch (button)
	{
	case HQKeyCode::LBUTTON:
		m_leftMousePressed = false;
		break;
	}
}
void BaseApp::MouseMove(const HQPointi &point)
{
	if (m_leftMousePressed)
	{
		HQPointi dpos;
		dpos.x = point.x - m_prevMousePos.x;
		dpos.y = point.y - m_prevMousePos.y;

		//rotate camera
		m_camera->RotateY(dpos.x * 0.001f);
		m_camera->RotateVertical(dpos.y * -0.001f);

		m_prevMousePos = point; //store current position
	}//if (m_leftMousePressed)
}