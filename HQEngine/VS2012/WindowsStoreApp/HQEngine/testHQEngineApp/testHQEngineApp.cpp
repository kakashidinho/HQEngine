#include "pch.h"
#define HQ_CUSTOM_MALLOC
#define HQ_CUSTOM_NEW_OPERATOR
#include "HQEngineCustomHeap.h"
#include "HQEngineApp.h"
#include "HQEngine\winstore\HQWinStoreFileSystem.h"

HQEngineApp *app;

class KeyListener: public HQEngineKeyListener
{
public:
	 void KeyReleased(HQKeyCodeType keyCode)
	 {
		 switch (keyCode)
		 {
		 case HQKeyCode::ESCAPE:
			 app->Stop();
			 break;
		 }
	 }
};

class MouseListener: public HQEngineMouseListener
{
public:
	void MouseMove(const HQPointi &point)
	{
		char buffer[256];
		sprintf_s(buffer, 256, "mouse moved: %d, %d\n", point.x, point.y);
		OutputDebugStringA(buffer);
	}
};

class Renderer: public HQEngineRenderDelegate
{
public:
	 void Render(HQTime dt)
	 {
		 HQColor red = {1, 0, 0, 1};
		 app->GetRenderDevice()->SetClearColor(red);
		 app->GetRenderDevice()->Clear(HQ_TRUE, HQ_FALSE, HQ_FALSE);
	 }
};

int HQEngineMain (int argc, char **argv)
{
	int *p = HQ_NEW int;

	//test file system
	auto file = HQWinStoreFileSystem::OpenFileForRead("Assets/input.txt");
	char buffer[256];
	HQWinStoreFileSystem::fread(buffer, 256, 1, file);
	fclose(file);


	app = HQEngineApp::CreateInstance();
	app->InitWindow();

	app->SetRenderDelegate(Renderer());

	app->SetKeyListener(KeyListener());

	app->SetMouseListener(MouseListener());

	app->EnableMouseCursor(false);

	app->Run();

	app->Release();

	return 0;
}
