#ifndef BASE_APP_H
#define BASE_APP_H

#include "BaseCommon.h"

#include "../../HQEngine/Source/HQEngineApp.h"
#include "../../HQEngine/Source/HQCamera.h"

#include "../../ThirdParty-mod/MyGUI/include/MyGUI.h"
#include "../../ThirdParty-mod/MyGUI/include/MyGUI_HQEnginePlatform.h"

class BaseApp : public HQEngineRenderDelegate {
public:
	BaseApp(const char* rendererAPI, 
			HQLogStream *logStream,
			const char * additionalAPISettings = NULL);
	~BaseApp();

	void Run();

protected:
	virtual void Update(HQTime dt) = 0;
	virtual void RenderImpl(HQTime dt) = 0;///don't need to call BeginRender() & EndRender()

	char m_renderAPI_name[6];//"D3D9" or "GL"
	HQRenderAPI m_renderAPI_type;

	HQRenderDevice *m_pRDevice;
	HQCamera * m_camera;
	HQSceneNode* m_scene;//the whole scene

	//GUI 
	MyGUI::HQEnginePlatform* m_guiPlatform;
	MyGUI::Gui *m_myGUI;
	MyGUI::TextBox* m_fpsTextBox;

private:
	//implement HQEngineRenderDelegate
	virtual void Render(HQTime dt) __FINAL__;
};

#endif