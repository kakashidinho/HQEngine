#include "Game.h"

#include <string>

//#define HQ_OPENGLES
#if !defined HQ_ANDROID_PLATFORM && !defined HQ_IPHONE_PLATFORM
#define GLES_EMULATION
#endif

//#define HQ_CUSTOM_MALLOC
#include "../HQEngineCustomHeap.h"

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	define TEXTURE_BUFFER 0
#endif

#define OGL_RENDERER  1
#define D3D9_RENDERER 0
#define D3D10_RENDERER 2
#define D3D11_RENDERER 3

#define USE_CORE_OPENGL_3_1 0


struct BUFFER2
{
	HQBaseMatrix3x4 rotation;
	HQBaseMatrix4 viewProj;
	HQBaseMatrix3x4 bones[36];
};


Game::Game() 
: m_offsetX(0), m_offsetY(0)
#if defined HQ_IPHONE_PLATFORM || defined HQ_ANDROID_PLATFORM || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
 , app_exit( false ), app_pause(false)
#endif

{
    
#ifdef __APPLE__
	/*resource directory*/
	NSString *resourcePath = [[NSBundle mainBundle] resourcePath];
    
	int len = [resourcePath lengthOfBytesUsingEncoding:NSASCIIStringEncoding] + 1 ;
	char * cDir = HQ_NEW char[len];
	[resourcePath getCString:cDir maxLength:len
					encoding:NSASCIIStringEncoding];
    

    //simulate the working directory of Windows
	chdir(cDir);
    
	delete[] cDir;
    
#endif

	HQEngineApp::CreateInstance(true); 
	
	//determine the API
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	FILE *apiSetting = fopen("../test/setting/API.txt" , "r");
	char api_str[20];
	fscanf(apiSetting , "%s" , api_str);
	if (!strcmp(api_str , "GL"))
#elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	char api_str[] = "D3D11";
#else
	char api_str[] = "GL";
#endif

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		API = D3D11_RENDERER;
#else
		API = OGL_RENDERER;
#ifdef WIN32
	else if (!strcmp(api_str , "D3D9"))
		API = D3D9_RENDERER;
	else if (!strcmp(api_str , "D3D11"))
		API = D3D11_RENDERER;

	fclose(apiSetting);
#endif
#endif//#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

	//create log stream
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	logFile = HQCreateDebugLogStream();
#elif defined WIN32 || defined HQ_LINUX_PLATFORM
	logFile = HQCreateFileLogStream("log.txt");
#elif defined HQ_ANDROID_PLATFORM
	logFile = HQCreateLogCatStream();
#else
	logFile = HQCreateConsoleLogStream();
#endif

	//init application
	HQEngineApp::WindowInitParams params = HQEngineApp::WindowInitParams::Default();
#if defined _DEBUG || defined DEBUG
	params.flushDebugLog = true;
#else
	params.flushDebugLog = false;
#endif
	params.logStream = logFile;
	params.platformSpecific = NULL;
#if USE_CORE_OPENGL_3_1
	params.rendererAdditionalSetting = "Core-GL3.1";
#else
	params.rendererAdditionalSetting = "";//"Core-GL4.2";
#endif
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	params.rendererSettingFileDir = "Assets/Setting.txt";
#else
	params.rendererSettingFileDir = "setting/Setting.txt";
#endif
	params.rendererType = api_str;
	params.windowTitle = "test";
#ifdef HQ_ANDROID_PLATFORM
	HQWIPPlatformSpecificType additionalSettings;
	additionalSettings.openGL_ApiLevel = 2;
	params.platformSpecific = &additionalSettings;
#elif defined HQ_IPHONE_PLATFORM
	HQWIPPlatformSpecificType additionalSettings;
	additionalSettings.landscapeMode = false;
	params.platformSpecific = &additionalSettings;
#endif
	
	HQEngineApp::GetInstance()->InitWindow(&params);


#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	//setup file search paths
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/script");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/shader");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/audio");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/meshes");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/shader");

	char apiResFile[256] = "resourcesD3D11-winrt.script";
#else
	//setup file search paths
	HQEngineApp::GetInstance()->AddFileSearchPath("script");
	HQEngineApp::GetInstance()->AddFileSearchPath("shader");
	HQEngineApp::GetInstance()->AddFileSearchPath("audio");
	HQEngineApp::GetInstance()->AddFileSearchPath("meshes");
	HQEngineApp::GetInstance()->AddFileSearchPath("shader");

	char apiResFile[256] = "resourcesD3D9.script";
	if (API == D3D11_RENDERER)
		sprintf(apiResFile, "resourcesD3D11.script");
	else if (API == OGL_RENDERER)
#if defined HQ_OPENGLES
		sprintf(apiResFile, "resourcesGLES.script");
#else
	if (HQEngineApp::GetInstance()->GetRenderDevice()->IsShaderSupport(HQ_VERTEX_SHADER, "4.2"))
		sprintf(apiResFile, "resourcesGL4.2.script");
	else if (HQEngineApp::GetInstance()->GetRenderDevice()->IsShaderSupport(HQ_VERTEX_SHADER, "1.4"))
		sprintf(apiResFile, "resourcesGL3.1.script");
	else
		sprintf(apiResFile, "resourcesGL.script");

#endif

#endif

	//load resources
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile(apiResFile);
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile("resourcesCommon.script");
	HQEngineApp::GetInstance()->GetEffectManager()->AddEffectsFromFile("effects.script");


	//load audio	
#ifndef DISABLE_AUDIO	
	//audio
	this->audio = HQCreateAudioDevice(HQ_DEFAULT_SPEED_OF_SOUND , logFile, true);
	this->audio->SetListenerVolume(1.0f);
	hquint32 audiobufferID;
#	define audioFile "battletoads-double-dragons-2.ogg"
	//this->audio->CreateAudioBufferFromFile(audioFile, &audiobufferID);

	//this->audio->CreateStreamAudioBufferFromFile(audioFile, 5, 65536,  &audiobufferID);
	this->audio->CreateStreamAudioBufferFromFile(audioFile, &audiobufferID);


	HQAudioSourceInfo info = {HQ_AUDIO_PCM, 16, 1};
	//HQAudioSourceInfo info = {HQ_AUDIO_PCM, 16, 2};
	this->audio->CreateSource(info, audiobufferID, &music);

#endif//#ifndef DISABLE_AUDIO
	//this->audio->GetSourceController(music)->SetVolume(0.5f);

	this->audio->GetSourceController(music)->Enable3DPositioning(HQ_FALSE);


	pDevice = HQEngineApp::GetInstance()->GetRenderDevice();	
	pDevice->EndRender();

	HQResolution* resolutionList;
	hquint32 numRes;
	pDevice->GetAllDisplayResolution(NULL , numRes);
	resolutionList = HQ_NEW HQResolution[numRes];
	pDevice->GetAllDisplayResolution(resolutionList , numRes);

	{
		char msgBuffer[256];
		for(hquint32 i = 0 ; i < numRes ; ++i)
		{
			sprintf(msgBuffer, "%u x %u\n" , resolutionList[i].width , resolutionList[i].height);
			OutputDebugStringA(msgBuffer);
		}
	}
	delete[] resolutionList;

	/*--------viewproj matrix-----------*/
	HQA16ByteMatrix4Ptr view , proj;
	HQMatrix4rLookAtLH(HQA16ByteVector4Ptr(0,0,-2) , HQA16ByteVector4Ptr(0,0,0) , HQA16ByteVector4Ptr(0,1,0) , view);
	HQMatrix4rPerspectiveProjLH(HQPiFamily::_PIOVER4 , (hq_float32)pDevice->GetWidth() / pDevice->GetHeight() , 1.0f , 1000.0f , proj , (API == OGL_RENDERER)? HQ_RA_OGL : HQ_RA_D3D);

	HQMatrix4Multiply(view, proj, this->viewProj);
	

	memset(this->uniformBuffer, 0, sizeof(this->uniformBuffer));

	if (API == OGL_RENDERER)
	{
		pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(BUFFER2) , NULL , true , &this->uniformBuffer[1]);
		pDevice->GetShaderManager()->SetUniformBuffer(11 , this->uniformBuffer[0]);
		pDevice->GetShaderManager()->SetUniformBuffer(10 , this->uniformBuffer[1]);
	}
	else if (API == D3D9_RENDERER || API == D3D11_RENDERER)
	{
		pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(BUFFER2) , NULL , true , &this->uniformBuffer[1]);
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 11 , this->uniformBuffer[0]);
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 10 , this->uniformBuffer[1]);
	}
	/*------------check capabilities----*/
	bool support = pDevice->IsBlendStateExSupported();
	support = pDevice->IsIndexDataTypeSupported(HQ_IDT_UINT);

	/*---------meshes------------*/

	mesh[0] = HQ_NEW HQMeshNode("bat", "bat.hqmesh", pDevice, "vs-mesh", logFile);
	mesh[1] = HQ_NEW HQMeshNode("tiny", "tiny.hqmesh", pDevice, "vs-mesh", logFile);


#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	meshX = HQ_NEW MeshX("../test/meshes/tiny.x");
#endif

	deviceLost = false;

#ifndef DISABLE_AUDIO
	//play music
	this->audio->GetSourceController(music)->Play(HQ_TRUE);
	//this->audio->GetSourceController(music)->Play(HQ_FALSE);
#endif
}

Game::~Game()
{
	delete mesh[0];
	delete mesh[1];
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	delete meshX;
#endif
#ifndef DISABLE_AUDIO
	this->audio->Release();
#endif
	HQEngineApp::Release();
	
	if (logFile)
		logFile->Close();
	

}

void Game::OnLostDevice()
{
	deviceLost = true;
	OutputDebugStringA("OnLostDevice\n");
}

void Game::OnResetDevice()
{
	deviceLost = false;
	OutputDebugStringA("OnResetDevice\n");
	if (!pDevice->IsRunning() && API != D3D9_RENDERER)
		return;

	mesh[0]->OnResetDevice();
	mesh[1]->OnResetDevice();
}

void Game::Render(HQTime dt)
{
#ifdef WIN32
	//Sleep(13);
#endif

#if defined HQ_IPHONE_PLATFORM || defined HQ_ANDROID_PLATFORM || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	mutex.Lock();
	if (app_exit)
	{
		HQEngineApp::GetInstance()->Stop();	
	}
		
	mutex.Unlock();

	bool pause;
	mutex.Lock();
	pause = app_pause;
	mutex.Unlock();
#else
	bool pause = false;
#endif
	
	if (!pDevice->IsRunning() || pause)
		return;
	
	
	if (deviceLost)
		return;
		

	this->audio->GetSourceController(music)->UpdateStream();

	static hq_float32 angle = 0.0f;
	angle += dt * HQPiFamily::PI / 18.0f;
	HQ_DECL_STACK_MATRIX3X4( scale) ;
	//HQMatrix3x4Scale(0.8f, 0.8f, 0.8f, &scale);
	HQMatrix3x4cRotateY(-angle , &rotation[0]);
	HQMatrix3x4cRotateX(-angle , &rotation[1]);
	HQMatrix3x4cRotateZ(-angle , &rotation[2]);

	//rotation[0] *= rotation[1];
	//rotation[0] *= rotation[2];

	HQMatrix3x4MultiMultiply(rotation, 2, rotation);
	rotation[0] *= scale;
	
	/*------sound position----------*/
	HQ_DECL_STACK_VECTOR4_CTOR_PARAMS(soundPosition, (0, 1, 1));
	HQVector4TransformCoord(&soundPosition, &rotation[0], &soundPosition);

#ifndef DISABLE_AUDIO
	audio->GetSourceController(music)->SetPosition(soundPosition);
#endif
	
	//update meshes
	mesh[0]->SetUniformScale(0.3f);
	mesh[0]->RotateY(dt * HQPiFamily::PI / 18.f);
	mesh[0]->AdvanceAnimationTime(dt);
	mesh[0]->Update(dt);

	mesh[1]->SetUniformScale(0.003f);
	mesh[1]->RotateY(dt * HQPiFamily::PI / 18.f);
	mesh[1]->AdvanceAnimationTime(dt);
	mesh[1]->Update(dt);

	//TRACE("here %s %d", __FILE__, __LINE__);

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	meshX->GetAnimationController()->AdvanceTime(dt, NULL);
	meshX->Update();
#endif

	//clear whole render target
	pDevice->SetClearStencilVal(0xff);
	pDevice->SetClearDepthVal(1.0f);
	pDevice->SetClearColorf(0.5f , 1.0f ,0.5f , 1.0f);
	pDevice->BeginRender(HQ_TRUE , HQ_TRUE , HQ_TRUE ,HQ_TRUE);

	hquint32 halfw = pDevice->GetWidth() / 2;
	hquint32 halfh = pDevice->GetHeight() / 2;

	//clear depth
	pDevice->SetClearDepthVal(0.5f);
	HQViewPort viewport2 = {0 + m_offsetX , halfh + m_offsetY , halfw / 2 , halfh / 2};
	pDevice->SetViewPort(viewport2);
	pDevice->Clear(HQ_FALSE, HQ_TRUE, HQ_FALSE, HQ_FALSE);

	//clear stencil
	pDevice->SetClearStencilVal(0x1);
	HQViewPort viewport3 = {halfw / 2  + m_offsetX, halfh * 3 / 2  + m_offsetY, halfw / 2, halfh / 2};
	pDevice->SetViewPort(viewport3);
	pDevice->Clear(HQ_FALSE, HQ_FALSE, HQ_TRUE, HQ_FALSE);

	//clear color
	HQViewPort viewport = {0  + m_offsetX, halfh  + m_offsetY , halfw , halfh};
	pDevice->SetViewPort(viewport);
	pDevice->SetClearColorf(1, 0, 0, 1);
	pDevice->Clear(HQ_TRUE, HQ_FALSE, HQ_FALSE, HQ_FALSE);
	
	pDevice->SetPrimitiveMode(HQ_PRI_TRIANGLES);
	//pDevice->GetStateManager()->SetFillMode(HQ_FILL_WIREFRAME);


	for (int i = 0; i < 2; ++i)
	{
		HQEngineApp::GetInstance()->GetEffectManager()->GetEffect("mesh-effect")->GetPass(i)->Apply();

		BUFFER2 * pTBuffer0 = NULL;
		this->uniformBuffer[1]->Map (&pTBuffer0);

		if(pTBuffer0)
		{
			const HQMatrix3x4 * boneMatrices = mesh[i]->GetBoneTransformMatrices();
			hquint32 numBones = mesh[i]->GetNumBones();

			memcpy(&pTBuffer0->rotation, &scale, sizeof(HQMatrix3x4));
			memcpy(&pTBuffer0->bones, boneMatrices, numBones * sizeof(HQMatrix3x4));
			memcpy(&pTBuffer0->viewProj, viewProj, sizeof(HQMatrix4));
		}

		this->uniformBuffer[1]->Unmap();

		mesh[i]->DrawInOneCall();
	}//for (int i = 0; i < 2; ++i)

	pDevice->EndRender();



	pDevice->GetStateManager()->ActiveDepthStencilState(0);

	//HQTimer::Sleep(0.5f);
#if 0
	char fpsMsg[256];
	sprintf(fpsMsg, "fps = %.3f\n", HQEngineApp::GetInstance()->GetFPS());
	OutputDebugStringA(fpsMsg);
#endif
}

#if	defined WIN32 || defined HQ_MAC_PLATFORM || defined HQ_LINUX_PLATFORM || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

void Game::KeyPressed(HQKeyCodeType keycode)
{
	static bool playmusic = true; 
	static bool pauseMusic = false;
	switch(keycode)
	{
	case HQKeyCode::M:
		playmusic = !playmusic;
#ifndef DISABLE_AUDIO
		if (playmusic)
			audio->GetSourceController(music)->Play();
		else
			audio->GetSourceController(music)->Stop();
#endif
		break;
	case HQKeyCode::P:
		pauseMusic = !pauseMusic;
#ifndef DISABLE_AUDIO
		if (!pauseMusic)
			audio->GetSourceController(music)->Play();
		else
			audio->GetSourceController(music)->Pause();
#endif
		break;
	case HQKeyCode::L:
#ifndef DISABLE_AUDIO
		audio->GetSourceController(music)->Play(HQ_FALSE);
#endif
		break;
	case HQKeyCode::LSHIFT:
		OutputDebugStringA("lshift\n");
		break;
	case HQKeyCode::RSHIFT:
		OutputDebugStringA("rshift\n");
		break;
	case HQKeyCode::LCONTROL:
		OutputDebugStringA("lctrl\n");
		break;
	case HQKeyCode::RCONTROL:
		OutputDebugStringA("rctrl\n");
		break;
#ifdef WIN32
	case HQKeyCode::LALT:
		OutputDebugStringA("lalt\n");
		break;
	case HQKeyCode::RALT:
		OutputDebugStringA("ralt\n");
		break;
#endif
#if !(defined HQ_OPENGLES || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	case HQKeyCode::SPACE:

		pDevice->SetDisplayMode(800, 600 , !pDevice->IsWindowed());
		OnResetDevice();
		break;
	case HQKeyCode::Q:
		pDevice->SetDisplayMode(1280, 768 , !pDevice->IsWindowed());
		OnResetDevice();
#endif
		break;
	case HQKeyCode::TAB:
		HQEngineApp::GetInstance()->EnableMouseCursor(!HQEngineApp::GetInstance()->IsMouseCursorEnabled());
		break;
	case HQKeyCode::DIVIDE:
		OutputDebugStringA("/\n");
		break;
	case HQKeyCode::ESCAPE:
		HQEngineApp::GetInstance()->Stop();
		break;
	}
}

void Game::KeyReleased(HQKeyCodeType keycode)
{
	switch(keycode)
	{
		case HQKeyCode::NUM1:
			break;
		case HQKeyCode::NUM2:
			break;
		case HQKeyCode::NUM3:
			break;
		case HQKeyCode::NUM4:
			break;
		case HQKeyCode::LSHIFT:
			OutputDebugStringA("lshift released\n");
			break;
		case HQKeyCode::RSHIFT:
			OutputDebugStringA("rshift released\n");
			break;
		case HQKeyCode::LCONTROL:
			OutputDebugStringA("lctrl released\n");
			break;
		case HQKeyCode::RCONTROL:
			OutputDebugStringA("rctrl released\n");
			break;
#ifdef WIN32
		case HQKeyCode::LALT:
			OutputDebugStringA("lalt released\n");
			break;
		case HQKeyCode::RALT:
			OutputDebugStringA("ralt released\n");
			break;
#endif
	}
}

void Game::MousePressed( HQMouseKeyCodeType button, const HQPointi &point) 
{
	switch(button)
	{
	case HQKeyCode::LBUTTON:
		OutputDebugStringA("lmouse\n");
		break;
	case HQKeyCode::RBUTTON:
		OutputDebugStringA("rmouse\n");
		break;
	case HQKeyCode::MBUTTON:
		OutputDebugStringA("mmouse\n");
		break;
	}
}
void Game::MouseReleased( HQMouseKeyCodeType button, const HQPointi &point) 
{
	switch(button)
	{
		case HQKeyCode::LBUTTON:
			OutputDebugStringA("lmouse released\n");
			break;
		case HQKeyCode::RBUTTON:
			OutputDebugStringA("rmouse released\n");
			break;
		case HQKeyCode::MBUTTON:
			OutputDebugStringA("mmouse released\n");
			break;
	}
}
void Game::MouseMove( const HQPointi &point) 
{
#if 1
	char info[256];
	sprintf(info , "%d , %d\n", point.x, point.y);
	OutputDebugStringA(info);
#endif
}
void Game::MouseWheel( hq_float32 delta, const HQPointi &point)
{
	char info[10];
	int len;
#ifdef WIN32
	len = _snprintf(info , 10 , "%.5f\n", delta);
#else
	len = snprintf(info , 10 , "%.5f\n", delta);
#endif
	if (len >= 10 || len == -1)
	{
		info[8] = '\n';
		info[9] = '\0';
	}
	OutputDebugStringA(info);
}
#endif//defined WIN32 || defined HQ_MAC_PLATFORM || defined HQ_LINUX_PLATFORM || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#if defined HQ_IPHONE_PLATFORM || defined HQ_ANDROID_PLATFORM || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

void Game::TouchBegan(const HQTouchEvent &event) 
{
	static int index = 0;
	index = (index + 1) % 4;
	switch (index)
	{
		case 0:
			break;
		case 1:
			break;
		case 2:
			break;
		case 3:
			break;
	}
	
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchBegan : numTouches:%u first touch: id=%d, position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y);
	OutputDebugStringA(info);
}

void Game::TouchMoved(const HQTouchEvent &event) 
{
	
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	const HQPointf &prev_point = event.GetPrevPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchMoved : numTouches:%u first touch: id=%d, position={%.4f , %.4f}, prev_position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y, prev_point.x, prev_point.y);
	OutputDebugStringA(info);
}
void Game::TouchEnded(const HQTouchEvent &event) 
{
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	const HQPointf &prev_point = event.GetPrevPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchEnded : numTouches:%u first touch: id=%d, position={%.4f , %.4f}, prev_position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y, prev_point.x, prev_point.y);
	OutputDebugStringA(info);
}
void Game::TouchCancelled(const HQTouchEvent &event)
{
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchCancelled : numTouches:%u first touch: id=%d, position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y);
	OutputDebugStringA(info);
}

void Game::OnDestroy()
{
	HQMutex::ScopeLock sl(mutex);
	app_exit = true;
	
	OutputDebugStringA("app will be destroyed!\n");
}

void Game::OnPause() 
{
	OutputDebugStringA("app is paused!\n");

	audio->GetSourceController(music)->Pause();
}

void Game::OnResume()
{
	OutputDebugStringA("app is resumed!\n");

	audio->GetSourceController(music)->Play();
}

void Game::ChangedToPortrait() 
{
	m_offsetX = m_offsetY = 0;
	OutputDebugStringA("portrait mode\n");
}
void Game::ChangedToPortraitUpsideDown() 
{
	m_offsetX = pDevice->GetWidth()/2;
	m_offsetY = -(int)pDevice->GetHeight()/2;
	OutputDebugStringA("portrait upside down mode\n");
}
///right side is at top
void Game::ChangedToLandscapeLeft() 
{
	m_offsetX = 0;
	m_offsetY = -(int)pDevice->GetHeight()/2;
	OutputDebugStringA("landscape left mode\n");
}
///left side is at top
void Game::ChangedToLandscapeRight()
{
	m_offsetX = pDevice->GetWidth()/2;
	m_offsetY = 0;
	OutputDebugStringA("landscape right mode\n");
}

#if defined HQ_ANDROID_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
bool Game::BackButtonPressed()
{
	OutputDebugStringA("back key is pressed!\n");
	app_exit = true;

	return false;
}
#endif

#endif //#if	defined WIN32 || defined HQ_MAC_PLATFORM || defined HQ_LINUX_PLATFORM
