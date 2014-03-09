#include "Game.h"

#include <string>

//#define GLES
#if !defined HQ_ANDROID_PLATFORM && !defined HQ_IPHONE_PLATFORM
#define GLES_EMULATION
#endif

#define HQ_CUSTOM_MALLOC
#include "../HQEngineCustomHeap.h"

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	define TEXTURE_BUFFER 0
#endif

#define OGL_RENDERER  1
#define D3D9_RENDERER 0
#define D3D10_RENDERER 2
#define D3D11_RENDERER 3


struct BUFFER2
{
	HQBaseMatrix3x4 rotation;
	HQBaseMatrix4 viewProj;
	HQBaseMatrix3x4 bones[36];
};


#ifdef LINUX
#error need implement
#else
Game::Game() 
: m_offsetX(0), m_offsetY(0)
#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
 , app_exit( false ), app_pause(false)
#endif

{
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
#elif defined WIN32
	logFile = HQCreateFileLogStream("log.txt");
#elif defined ANDROID
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
	params.rendererAdditionalSetting = "GLSL-only";
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	params.rendererSettingFileDir = "Assets/Setting.txt";
#else
	params.rendererSettingFileDir = "setting/Setting.txt";
#endif
	params.rendererType = api_str;
	params.windowTitle = "test";
#ifdef ANDROID
	HQWIPPlatformSpecificType additionalSettings;
	additionalSettings.openGL_ApiLevel = 2;
	params.platformSpecific = &additionalSettings;
#elif defined IOS
	HQWIPPlatformSpecificType additionalSettings;
	additionalSettings.landscapeMode = false;
	params.platformSpecific = &additionalSettings;
#endif
	
	HQEngineApp::GetInstance()->InitWindow(&params);


#ifdef __APPLE__
	/*resource directory*/
	NSString *resourcePath = [[NSBundle mainBundle] resourcePath];

	int len = [resourcePath lengthOfBytesUsingEncoding:NSASCIIStringEncoding] + 1 ;
	char * cDir = HQ_NEW char[len + strlen("/test")];
	[resourcePath getCString:cDir maxLength:len
					encoding:NSASCIIStringEncoding];

	strcat(cDir, "/test");

	chdir(cDir);

	delete[] cDir;

#endif

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	//setup file search paths
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/script");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/shader");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/audio");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/meshes");
	HQEngineApp::GetInstance()->AddFileSearchPath("Assets/test/shader");

	char apiResFile[256] = "resourcesD3D11-winrt.xml";
#else
	//setup file search paths
	HQEngineApp::GetInstance()->AddFileSearchPath("script");
	HQEngineApp::GetInstance()->AddFileSearchPath("shader");
	HQEngineApp::GetInstance()->AddFileSearchPath("audio");
	HQEngineApp::GetInstance()->AddFileSearchPath("meshes");
	HQEngineApp::GetInstance()->AddFileSearchPath("shader");

	char apiResFile[256] = "resourcesD3D9.xml";
	if (API == D3D11_RENDERER)
		sprintf(apiResFile, "resourcesD3D11.xml");
	else if (API == OGL_RENDERER)
		sprintf(apiResFile, "resourcesGL.xml");

#endif

	//load resources
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromXML(apiResFile);
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromXML("resourcesCommon.xml");
	HQEngineApp::GetInstance()->GetEffectManager()->AddEffectsFromXML("effects.xml");


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

#if !(defined GLES || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM))
	HQResolution* resolutionList;
	hquint32 numRes;
	pDevice->GetAllDisplayResolution(NULL , numRes);
	resolutionList = HQ_NEW HQResolution[numRes];
	pDevice->GetAllDisplayResolution(resolutionList , numRes);

	for(hquint32 i = 0 ; i < numRes ; ++i)
	{
	    printf("%u x %u\n" , resolutionList[i].width , resolutionList[i].height);
	}
	delete[] resolutionList;
#endif
	/*--------viewproj matrix-----------*/
	HQA16ByteMatrix4Ptr view , proj;
	HQMatrix4rLookAtLH(HQA16ByteVector4Ptr(0,0,-2) , HQA16ByteVector4Ptr(0,0,0) , HQA16ByteVector4Ptr(0,1,0) , view);
	HQMatrix4rPerspectiveProjLH(HQPiFamily::_PIOVER4 , (hq_float32)pDevice->GetWidth() / pDevice->GetHeight() , 1.0f , 1000.0f , proj , (API == OGL_RENDERER)? HQ_RA_OGL : HQ_RA_D3D);

	HQMatrix4Multiply(view, proj, this->viewProj);
	

#ifndef GLES

	if (API == OGL_RENDERER)
	{
		pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(HQColor) , NULL , true , &this->uniformBuffer[1]);
		pDevice->GetShaderManager()->SetUniformBuffer(11 , this->uniformBuffer[0]);
		pDevice->GetShaderManager()->SetUniformBuffer(10 , this->uniformBuffer[1]);
	}
	else if (API == D3D10_RENDERER || API == D3D11_RENDERER)
	{
		pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(BUFFER2) , NULL , true , &this->uniformBuffer[1]);
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 11 , this->uniformBuffer[0]);
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 10 , this->uniformBuffer[1]);
	}
#endif
	/*------------check capabilities----*/
	bool support = pDevice->IsBlendStateExSupported();
	support = pDevice->IsIndexDataTypeSupported(HQ_IDT_UINT);

	/*---------mesh------------*/

	mesh = HQ_NEW HQMeshNode("tiny", "bat.hqmesh", pDevice, "vs-mesh", logFile);


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
	delete mesh;
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	delete meshX;
#endif
#ifndef DISABLE_AUDIO
	this->audio->Release();
#endif
	HQEngineApp::Release();
#ifdef LINUX
#else
	if (logFile)
		logFile->Close();
	
#endif
}

void Game::OnLostDevice()
{
	deviceLost = true;
	OutputDebugStringA("OnLostDevice");
}

void Game::OnResetDevice()
{
	deviceLost = false;
	OutputDebugStringA("OnResetDevice");
	if (!pDevice->IsRunning() && API != D3D9_RENDERER)
		return;

	mesh->OnResetDevice();
}

void Game::Render(HQTime dt)
{
#ifdef WIN32
	//Sleep(13);
#endif

#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
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
	angle += HQPiFamily::PI / 180.0f;
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
	
	//update mesh
	mesh->SetUniformScale(0.3f);
	mesh->RotateY(HQPiFamily::PI / 180.f);
	mesh->AdvanceAnimationTime(dt);
	mesh->Update(dt);
	const HQMatrix3x4 * boneMatrices = mesh->GetBoneTransformMatrices();

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

	HQEngineApp::GetInstance()->GetEffectManager()->GetEffect("mesh-effect")->GetPassByName("pass-0")->Apply();


	if (API == OGL_RENDERER)
	{
		
#ifndef GLES		
		
		pDevice->GetShaderManager()->SetUniformMatrix("rotation" , &scale , 1);

		pDevice->GetShaderManager()->SetUniformMatrix("boneMatrices" , mesh->GetBoneTransformMatrices() , mesh->GetNumBones());
#else
		pDevice->GetShaderManager()->SetUniform4Float("rotation" , scale , 3);

		pDevice->GetShaderManager()->SetUniform4Float("boneMatrices" , (float*)mesh->GetBoneTransformMatrices() , mesh->GetNumBones() * 3);
#endif
		pDevice->GetShaderManager()->SetUniformMatrix("viewProj" , *viewProj);


	}
	
#ifdef WIN32
	else if (API == D3D9_RENDERER)
	{
		pDevice->GetShaderManager()->SetUniformMatrix("viewProj" , *viewProj);
		pDevice->GetShaderManager()->SetUniformMatrix("rotation" , &scale , 1);
		pDevice->GetShaderManager()->SetUniformMatrix("boneMatrices" , mesh->GetBoneTransformMatrices() , mesh->GetNumBones());
		
	}

	else if (API == D3D10_RENDERER || API == D3D11_RENDERER)
	{
		BUFFER2 * pTBuffer0 = NULL;
#if TEXTURE_BUFFER
#		pragma message ("use TEXTURE_BUFFER")
		pDevice->GetTextureManager()->MapTextureBuffer(textureBuffer , (void**)&pTBuffer0);
#else
		pDevice->GetShaderManager()->MapUniformBuffer(this->uniformBuffer[1] , (void**)&pTBuffer0);
#endif
		if(pTBuffer0)
		{
			const HQMatrix3x4 *boneMatrices = mesh->GetBoneTransformMatrices();
			hquint32 numBones = mesh->GetNumBones();

			memcpy(&pTBuffer0->rotation , &scale , sizeof(HQMatrix3x4));
			memcpy(&pTBuffer0->bones , boneMatrices , numBones * sizeof(HQMatrix3x4));
			memcpy(&pTBuffer0->viewProj, viewProj , sizeof(HQMatrix4));
		}
#if TEXTURE_BUFFER
		pDevice->GetTextureManager()->UnmapTextureBuffer(textureBuffer );
#else
		pDevice->GetShaderManager()->UnmapUniformBuffer(this->uniformBuffer[1] );
#endif

	}
	
#endif //#ifdef WIN32

	mesh->DrawInOneCall();

	pDevice->EndRender();



	pDevice->GetStateManager()->ActiveDepthStencilState(0);

	//HQTimer::Sleep(0.5f);

}

#if	defined WIN32 || defined APPLE || defined LINUX || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

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
#if !(defined GLES || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	case HQKeyCode::SPACE:

		pDevice->SetDisplayMode(800, 600 , !pDevice->IsWindowed());
		OnResetDevice();
		break;
	case HQKeyCode::Q:
		pDevice->SetDisplayMode(1280, 800 , !pDevice->IsWindowed());
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
#if 0
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
#endif//defined WIN32 || defined APPLE || defined LINUX || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

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

#if defined ANDROID || defined HQ_WIN_PHONE_PLATFORM
bool Game::BackButtonPressed()
{
	OutputDebugStringA("back key is pressed!\n");
	app_exit = true;

	return false;
}
#endif

#endif //#if	defined WIN32 || defined APPLE || defined LINUX
