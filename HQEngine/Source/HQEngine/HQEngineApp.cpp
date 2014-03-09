/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "../HQEngineApp.h"
#include "HQEngineWindow.h"
#include "HQEngineResManagerImpl.h"
#include "HQEngineEffectManagerImpl.h"
#include "HQDefaultFileManager.h"

#include <iostream>
#include <string.h>

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#if defined _DEBUG || defined DEBUG
#	ifdef _STATIC_CRT
#		pragma comment(lib, "../../VS/Output/StaticDebug static CRT/HQUtilMathD.lib")
#		pragma comment(lib , "../../VS/Output/Debug static CRT/HQEngineRendererD.lib")
#		pragma comment(lib , "../../VS/Output/Debug static CRT/HQUtilD.lib")
#	else
#		pragma comment(lib, "../../VS/Output/StaticDebug/HQUtilMathD.lib")
#		pragma comment(lib , "../../VS/Output/Debug/HQEngineRendererD.lib")
#		pragma comment(lib , "../../VS/Output/Debug/HQUtilD.lib")
#	endif
#elif defined _STATIC_CRT
#	pragma comment(lib, "../../VS/Output/StaticRelease static CRT/HQUtilMath.lib")
#	pragma comment(lib , "../../VS/Output/Release static CRT/HQEngineRenderer.lib")
#	pragma comment(lib , "../../VS/Output/Release static CRT/HQUtil.lib")
#else
#	pragma comment(lib, "../../VS/Output/StaticRelease/HQUtilMath.lib")
#	pragma comment(lib , "../../VS/Output/Release/HQEngineRenderer.lib")
#	pragma comment(lib , "../../VS/Output/Release/HQUtil.lib")
#endif

#define HQ_DEFAULT_RENDERER_TYPE "D3D9"
#elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#define HQ_DEFAULT_RENDERER_TYPE "D3D11"
#else
#define HQ_DEFAULT_RENDERER_TYPE "GL"
#endif

#ifdef ANDROID
#	define HQ_RENDER_DEVICE_LOST_POSSIBLE
#endif

#if	defined IOS || defined ANDROID
//for knowing if app is paused or not
extern int hq_internalAppIsPaused;
extern int hq_internalAppExit;
extern int HQAppInternalGetStatusFlag();
#endif

#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
extern void HQAppInternalBlockGameLoopIfNeeded();
#endif

#if defined IOS || defined ANDROID || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#	define USE_SEPERATE_GAME_THREAD
#endif

#ifdef USE_SEPERATE_GAME_THREAD
#include "HQEventSeparateThread.h"

extern HQEventQueue* hq_engine_eventQueue_internal;
#endif

enum HQStateFlag
{
	HQ_APP_START = 0x1,
	HQ_APP_RUNNING = 0x2,//application loop has been begun
	HQ_CURSOR_ENABLE = 0x4,
	HQ_RENDER_DEVICE_LOST = 0x8,
	HQ_RENDERER_D3D9 = 0x10,
	HQ_RENDERER_D3D11 = 0x20,
	HQ_RENDERER_GL = 0x40,
	HQ_APP_EVENTS_DISABLE = 0x80
};

class HQDefaultRenderDelegate : public HQEngineRenderDelegate
{
public:
	void Render(HQTime dt) {}//do nothing
};

static HQDefaultRenderDelegate defaultRenderDelegate;

static HQEngineWindowListener defaultWinListener;
static HQEngineKeyListener defaultKeyListener;
static HQEngineMouseListener defaultMouseListener;


static HQEngineMotionListener defaultMotionListener;
HQEngineAppListener hq_internalDefaultAppListener;
static HQEngineOrientationListener defaultOrientListener;


HQEngineApp *HQEngineApp::sm_instance = NULL;

/*----HQEngineApp::RenderDevice---------*/
HQReturnVal HQEngineApp::RenderDevice::Release() 
{ 
	return HQ_OK;
}


/*-----------HQEngineApp-----------*/
HQEngineApp::HQEngineApp(bool rendererDebugLayer)
:m_window(NULL) ,
m_renderer(rendererDebugLayer) , 
m_pRenderDevice(NULL) ,

m_renderDelegate(&defaultRenderDelegate),
m_waitRenderDelegate(NULL) ,

m_windowListener(&defaultWinListener),
m_waitWindowListener(NULL),

m_keyListener(&defaultKeyListener),
m_waitKeyListener(NULL),

m_mouseListener(&defaultMouseListener),
m_waitMouseListener(NULL),

m_motionListener(&defaultMotionListener),
m_waitMotionListener(NULL),

m_appListener (&hq_internalDefaultAppListener),
m_waitAppListener(NULL),

m_orientListener(&defaultOrientListener),
m_waitOrientListener(NULL),

m_flags(HQ_CURSOR_ENABLE)
{
	this->PlatformInit();

	//stop reading events when app is not running
#ifdef USE_SEPERATE_GAME_THREAD
	hq_engine_eventQueue_internal->Lock();
	hq_engine_eventQueue_internal->Enable(false);
	hq_engine_eventQueue_internal->Unlock();
#endif

	//create default file manager
	m_fileManagers.PushBack(HQ_NEW HQDefaultFileManager());
}

HQEngineApp::~HQEngineApp()
{
	this->DestroyWindow();
	this->PlatformRelease();

	HQLinkedList<HQFileManager*>::Iterator ite;
	m_fileManagers.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) {
		(*ite)->Release();
	}
}

HQEngineApp * HQEngineApp::CreateInstance(bool rendererDebugLayer )
{
	if (sm_instance == NULL)
	{
		sm_instance = HQ_NEW HQEngineApp(rendererDebugLayer);
	}
	return sm_instance;
}

HQReturnVal HQEngineApp::CreateInstanceAndWindow(
		const WindowInitParams* initParams,
		bool rendererDebugLayer, 
		HQEngineApp **ppAppOut
		)
{
	HQEngineApp * pApp = CreateInstance(rendererDebugLayer);
	if (ppAppOut != NULL)
		*ppAppOut = pApp;
	return pApp->InitWindow(initParams);
}

HQReturnVal HQEngineApp::Release()
{
	if (sm_instance != NULL)
	{
		if (sm_instance->m_flags & HQ_APP_RUNNING)
			return HQ_FAILED;
		HQ_DELETE (sm_instance);
		sm_instance = NULL;
	}
	return HQ_OK;
}

HQEngineApp::RenderDevice * HQEngineApp::GetRenderDevice() 
{
	//ugly casting
	return (RenderDevice *)m_pRenderDevice;
}

HQReturnVal HQEngineApp::InitWindow(const WindowInitParams* initParams)
{
	if (m_window == NULL)
	{
		const WindowInitParams nullParams = {NULL , NULL , NULL , NULL , NULL , false , NULL};
		if (initParams == NULL)
			initParams = &nullParams;
		/*---------create window------------*/
		try{
			m_window = HQ_NEW HQEngineWindow(initParams->windowTitle == NULL ? "untitled" : initParams->windowTitle , 
										initParams->rendererSettingFileDir,
										initParams->platformSpecific);
		}
		catch (std::bad_alloc e)
		{
			return HQ_FAILED;
		}

		/*-------create render device-------*/
		HQReturnVal re = this->CreateRenderDevice(initParams);
		if (HQFailed(re))
		{
			SafeDelete(m_window);
			return re;
		}
		/*---------clear window surface's color-----*/
		m_pRenderDevice->Clear(HQ_TRUE , HQ_FALSE , HQ_FALSE, HQ_TRUE);
		m_pRenderDevice->DisplayBackBuffer();

		/*-----------create resource and effect manager--------*/
		m_resManager = HQ_NEW HQEngineResManagerImpl(initParams->logStream, initParams->flushDebugLog);
		m_effectManager = HQ_NEW HQEngineEffectManagerImpl(initParams->logStream, initParams->flushDebugLog);
		return re;
	}
	return HQ_FAILED;
}



HQReturnVal HQEngineApp::CreateRenderDevice(const WindowInitParams* initParams)
{
	const char *l_rendererType = initParams->rendererType;
	if (l_rendererType == NULL)
			l_rendererType = HQ_DEFAULT_RENDERER_TYPE;
		
	HQReturnVal re = HQ_FAILED;

	/*-------switch renderer type---------*/
	if (!strcmp(l_rendererType , "GL"))
#if defined LINUX
#error need implement
#else 
	{
#	if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		m_flags |= HQ_RENDERER_GL;
		re = m_renderer.CreateGLDevice(initParams->flushDebugLog);
#	endif
	}
#endif
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#	if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	else if (!strcmp(l_rendererType , "D3D9"))
	{
		m_flags |= HQ_RENDERER_D3D9;
		re = m_renderer.CreateD3DDevice9(initParams->flushDebugLog);
	}
#	endif
	else if (!strcmp(l_rendererType , "D3D11"))
	{
		m_flags |= HQ_RENDERER_D3D11;
		re = m_renderer.CreateD3DDevice11(initParams->flushDebugLog);
	}
#endif
	
	/*-------init device------------*/
	if (re == HQ_OK)
	{
		re = m_renderer.GetDevice()->Init(
			m_window->GetRenderDeviceInitInput(), 
			initParams->rendererSettingFileDir,
			initParams->logStream,
			initParams->rendererAdditionalSetting);
		if (re == HQ_OK)
			m_pRenderDevice = m_renderer.GetDevice();
	}
	return re;
}

HQReturnVal HQEngineApp::DestroyWindow()
{
	//delete managers
	HQEngineResManagerImpl* resManagerImpl = static_cast<HQEngineResManagerImpl*> (m_resManager);
	HQEngineEffectManagerImpl* effectManagerImpl = static_cast<HQEngineEffectManagerImpl*> (m_effectManager);
	SafeDelete(resManagerImpl);
	SafeDelete(effectManagerImpl);

	//release render device
	if (m_pRenderDevice)
	{
		m_pRenderDevice->Release();
		m_pRenderDevice = NULL;
	}


	//delete window
	SafeDelete(m_window);
	m_renderer.Release();
	return HQ_OK;
}

HQReturnVal HQEngineApp::ShowWindow()
{
	if (m_window == NULL)
		return HQ_FAILED;
	return m_window->Show();
}

void HQEngineApp::SetRenderDelegate(HQEngineRenderDelegate &_delegate)
{
	m_waitRenderDelegate = &_delegate;
}

void HQEngineApp::SetWindowListener(HQEngineWindowListener &_listener)
{
	m_waitWindowListener = &_listener;
}

void HQEngineApp::SetKeyListener(HQEngineKeyListener &_listener)
{
	m_waitKeyListener = &_listener;
}

void HQEngineApp::SetMouseListener(HQEngineMouseListener &_listener)
{
	m_waitMouseListener = &_listener;
}

HQReturnVal HQEngineApp::EnableMouseCursor(bool enable)
{
	HQReturnVal re = this->PlatformEnableMouseCursor(enable);
	if (re == HQ_OK)
	{
		if (enable)
			m_flags |= HQ_CURSOR_ENABLE;
		else
			m_flags &= ~HQ_CURSOR_ENABLE;
	}
	return re;
}

bool HQEngineApp::IsMouseCursorEnabled()
{
	return (m_flags & HQ_CURSOR_ENABLE) != 0;
}

void HQEngineApp::SetMotionListener(HQEngineMotionListener &_listener)
{
	m_waitMotionListener = &_listener;
}

void HQEngineApp::SetAppListener(HQEngineAppListener &_listener)
{
	m_waitAppListener = &_listener;
}

void HQEngineApp::SetOrientationListener(HQEngineOrientationListener& _listener)
{
	m_waitOrientListener = &_listener;
}

void HQEngineApp::CheckForNewDelegates()
{
	if (m_waitRenderDelegate != NULL)
	{
		m_renderDelegate = m_waitRenderDelegate;
		m_waitRenderDelegate = NULL;
	}

	if (m_waitWindowListener != NULL)
	{
		m_windowListener = m_waitWindowListener;
		m_waitWindowListener = NULL;
	}
	if (m_waitKeyListener != NULL)
	{
		m_keyListener = m_waitKeyListener;
		m_waitKeyListener = NULL;
	}
	if (m_waitMouseListener != NULL)
	{
		m_mouseListener = m_waitMouseListener;
		m_waitMouseListener = NULL;
	}
	if (m_waitMotionListener != NULL)
	{
		m_motionListener = m_waitMotionListener;
		m_waitMotionListener = NULL;
	}
	if (m_waitAppListener != NULL)
	{
		m_appListener = m_waitAppListener;
		m_waitAppListener = NULL;
	}
	if (m_waitOrientListener != NULL)
	{
		m_orientListener = m_waitOrientListener;
		m_waitOrientListener = NULL;
	}
}

HQReturnVal HQEngineApp::Run(hq_uint32 fpsLimit)
{
	if ( m_flags & HQ_APP_RUNNING || m_window == NULL)
		return HQ_FAILED;

//start reading events when app is not running
#ifdef USE_SEPERATE_GAME_THREAD
	hq_engine_eventQueue_internal->Lock();
	hq_engine_eventQueue_internal->Enable(true);
	hq_engine_eventQueue_internal->Unlock();
#endif

	HQTimeCheckPoint lastPoint, currentPoint;
	HQTime dt;
	HQTime diff = 0.0f;
	HQTime desiredRenderTime = 0.0f;
	if (fpsLimit)
		desiredRenderTime = 1.0f / fpsLimit;

	m_flags |= HQ_APP_START | HQ_APP_RUNNING;


#if defined IOS || defined ANDROID
	int appStatus;
#endif

	do
	{
		/*----check for new delegates---------*/
		this->CheckForNewDelegates();
		/*---------check device "lost" state------*/
		if ((m_flags & HQ_RENDERER_D3D9) != 0
#ifdef HQ_RENDER_DEVICE_LOST_POSSIBLE
			|| (m_flags & HQ_RENDERER_GL) != 0
#endif
			)
		{
			if ((m_flags & HQ_RENDER_DEVICE_LOST) == 0)
			{
				if (m_pRenderDevice->IsDeviceLost())
				{
					m_flags |= HQ_RENDER_DEVICE_LOST;
					m_renderDelegate->OnLostDevice();
				}
			}
			else if (!m_pRenderDevice->IsDeviceLost())//recovered
			{
				m_flags &= ~HQ_RENDER_DEVICE_LOST;
				m_renderDelegate->OnResetDevice();
			}
		}
		/*-------check for event-------------*/
		//if has no event and app is not paused
		if (!this->EventHandle()
#if defined IOS || defined ANDROID
		&& ((appStatus = HQAppInternalGetStatusFlag()) != hq_internalAppIsPaused)
#endif
		)
		{
			bool skipFrame = false;

			/*--------get current time-----------*/
			HQTimer::GetCheckPoint(currentPoint);
			
			if (m_flags & HQ_APP_START)//first start
			{
				m_flags &= ~HQ_APP_START;

				//render first frame
				m_renderDelegate->Render(0.0f);
			}
			else
			{
				dt = HQTimer::GetElapsedTime(lastPoint , currentPoint);
				
				if (fpsLimit)
				{
					diff += dt;

					skipFrame = diff < desiredRenderTime;

					if (!skipFrame)//only render when elapsed time since last frame larger than or equal to desired rendering time
					{
						/*------game rendering---------------*/
						m_renderDelegate->Render(diff);
						
						diff = 0.0f;
					}
				}
				else
					/*------game rendering---------------*/
				{
					m_renderDelegate->Render(dt);

				}
			}//if (m_flags & HQ_APP_START)

			if (!skipFrame)
				m_pRenderDevice->DisplayBackBuffer();

			lastPoint = currentPoint;

			HQThread::TempPause();
		}//if (!this->EventHandle())
		
#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
		HQAppInternalBlockGameLoopIfNeeded();
#endif

	}while( m_flags & HQ_APP_RUNNING
#if defined IOS || defined ANDROID
		&& appStatus != hq_internalAppExit
#endif
		);

	return HQ_OK;
}

void HQEngineApp::Stop()
{
	m_flags &= ~HQ_APP_RUNNING;
//stop reading events when app is not running
#ifdef USE_SEPERATE_GAME_THREAD
	hq_engine_eventQueue_internal->Lock();
	hq_engine_eventQueue_internal->Enable(false);
	hq_engine_eventQueue_internal->Unlock();
#endif
}

HQDataReaderStream* HQEngineApp::OpenFileForRead(const char *file){
	HQLinkedList<HQFileManager*>::Iterator ite;
	m_fileManagers.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) {
		HQDataReaderStream* stream = (*ite)->OpenFileForRead(file);
		if (stream != NULL)
			return stream;
	}

	return NULL;
}

bool HQEngineApp::AddFileSearchPath(const char* path)
{
	HQDefaultFileManager *defaultFileManager = (HQDefaultFileManager*) m_fileManagers.GetFront();
	defaultFileManager->AddLastSearchPath(path);
	return true;
}

bool HQEngineApp::PushFileSearchPath(const char* path)
{
	HQDefaultFileManager *defaultFileManager = (HQDefaultFileManager*) m_fileManagers.GetFront();
	defaultFileManager->AddFirstSearchPath(path);
	return true;
}