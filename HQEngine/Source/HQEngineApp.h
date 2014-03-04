/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_APP_H
#define HQ_ENGINE_APP_H
#include "HQEngineCommon.h"
#include "HQRenderer.h"
#include "HQEngineAppDelegate.h"
#include "HQDataStream.h"


/*-----------win32---------------*/
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

typedef HWND HQNativeWindow;

typedef struct HQWIPWin32SpecificType
{
	HICON icon;//icon for window 
	HICON sicon;//small icon for window
} HQWIPPlatformSpecificType;

/*---------win rt----------------*/
#elif defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM

typedef struct HQWIPWinStoreSpecificType
{
#ifdef HQ_WIN_STORE_PLATFORM
	Platform::Agile<Windows::UI::Core::CoreCursor> cursor;
#endif
} HQWIPPlatformSpecificType;

/*-------Mac OSX-----------------*/
#elif defined APPLE
typedef struct HQWIPMacSpecificType{
	NSImage *icon;//application icon
} HQWIPPlatformSpecificType;

/*------IOS----------------------*/
#elif defined IOS
typedef struct HQWIPIOSSpecificType{
	bool landscapeMode;//is landscape mode
} HQWIPPlatformSpecificType;

/*--------Android----------------*/
#elif defined ANDROID

typedef struct HQWIPAndroidSpecificType{
	int openGL_ApiLevel;//OpenGL ES version 1 or 2.other values are equal to 1
} HQWIPPlatformSpecificType;


#else
#	error need implement
#endif

class HQEngineWindow;


///
///HQEngine application
///not thread safe
///
#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4251 )
#endif
class HQENGINE_API HQEngineApp
{
public:

	class RenderDevice : public HQRenderDevice
	{
	private:
		///prevent release outside application class
		HQReturnVal Release() ;
	};
	

	struct WindowInitParams{
		inline static WindowInitParams Default();//return default params 
		inline static WindowInitParams Construct(const char *windowTitle = NULL,//if NULL => "untitled"
												const char *rendererType = NULL,//"D3D9", "D3D11" , "GL" .if NULL => "D3D9" for win32 and "GL" for the others
												const char *rendererSettingFileDir = NULL,//can be NULL => default renderer setting
												const char *rendererAdditionalSetting = NULL,//see HQRenderDevice::Init().can be NULL
												HQLogStream* logStream = NULL,//can be NULL
												bool flushDebugLog = false,
												HQWIPPlatformSpecificType* platformSpecific = NULL);

		HQLogStream* logStream;//can be NULL
		const char *windowTitle;//if NULL => "untitled"
		const char *rendererType;//"D3D9", "D3D11" , "GL" .if NULL => "D3D9" for win32, "D3D11" for winRT and "GL" for the others
		const char *rendererSettingFileDir;//can be NULL => default renderer setting
		const char *rendererAdditionalSetting;//see HQRenderDevice::Init().can be NULL
		bool flushDebugLog;
		HQWIPPlatformSpecificType* platformSpecific;//it can be NULL. On Android : If NULL => Android OpenGL ES 2.0
	};
	///
	///create application instance. 
	///if it's already created and not destroyed,return already existing instance. 
	///{rendererDebugLayer} is ignored in release build
	///
	static HQEngineApp * CreateInstance(bool rendererDebugLayer = false);
	///
	///create application instance. 
	///if it's already created and not destroyed,return already existing instance. 
	///{initParams} = NULL equals to a parameter with all NULL members and
	///				 {flushDebugLog} member = false. 
	///{rendererDebugLayer} is ignored in release build
	///
	static HQReturnVal CreateInstanceAndWindow(
		const WindowInitParams* initParams,
		bool rendererDebugLayer, 
		HQEngineApp **ppAppOut
		);
	///get instance of application
	static HQEngineApp *GetInstance() {return sm_instance;}
	///
	///release application instance (including its window, etc). 
	///only usable when application loop is stopped or hasn't been begun. 
	///
	static HQReturnVal Release();

	///Init window and its render device
	///fail if window already init and hasn't been destroyed
	///{initParams} = NULL equals to a parameter with all NULL members and
	///				 {flushDebugLog} member = false
	///
	HQReturnVal InitWindow(const WindowInitParams* initParams = NULL);
	HQReturnVal DestroyWindow();
	HQReturnVal ShowWindow();

	RenderDevice * GetRenderDevice() ;
	///
	///new delegate will be used after current frame is rendered
	///
	void SetRenderDelegate(HQEngineRenderDelegate &_delegate) ;
	
	///
	///new delegate will be used after current frame is rendered
	///
	void SetWindowListener(HQEngineWindowListener &_listener) ;
	HQEngineWindowListener *GetWindowListener() {return m_windowListener;}
	///
	///new delegate will be used after current frame is rendered
	///
	void SetKeyListener(HQEngineKeyListener &_listener) ;
	HQEngineKeyListener *GetKeyListener() {return m_keyListener;}

	///
	///new delegate will be used after current frame is rendered
	///
	void SetMouseListener(HQEngineMouseListener &_listener) ;
	HQEngineMouseListener *GetMouseListener() {return m_mouseListener;}

	///enable/disable mouse cursor. Can only call this method after window is created
	HQReturnVal EnableMouseCursor(bool enable);
	///is mouse cursor enabled?
	bool IsMouseCursorEnabled();
	
	void SetMotionListener(HQEngineMotionListener &_listener);
	HQEngineMotionListener *GetMotionListener() {return m_motionListener;}

	void SetOrientationListener(HQEngineOrientationListener& _listener);
	HQEngineOrientationListener *GetOrientationListener() {return m_orientListener;}
	
	
	void SetAppListener(HQEngineAppListener &_listener);
	HQEngineAppListener *GetAppListener() {return m_appListener;}

	///
	///begin application loop. 
	///fail if application loop has already been begun or there's no created window
	///{fpsLimit} = 0 => no limit
	///
	HQReturnVal Run(hq_uint32 fpsLimit = 0);
	///
	///stop application loop. 
	///only usable when application loop has begun
	///
	void Stop();

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	HQNativeWindow GetNativeWindow();///Don't remove window procedure on Win32 or App delegates will not work. Internal use only
#endif

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	///
	///these methods will add current working directory path to the start of any file path in any file operation withing engine
	///
	static const wchar_t * GetCurrentDir();
	static void SetCurrentDir(const wchar_t *dir);
	static void SetCurrentDir(const char *dir);
#endif

	HQDataReaderStream* OpenFileStream(const char *file);///open a file

private:
	static HQEngineApp *sm_instance;

	HQEngineApp(bool rendererDebugLayer);
	~HQEngineApp();
	
	HQReturnVal CreateRenderDevice(const WindowInitParams* initParams);
	///check for new delegates
	void CheckForNewDelegates();
	///event handling procedure for application's  loop.return false if has no event
	bool EventHandle();

	///platform specific
	HQReturnVal PlatformEnableMouseCursor(bool enable);

	///platform specific
	void PlatformInit();
	///platform specific
	void PlatformRelease();
	
	HQRenderer m_renderer;//render device's container
	HQRenderDevice *m_pRenderDevice;//render device
	
	HQEngineWindow * m_window;//window

	HQEngineRenderDelegate *m_renderDelegate;//rendering delegate
	HQEngineRenderDelegate *m_waitRenderDelegate;

	HQEngineWindowListener *m_windowListener;
	HQEngineWindowListener *m_waitWindowListener;

	HQEngineKeyListener *m_keyListener;
	HQEngineKeyListener *m_waitKeyListener;

	HQEngineMouseListener *m_mouseListener;
	HQEngineMouseListener *m_waitMouseListener;

	HQEngineMotionListener *m_motionListener;
	HQEngineMotionListener *m_waitMotionListener;
	
	HQEngineAppListener *m_appListener;
	HQEngineAppListener *m_waitAppListener;

	HQEngineOrientationListener *m_orientListener;
	HQEngineOrientationListener *m_waitOrientListener;

	hq_uint32 m_flags;
};

#ifdef WIN32
#	pragma warning( pop )
#endif

/*------HQEngineApp::WindowInitParams-----*/
inline HQEngineApp::WindowInitParams HQEngineApp::WindowInitParams::Default()
{
	HQEngineApp::WindowInitParams _default;
	_default.logStream = NULL;
	_default.windowTitle = NULL;
	_default.rendererSettingFileDir = NULL;
	_default.rendererAdditionalSetting = NULL;
	_default.rendererType = NULL;
	_default.flushDebugLog = false;
	_default.platformSpecific = NULL;
	return _default;
}

inline HQEngineApp::WindowInitParams HQEngineApp::WindowInitParams::Construct(const char *windowTitle ,//if NULL => "untitled"
										const char *rendererType ,//"D3D9", "D3D11" , "GL" .if NULL => "D3D9" for win32 and "GL" for the others
										const char *rendererSettingFileDir ,//can be NULL => default renderer setting
										const char *rendererAdditionalSetting ,//see HQRenderDevice::Init().can be NULL
										HQLogStream* logStream,//can be NULL
										bool flushDebugLog ,
										HQWIPPlatformSpecificType* platformSpecific)
{
	HQEngineApp::WindowInitParams params;
	params.logStream = logStream;
	params.windowTitle = windowTitle;
	params.rendererSettingFileDir = rendererSettingFileDir;
	params.rendererAdditionalSetting = rendererAdditionalSetting;
	params.rendererType = rendererType;
	params.flushDebugLog = flushDebugLog;
	params.platformSpecific = platformSpecific;
	return params;
}

#endif
