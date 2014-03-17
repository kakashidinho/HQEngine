#ifndef _GAME_H_
#define _GAME_H_
#ifdef __APPLE__
#	include <TargetConditionals.h>
#	if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR //ios
#   else
#       define APPLE_FRAMEWORK
#   endif
#endif

#ifdef APPLE_FRAMEWORK
#include "HQEngine/HQEngineApp.h"
#include "HQEngine/HQMeshNode.h"
#else

#ifdef USE_DIRECTX_MATH
#pragma message ("use DirectX Math")
#include "../HQUtilDXMath.h"
#endif

#include "../HQEngineApp.h"
#include "../HQMeshNode.h"
#endif
#include <fstream>

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
//#	define DISABLE_AUDIO
#endif

#ifndef DISABLE_AUDIO
#ifdef APPLE_FRAMEWORK
#include "HQEngine/HQAudio.h"
#else
#include "../HQAudio.h"
#endif
#endif//#ifndef DISABLE_AUDIO

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include "MeshX.h"
#endif

#ifdef ANDROID
#include <android/log.h>
#endif

#define DEBUG_LOG_CAT 1

#ifdef ANDROID
#	define TRACE(...) __android_log_print(ANDROID_LOG_DEBUG, "test", __VA_ARGS__)
#else
#	define TRACE(...)
#endif

#ifndef WIN32
#	ifdef ANDROID
#		if DEBUG_LOG_CAT
#			define OutputDebugStringA(str) {__android_log_print(ANDROID_LOG_DEBUG, "test", str);}
#		else
#			define OutputDebugStringA(str) 
#		endif
#	else
#		define OutputDebugStringA(str) {printf(str);}
#	endif
#endif

class Game : public HQEngineRenderDelegate 
#if	defined WIN32 || defined HQ_MAC_PLATFORM || defined LINUX || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	, public HQEngineKeyListener , public HQEngineMouseListener , public HQEngineWindowListener
#endif
#if defined HQ_IPHONE_PLATFORM || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	, public HQEngineMotionListener, public HQEngineAppListener, public HQEngineOrientationListener
#endif
{
public:
	Game();
	~Game();
	void Render(HQTime dt);
	void OnLostDevice();
	void OnResetDevice();
	
#if	defined WIN32 || defined HQ_MAC_PLATFORM || defined LINUX || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	bool WindowClosing() {return false;}
	
	void KeyPressed(HQKeyCodeType keycode);
	void KeyReleased(HQKeyCodeType keycode);
	void MousePressed( HQMouseKeyCodeType button, const HQPointi &point) ;
	void MouseReleased( HQMouseKeyCodeType button, const HQPointi &point) ;
	void MouseMove( const HQPointi &point) ;
	void MouseWheel( hq_float32 delta, const HQPointi &point) ;
#endif
#if defined HQ_IPHONE_PLATFORM || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	
	void TouchBegan(const HQTouchEvent &event) ;
	void TouchMoved(const HQTouchEvent &event) ;
	void TouchEnded(const HQTouchEvent &event) ;
	void TouchCancelled(const HQTouchEvent &event);
	
	void ChangedToPortrait() ;
	void ChangedToPortraitUpsideDown() ;
	///right side is at top
	void ChangedToLandscapeLeft() ;
	///left side is at top
	void ChangedToLandscapeRight() ;

	void OnDestroy();
	void OnPause() ;
	void OnResume() ;
#if defined ANDROID || defined HQ_WIN_PHONE_PLATFORM
	bool BackButtonPressed();
#endif

private:
	
	HQMutex mutex;
	
	bool app_exit;
	bool app_pause;
	
#endif //#if	defined HQ_IPHONE_PLATFORM || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	
	
private:
	bool deviceLost;
	int API;
	hq_uint32 uniformBuffer[2];
	hq_uint32 vertexbuffer[4];
	hq_uint32 indexbuffer;
	hq_uint32 vertexLayout[3];
	hq_uint32 colorTexture;
	hq_uint32 music;
	HQA16ByteStorageArrayPtr< HQMatrix3x4 , 3> rotation;
	HQA16ByteMatrix4Ptr viewProj;
	HQLogStream* logFile;

	HQMeshNode * mesh;
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	MeshX *meshX;
#endif

	hquint32 m_offsetX, m_offsetY;

	HQEngineApp::RenderDevice *pDevice;

#ifndef DISABLE_AUDIO
	HQAudioDevice *audio;
#endif
};
#endif
