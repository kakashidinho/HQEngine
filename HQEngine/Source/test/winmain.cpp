#ifdef APPLE
#include "HQEngine/HQPlatformDef.h"
#else
#include "../HQPlatformDef.h"
#endif

#include "../HQConditionVariable.h"

#include "../HQAtomic.h"

#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#if 0
#define HQ_CUSTOM_MALLOC
#include "../HQEngineCustomHeap.h"
#endif

//#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
//#define new DEBUG_NEW

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include "../ImagesLoader/Bitmap.h"
#include <windows.h>
#include <Commctrl.h>
#include <string.h>
#pragma comment(lib,"Comctl32.lib")
#ifdef _DEBUG
#	ifdef _STATIC_CRT
#		pragma comment(lib,"../../VS/Output/Debug static CRT/HQAudioD.lib")
#		pragma comment(lib,"../../VS/Output/Debug static CRT/HQSceneManagement.lib")
#		pragma comment(lib,"../../VS/Output/Debug static CRT/ImagesLoaderD.lib")
#		pragma comment(lib,"../../VS/Output/Debug static CRT/HQEngineD.lib")
#		pragma comment(lib,"../../VS/Output/Debug static CRT/HQUtilD.lib")
#		pragma comment(lib,"../../VS/Output/Debug static CRT/HQUtilMathD.lib")
#	else
#		pragma comment(lib,"../../VS/Output/Debug/HQAudioD.lib")
#		pragma comment(lib,"../../VS/Output/Debug/HQSceneManagement.lib")
#		pragma comment(lib,"../../VS/Output/Debug/ImagesLoaderD.lib")
#		pragma comment(lib,"../../VS/Output/Debug/HQEngineD.lib")
#		pragma comment(lib,"../../VS/Output/Debug/HQUtilD.lib")
#		pragma comment(lib,"../../VS/Output/Debug/HQUtilMathD.lib")
#	endif
#elif defined (_NOOPTIMIZE)
#	pragma comment(lib,"../../VS/Output/Release/ImagesLoader.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQAudio.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQEngine.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQUtil.lib")
#	pragma comment(lib,"../../VS/Output/release noptimize/HQUtilMath.lib")
#elif defined (_STATIC_CRT)
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQSceneManagement.lib")
#	pragma comment(lib,"../../VS/Output/Release static CRT/ImagesLoader.lib")
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQAudio.lib")
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQEngine.lib")
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQUtil.lib")
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQUtilMath.lib")
#else
#	pragma comment(lib,"../../VS/Output/Release/HQSceneManagement.lib")
#	pragma comment(lib,"../../VS/Output/Release/ImagesLoader.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQAudio.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQEngine.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQUtil.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQUtilMath.lib")
#endif
#include <stdio.h>
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

bool changeDisplay = false;
bool vsyncChange = false;

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message) 
	{
		case WM_MOUSEMOVE:
			{
				
			}
			break;

		case WM_KILLFOCUS :
			
			break;
		case WM_ACTIVATE:
			
			break;
		case WM_SIZE:
			
			break;
		case WM_KEYDOWN: 
		{
			switch (wParam) 
			{
			case VK_ESCAPE: 
				{
				PostMessage(hwnd, WM_CLOSE, 0, 0);
				return 0;
				} break;
			case VK_SPACE:
				changeDisplay= true;
				break;
			case VK_TAB:
				vsyncChange = true;
				break;
			}
		} break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;

	}


	return DefWindowProc(hwnd, message, wParam, lParam);
}
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#endif

#include "Game.h"

struct A{
	int b;
	A * c;
	HQMatrix4 mat;
};

HQMutex mutex;
class _Thread : public HQThread
{
public:
	void Run() {
		static int i = 0;
		++i;
		char buf[4];
		sprintf(buf, "%d\n" , i);
		OutputDebugStringA(buf);
	}
};

struct OneTimeInit
{
	OneTimeInit()
	{
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#	if defined (_DEBUG) || defined(DEBUG)
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	//_crtBreakAlloc = 1053;

#	endif
#endif
	}
};

static OneTimeInit ontTimeInit;

int HQEngineMain(int argc, char **argv)
{	
/*-----------------------------------------------*/
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	InitCommonControls();
#endif

	
	Game game;

	HQEngineApp::GetInstance()->SetRenderDelegate(game);

	
#if	defined WIN32 || defined APPLE || defined LINUX
	HQEngineApp::GetInstance()->SetKeyListener(game);

	HQEngineApp::GetInstance()->SetMouseListener(game);
	
	HQEngineApp::GetInstance()->SetWindowListener(game);
#endif
#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	HQEngineApp::GetInstance()->SetMotionListener(game);
	HQEngineApp::GetInstance()->SetOrientationListener(game);
	HQEngineApp::GetInstance()->SetAppListener(game);
#endif
	
	HQEngineApp::GetInstance()->ShowWindow();
	TRACE("here %s %d", __FILE__, __LINE__);


	HQEngineApp::GetInstance()->Run(0);
	

	return 0;
}
