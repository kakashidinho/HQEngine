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

#define HQ_CUSTOM_MALLOC
#include "../HQEngineCustomHeap.h"

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
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQAudio.lib")
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
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQAudio.lib")
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
	//_crtBreakAlloc = 1058;

#	endif
#endif
	}
};

static OneTimeInit ontTimeInit;

int HQEngineMain(int argc, char **argv)
{	
	int * p = HQ_NEW int;

	TRACE("here %s %d", __FILE__, __LINE__);

	HQAtomic<short> si;

	HQVector4 * pvec = HQVector4::New(1,1,1);
	HQFloat3 f3 = *pvec;
	HQFloat4 f4 = *pvec;
	delete pvec;

	{
		const HQVector4 & buitin1 = HQVector4::Origin();
		TRACE("addres of HQVector4::Origin() = %p", &buitin1);
	}
	{
		const HQVector4 & buitin = HQVector4::PositiveX();
		TRACE("addres of HQVector4::PositiveX() = %p", &buitin);
	}
	{
		const HQVector4 & buitin = HQVector4::NegativeX();
		TRACE("addres of HQVector4::NegativeX() = %p", &buitin);
	}
	{
		const HQVector4 & buitin = HQVector4::PositiveY();
		TRACE("addres of HQVector4::PositiveY() = %p", &buitin);
	}
	{
		
		const HQVector4 & buitin = HQVector4::NegativeY();
		TRACE("addres of HQVector4::NegativeY() = %p", &buitin);
	}
	{
		const HQVector4 & buitin = HQVector4::PositiveZ();
		TRACE("addres of HQVector4::PositiveZ() = %p", &buitin);
	}
	{
		const HQVector4 & buitin = HQVector4::NegativeZ();
		TRACE("addres of HQVector4::NegativeZ() = %p", &buitin);
	}
	size_t svec = sizeof(HQVector4);
	svec = sizeof(HQFloat4);

	svec = sizeof(HQQuaternion);

	size_t smat = sizeof(HQMatrix4);
	smat = sizeof(HQMatrix3x4);

	HQ_DECL_STACK_VECTOR4_CTOR_PARAMS( vec2, (1, 2 ,3));
	

	TRACE("HERE %s, %d", __FILE__, __LINE__);

	HQMutex mutex;
	mutex.Lock();
	mutex.Unlock();
	mutex.Lock();
	

	HQSemaphore sem(3);
	HQSemaphore sem2(3);
	sem.Lock();
	sem.Lock();
	sem.Lock();
	sem.Unlock();
	sem.Lock();
	
	/*
	_Thread thread;
	thread.Start();
	thread.Join();
	thread.Start();
	thread.Join();
	thread.Start();
	thread.Join();
	thread.Start();
	thread.Join();
	thread.Start();
	*/

	TRACE("HERE %s, %d", __FILE__, __LINE__);

	//test HQUtilMath
	HQA16ByteMatrix4Ptr mat(1, 2, 3, 4,
							5, 6, 7, 8,
							9, 10, 11, 12,
							13, 14, 15, 16);

	HQA16ByteMatrix3x4Ptr mat2(1, 2, 3, 4,
							5, 6, 7, 8,
							9, 10, 11, 12);

	TRACE("HERE %s, %d", __FILE__, __LINE__);

	mat->Transpose();

	HQA16ByteVector4Ptr vec3(1, 2, 3);

	TRACE("HERE %s, %d: vec3=%p", __FILE__, __LINE__, vec3.operator void *());

	vec3->Normalize();

	TRACE("HERE %s, %d: vec3=%p", __FILE__, __LINE__, vec3.operator void *());

	HQVector4 *re = vec3;

	size_t sizeVec = sizeof(HQVector4);
	
	TRACE("HERE %s, %d", __FILE__, __LINE__);

	HQA16ByteQuaternionPtr quat;
	quat->QuatFromRotAxisOx(HQPiFamily::_PIOVER3);

	TRACE("quat = {%f, %f, %f, %f}", quat->x, quat->y, quat->z, quat->w);

	quat->QuatToMatrix4r(mat);

	quat->QuatToMatrix3x4c(mat2);

	TRACE("matrix 4 from quat:");
	TRACE("====>{%f, %f, %f, %f}", mat->_11, mat->_12, mat->_13, mat->_14);
	TRACE("====>{%f, %f, %f, %f}", mat->_21, mat->_22, mat->_23, mat->_24);
	TRACE("====>{%f, %f, %f, %f}", mat->_31, mat->_32, mat->_33, mat->_34);
	TRACE("====>{%f, %f, %f, %f}", mat->_41, mat->_42, mat->_43, mat->_44);

	TRACE("matrix 3x4 from quat:");
	TRACE("====>{%f, %f, %f, %f}", mat2->_11, mat2->_12, mat2->_13, mat2->_14);
	TRACE("====>{%f, %f, %f, %f}", mat2->_21, mat2->_22, mat2->_23, mat2->_24);
	TRACE("====>{%f, %f, %f, %f}", mat2->_31, mat2->_32, mat2->_33, mat2->_34);


	HQA16ByteQuaternionPtr quat2;
	HQQuaternion *quat2ptr = quat2;
	TRACE("quat2 = %p", quat2ptr);
	quat2->QuatFromMatrix3x4c(*mat2);

	TRACE("quat from matrix 3x4 = {%f, %f, %f, %f}", quat2->x, quat2->y, quat2->z, quat2->w);

	HQA16ByteQuaternionPtr quat3;
	quat2->QuatFromMatrix4r(*mat);

	TRACE("quat from matrix 4 = {%f, %f, %f, %f}", quat2->x, quat2->y, quat2->z, quat2->w);


	float det;
	HQMatrix4Inverse(mat, &det, mat);

	HQA16ByteMatrix4Ptr mat2Inv;
	HQMatrix3x4Inverse(mat2, mat2Inv);

	//HQMatrix4 identity = *mat2Inv * *mat2  ;

	float length = quat->GetMagnitude();

	TRACE("HERE %s, %d", __FILE__, __LINE__);
/*-----------------------------------------------*/
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	InitCommonControls();
#endif

	TRACE("here %s %d", __FILE__, __LINE__);
	
	Game game;

	TRACE("here %s %d", __FILE__, __LINE__);
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
