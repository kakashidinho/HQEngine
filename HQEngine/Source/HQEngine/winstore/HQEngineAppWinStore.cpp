/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../../HQEngineApp.h"
#include "../../HQAtomic.h"
#include "../../HQConditionVariable.h"
#include "../HQEngineWindow.h"
#include "../HQEventSeparateThread.h"
#include "HQWinStoreFileSystem.h"
#include "HQWinStoreUtil.h"

using namespace Windows::ApplicationModel;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::UI::Core;
#if defined HQ_WIN_PHONE_PLATFORM
using namespace Windows::Phone::UI::Input;
#endif
using namespace Windows::System;
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;


static volatile ULONGLONG g_cursorState;

#define HQ_THREAD_STARTED 0x1
#define HQ_THREAD_RUNNING 0x2

#define HQ_CURSOR_VISIBLE 0x1

#define USE_MUTEX 0

int hq_internalAppNormalState = 0;
int hq_internalAppExit = 1;
int hq_internalAppIsPaused = 2;

static HQSimpleConditionVar g_pauseGameThreadCnd;
static HQAtomic<bool> g_game_thread_blocked = false;


#if USE_MUTEX
static int g_appStatusFlag = hq_internalAppNormalState;
static HQMutex g_mutex;
#else
static volatile ULONGLONG g_appStatusFlag = hq_internalAppNormalState;
#endif

void HQAppInternalSetStatusFlag(int flag)
{
#if USE_MUTEX
	HQMutex::ScopeLock sl(g_mutex);
	g_appStatusFlag = flag;
#else
	InterlockedExchange(&g_appStatusFlag, flag);
#endif
}


int HQAppInternalGetStatusFlag()
{
#if USE_MUTEX
	HQMutex::ScopeLock sl(g_mutex);
	return g_appStatusFlag;
#else
	return InterlockedOr(&g_appStatusFlag, 0x0);
#endif
}

void HQAppInternalBlockGameLoopIfNeeded()
{
	if (!g_pauseGameThreadCnd.TryLock())
		return;

	if (HQAppInternalGetStatusFlag() == hq_internalAppIsPaused)
	{
		g_game_thread_blocked = true;

		g_pauseGameThreadCnd.Wait();

	}

	g_pauseGameThreadCnd.Unlock();
}

static void HQAppInternalWakeGameLoopIfNeededNoLock()
{
	bool shouldWake = false;

	shouldWake = g_game_thread_blocked;
	g_game_thread_blocked = false;

	if (shouldWake)
		g_pauseGameThreadCnd.Signal();

}

static void HQAppInternalWakeGameLoopIfNeeded()
{
	g_pauseGameThreadCnd.Lock();
	
	HQAppInternalWakeGameLoopIfNeededNoLock();

	g_pauseGameThreadCnd.Unlock();

}


//stuff that init once the app starts
struct AppInitOnce
{
	AppInitOnce()
	{
	}

	~AppInitOnce()
	{
	}
};

static AppInitOnce oneTimeInit;

/*-----------thread safe functions--------*/
static void HQSetCursorStateThreadSafe(bool visible)
{
	if (visible)
		InterlockedOr(&g_cursorState, HQ_CURSOR_VISIBLE);
	else
		InterlockedAnd(&g_cursorState, ~HQ_CURSOR_VISIBLE);
}

static bool HQIsCursorVisibleThreadSafe()
{
	register ULONGLONG state;
	InterlockedExchange(&state, g_cursorState);

	return (state & HQ_CURSOR_VISIBLE) != 0;
}

/*-----------game thread------------------*/
HQEventQueue* hq_engine_eventQueue_internal;
HQGameThead* hq_engine_GameThread_internal;
Platform::Agile <CoreWindow> hq_engine_coreWindow_internal;
#ifdef HQ_WIN_STORE_PLATFORM
Platform::Agile <CoreCursor> hq_engine_defaultCursor_internal;
#endif

HQGameThead ::HQGameThead(const char *threadName)
	:HQThread(threadName)
{
	InterlockedExchange(&m_state, 0);
}

bool HQGameThead :: IsStarted()
{
	register ULONGLONG state;
	InterlockedExchange(&state, m_state);

	return (state & HQ_THREAD_STARTED) != 0;
}


bool HQGameThead :: IsRunning()
{
	register ULONGLONG state;
	InterlockedExchange(&state, m_state);

	return (state & HQ_THREAD_RUNNING) != 0;
}

void HQGameThead :: Run()
{
	InterlockedOr(&m_state, HQ_THREAD_STARTED);

	InterlockedOr(&m_state, HQ_THREAD_RUNNING);
	
	m_args->re = m_args->entryFunc(m_args->argc, m_args->argv);

	delete hq_engine_eventQueue_internal;


	HQWinStoreUtil::RunOnUIThread(hq_engine_coreWindow_internal->Dispatcher, [&] ()
		{
			InterlockedAnd(&m_state, ~HQ_THREAD_RUNNING);//release and notify the blocked ui thread
		}
	);
}

/*------HQEngineApp implemetation---------*/
void HQEngineApp::PlatformInit()
{
}

void HQEngineApp::PlatformRelease()
{
}

HQDataReaderStream* HQEngineApp::OpenFileStream(const char *file)
{
	return HQWinStoreFileSystem::OpenFileForRead(file);
}

bool HQEngineApp::EventHandle()
{
	bool hasEvent = false;
	HQEvent *nextEvent;
	hq_engine_eventQueue_internal->Lock();
	while ((nextEvent = hq_engine_eventQueue_internal->GetFirstEvent()) != NULL)
	{
		HQEvent curEvent = *nextEvent;//copy event
		hq_engine_eventQueue_internal->RemoveFirstEvent();//remove from event queue		
		hq_engine_eventQueue_internal->Unlock();

		switch (curEvent.type) {
			case HQ_TOUCH_BEGAN:
				m_motionListener->TouchBegan(curEvent);
				break;
			case HQ_TOUCH_MOVED:
				m_motionListener->TouchMoved(curEvent);
				break;
			case HQ_TOUCH_ENDED:
				m_motionListener->TouchEnded(curEvent);
				break;
			case HQ_TOUCH_CANCELLED:
				m_motionListener->TouchCancelled(curEvent);
				break;
			case HQ_ORIENTATION_PORTRAIT:
				m_orientListener->ChangedToPortrait();
				break;
			case HQ_ORIENTATION_PORTRAIT_UPSIDE_DOWN:
				m_orientListener->ChangedToPortraitUpsideDown();
				break;
			case HQ_ORIENTATION_LANDSCAPE_LEFT:
				m_orientListener->ChangedToLandscapeLeft();
				break;
			case HQ_ORIENTATION_LANDSCAPE_RIGHT:
				m_orientListener->ChangedToLandscapeRight();
				break;
			case HQ_MOUSE_MOVED:
				m_mouseListener->MouseMove(curEvent.mouseData.position);
				break;
			case HQ_MOUSE_RELEASED:
				m_mouseListener->MouseReleased(curEvent.mouseData.keyCode, curEvent.mouseData.position);
				break;
			case HQ_MOUSE_PRESSED:
				m_mouseListener->MousePressed(curEvent.mouseData.keyCode, curEvent.mouseData.position);
				break;
			case HQ_MOUSE_WHEEL:
				m_mouseListener->MouseWheel(curEvent.mouseData.wheelDelta, curEvent.mouseData.position);
				break;
			case HQ_KEY_PRESSED:
				m_keyListener->KeyPressed(curEvent.keyData.keyCode);
				break;
			case HQ_KEY_RELEASED:
				m_keyListener->KeyReleased(curEvent.keyData.keyCode);
				break;
			default:
				break;
		}
		
		
		hasEvent = true;
	}
	
	hq_engine_eventQueue_internal->Unlock();	

	
	return hasEvent;
}

HQReturnVal HQEngineApp::PlatformEnableMouseCursor(bool enable)
{
	if (m_window == NULL)
		return HQ_FAILED;

	m_window->GetWindowEventHandler()->HideMouse(!enable);

	//HQSetCursorStateThreadSafe(enable); //not needed for now
	
	return HQ_OK;
}

const wchar_t * HQEngineApp::GetCurrentDir()
{
	return HQWinStoreFileSystem::GetCurrentDirConst();
}
void HQEngineApp::SetCurrentDir(const wchar_t *dir)
{
	HQWinStoreFileSystem::SetCurrentDir(dir);
}
void HQEngineApp::SetCurrentDir(const char *dir)
{
	HQWinStoreFileSystem::SetCurrentDir(dir);
}

/*----framework view class--------------*/
ref class HQFrameWorkView sealed : public Windows::ApplicationModel::Core::IFrameworkView
{
public:
	HQFrameWorkView()
		:m_windowVisible(true),
		m_windowClosed(false)
	{
		//game thread event queue must be initialized first
		hq_engine_eventQueue_internal = HQ_NEW HQEventQueue();
	}
	
	// IFrameworkView Methods.
	virtual void Initialize(Windows::ApplicationModel::Core::CoreApplicationView^ applicationView)
	{

		applicationView->Activated +=
        ref new TypedEventHandler<CoreApplicationView^, IActivatedEventArgs^>(this, &HQFrameWorkView::OnActivated);

		CoreApplication::Suspending +=
			ref new EventHandler<SuspendingEventArgs^>(this, &HQFrameWorkView::OnSuspending);

		CoreApplication::Resuming +=
			ref new EventHandler<Platform::Object^>(this, &HQFrameWorkView::OnResuming);

#if defined HQ_WIN_PHONE_PLATFORM
		HardwareButtons::BackPressed += ref new EventHandler<BackPressedEventArgs^>(this, &HQFrameWorkView::OnBackButtonPressed); 
#endif

		Windows::Graphics::Display::DisplayProperties::OrientationChanged += ref new Windows::Graphics::Display::DisplayPropertiesEventHandler(this, &HQFrameWorkView::OnOrientationChanged); 
	}
	virtual void SetWindow(Windows::UI::Core::CoreWindow^ window)
	{
		window->VisibilityChanged +=
			ref new TypedEventHandler<CoreWindow^, VisibilityChangedEventArgs^>(this, &HQFrameWorkView::OnVisibilityChanged);
		
		window->Closed +=
			ref new TypedEventHandler<CoreWindow^, CoreWindowEventArgs^>(this, &HQFrameWorkView::OnWindowClosed);

		hq_engine_coreWindow_internal = CoreWindow::GetForCurrentThread();
#ifdef HQ_WIN_STORE_PLATFORM
		hq_engine_defaultCursor_internal = ref new CoreCursor(CoreCursorType::Arrow, 1);
#endif

		//everything needed is available, start the game thread
		hq_engine_GameThread_internal->Start();
	}
	virtual void Load(Platform::String^ entryPoint)
	{
	}
	virtual void Run()
	{
		//wait for game thread start running
		while (!hq_engine_GameThread_internal->IsStarted())
		{
			HQTimer::Sleep(0.01f);//sleep 10 microseconds
		}

		//wait for game thread exit
		while (hq_engine_GameThread_internal->IsRunning())
		{
			hq_engine_coreWindow_internal->Dispatcher->ProcessEvents(CoreProcessEventsOption::ProcessOneAndAllPending);//blocking for next events
		}
	}
	virtual void Uninitialize()
	{
	}

	void OnActivated(CoreApplicationView^ applicationView, IActivatedEventArgs^ args)
	{
		CoreWindow::GetForCurrentThread()->Activate();
	}

	void OnSuspending(Platform::Object^ sender, Windows::ApplicationModel::SuspendingEventArgs^ args)
	{
		_OndPause(); 
	}
	void OnResuming(Platform::Object^ sender, Platform::Object^ args)
	{
		_OnResume();
	}
	void OnVisibilityChanged(CoreWindow^ sender, VisibilityChangedEventArgs^ args)
	{
		m_windowVisible = args->Visible;
		if (m_windowVisible)
		{
			_OnResume();
		}
		else
		{
			_OndPause();
		}
	}

	void OnWindowClosed(CoreWindow^ sender, CoreWindowEventArgs ^ ags)
	{
		m_windowClosed = true;
	}
#if defined HQ_WIN_PHONE_PLATFORM
	void OnBackButtonPressed(Platform::Object^ sender, Windows::Phone::UI::Input::BackPressedEventArgs^ args)
	{
		bool exit = true;
		if (HQEngineApp::GetInstance() != NULL)
			exit = HQEngineApp::GetInstance()->GetAppListener()->BackButtonPressed();

		args->Handled = !exit;
	}
#endif

	void OnOrientationChanged(Platform::Object^ sender)
	{
		HQEventQueue::ScopeLock sl (hq_engine_eventQueue_internal);


		switch (DisplayProperties::CurrentOrientation)
		{
			case DisplayOrientations::Landscape:
				{
					HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
					newEvent.type = HQ_ORIENTATION_LANDSCAPE_LEFT;
					hq_engine_eventQueue_internal->EndAddEvent();
				}
				break;

			case DisplayOrientations::Portrait:
				{
					HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
					newEvent.type = HQ_ORIENTATION_PORTRAIT;
					hq_engine_eventQueue_internal->EndAddEvent();
				}
				break;

			case DisplayOrientations::LandscapeFlipped:
				{
					HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
					newEvent.type = HQ_ORIENTATION_LANDSCAPE_RIGHT;
					hq_engine_eventQueue_internal->EndAddEvent();
				}
				break;

			case DisplayOrientations::PortraitFlipped:
				{
					HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
					newEvent.type = HQ_ORIENTATION_PORTRAIT_UPSIDE_DOWN;
					hq_engine_eventQueue_internal->EndAddEvent();
				}
				break;

			default:
				break;
		}
	}
private :

	void _OnResume()
	{
		if (HQAppInternalGetStatusFlag() == hq_internalAppIsPaused)
		{
			g_pauseGameThreadCnd.Lock();
			HQAppInternalSetStatusFlag(hq_internalAppNormalState);

			HQAppInternalWakeGameLoopIfNeededNoLock();//wake the game thread

			g_pauseGameThreadCnd.Unlock();

			if (HQEngineApp::GetInstance() != NULL)
				HQEngineApp::GetInstance()->GetAppListener()->OnResume();
		}
	}

	void _OndPause()
	{
		if (HQAppInternalGetStatusFlag() == hq_internalAppNormalState)
		{
			HQAppInternalSetStatusFlag(hq_internalAppIsPaused);
			if (HQEngineApp::GetInstance() != NULL)
				HQEngineApp::GetInstance()->GetAppListener()->OnPause();
		}
	}

	bool m_windowVisible;
	bool m_windowClosed;
};

/*----framework view source class--------------*/
ref class HQFrameWorkViewSource sealed : Windows::ApplicationModel::Core::IFrameworkViewSource
{
public:
	virtual Windows::ApplicationModel::Core::IFrameworkView^ CreateView()
	{
		return ref new HQFrameWorkView();
	}
};

/*---------helper functions----------*/
char ** HQWinStoreGetCmdLineWrapper(Platform::Array<Platform::String^>^refArgs, int &argc)
{
	if (refArgs == nullptr)
	{
		argc = 0;
		return NULL;
	}

	argc = refArgs->Length;

	/*------get ANSI arguments-----------*/
	char **argv = NULL;
	try{
		argv = HQ_NEW char*[argc];
		for (int i = 0 ; i < argc; ++i)
		{
			argv[i] = NULL;//initialize all to null
		}
		for (int i = 0 ; i < argc; ++i)
		{
			size_t len = wcstombs(NULL, refArgs[i]->Data(), 0);
			argv[i] = HQ_NEW char[len + 1];
			wcstombs(argv[i], refArgs[i]->Data(), len + 1);
		}

	}
	catch (std::bad_alloc e)
	{
		HQWinStoreFreeCmdLineArgs(argv, argc);
		return NULL;
	}

	return argv;
}

void HQWinStoreFreeCmdLineArgs(char ** &argv, int argc)
{
	if (argv == NULL)
		return;
	for (int i = 0 ; i < argc; ++i)
	{
		if (argv[i] != NULL)
		{
			delete[] argv[i];
			argv[i] = NULL;
		}
	}
	delete[] argv;
	argv = NULL;
}

Windows::ApplicationModel::Core::IFrameworkViewSource ^ HQWinStoreCreateFWViewSource()
{
	return ref new HQFrameWorkViewSource();
}

/*--------debug operator new---------------*/

#ifdef malloc
#undef malloc
#endif

#ifdef free
#undef free
#endif

#ifdef realloc
#undef realloc
#endif

#if (defined HQ_WIN_STORE_PLATFORM  || defined HQ_WIN_PHONE_PLATFORM) && (defined _DEBUG || defined DEBUG)
#	define HQ_DEBUG_MALLOC 1
#endif

#if HQ_DEBUG_MALLOC
#include "../../HQMutex.h"

#include <unordered_map>

struct MallocInfo
{
	void c_tor(void *addr, const char *file, int line, ULONG64 id)
	{
		this->addr = addr;
		if (file != NULL)
		{
			this->file = (char *)malloc (strlen(file) + 1);
			strcpy(this->file, file);
		}
		else
			this->file = NULL;
		this->line = line;
		this->id = id;
	}

	void d_tor()
	{
		if (this->file)
		{
			free(this->file);
			this->file = NULL;
		}
	}

	void *addr;
	char *file;
	int line;
	ULONG64 id;
};

struct MallocManager
{
	template <class elemType>
	class TableAllocator
	{
	public:
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;
		typedef elemType* pointer;
		typedef const elemType* const_pointer;
		typedef elemType& reference;
		typedef const elemType& const_reference;
		typedef elemType value_type;

		template <class U> struct rebind { typedef TableAllocator<U>
											other; };
 
		TableAllocator() throw() {}
		TableAllocator(const TableAllocator& src) throw() {}
		template <class U> TableAllocator(const TableAllocator<U>& src) throw() { }
		~TableAllocator() throw() {}

		elemType* allocate(size_t size, const elemType* hint = 0)
		{
			return (elemType*) malloc(size * sizeof (elemType));
		}

		void deallocate(elemType* p, size_t size)
		{
			free(p);
		}

		size_t max_size() { return 0xffffffff; };
 
		void construct(elemType* p, const elemType& val)
		{
			new (p) elemType(val);
		}

		void destroy(elemType* p)
		{
			p->~elemType();
		}
	};

	typedef std::unordered_map<
		void*,  
		MallocInfo* , 
		std::hash <void*>, 
		std::equal_to< void*>,
		TableAllocator <std::pair<void*, MallocInfo*>>> 
			TableType;

	MallocManager()
	{
		m_currentID = 0;
		InitializeCriticalSectionEx(&m_mutex, 0, 0);
	}
	~MallocManager()
	{
		EnterCriticalSection (&m_mutex);

		for (auto ite = m_mallocTable.begin();
			ite != m_mallocTable.end();
			++ite)
		{
			MallocInfo *current = ite->second;

			char *buffer = (char*)malloc(1024);
			sprintf(buffer, "Detected memory leak: ID = %llu", current->id);
			OutputDebugStringA(buffer);

			if (current->file != NULL)
			{
				OutputDebugStringA(" in file:\"");
				OutputDebugStringA(current->file);
				OutputDebugStringA("\" line:");
				sprintf(buffer, "%d\n", current->line);
				OutputDebugStringA(buffer);
			}
			else
				OutputDebugStringA("\n");

			free(buffer);

			current->d_tor();
			free(current);
		}

		LeaveCriticalSection(&m_mutex);
	}

	void* Add(size_t size, const char *file, int line)
	{
		EnterCriticalSection (&m_mutex);
		char *re_addr = NULL;
		char *newBlock = (char *)malloc(sizeof(MallocInfo) + size);
		if (newBlock == NULL)
			re_addr = NULL;
		else
		{
			re_addr = newBlock + sizeof(MallocInfo);

			MallocInfo *info = (MallocInfo*)newBlock;
			info->c_tor(re_addr, file, line, m_currentID++);

			m_mallocTable[re_addr] = info;
		}

		LeaveCriticalSection(&m_mutex);

		return re_addr;
	}

	void* Resize(void *addr, size_t size, const char *file, int line)
	{
		if (addr == NULL)
			return Add(size, file, line);

		char *re_addr = NULL;
		EnterCriticalSection (&m_mutex);

		//find the block
		auto ite = m_mallocTable.find(addr);
		if (ite == m_mallocTable.end())
		{
			//this block was not allocated by this manager
			re_addr = (char*)realloc(addr, size);
		}
		else
		{
			//found
			MallocInfo *info = ite->second;
			
			char *newBlock = (char *)realloc(info, size + sizeof(MallocInfo));
			if (newBlock != NULL)
			{
				m_mallocTable.erase(ite);

				info = (MallocInfo*) newBlock;
				info->d_tor();//clean up old values

				re_addr = newBlock + sizeof(MallocInfo);

				info->c_tor(re_addr, file, line, m_currentID++);

				m_mallocTable[re_addr] = info;
			}
		}
		LeaveCriticalSection(&m_mutex);

		return re_addr;
	}

	void Remove(void *addr)
	{
		if (addr == NULL)
			return;
		EnterCriticalSection (&m_mutex);

		auto ite = m_mallocTable.find(addr);
		if (ite == m_mallocTable.end())
			free(addr);//this block was not alloc by this manager
		else
		{
			MallocInfo *info = ite->second;
			m_mallocTable.erase(ite);
			info->d_tor();
			free(info);
		}

		LeaveCriticalSection(&m_mutex);
	}

	TableType m_mallocTable;
	CRITICAL_SECTION m_mutex;
	ULONG64 m_currentID;
};

#endif //HQ_DEBUG_MALLOC

#if HQ_DEBUG_MALLOC
static MallocManager *g_pMallocManager = NULL;
#endif

void *HQEngineMalloc (size_t size, const char *file, int line)
{
#if HQ_DEBUG_MALLOC
	static MallocManager mallocManager;
	g_pMallocManager = &mallocManager;
	return g_pMallocManager->Add(size, file, line);
#else
	return  malloc(size);
#endif
}

void *HQEngineRealloc (void *ptr, size_t size, const char *file, int line)
{
	if (ptr == NULL)
		return HQEngineMalloc(size, file, line);
#if HQ_DEBUG_MALLOC
	if (g_pMallocManager != NULL)
		return g_pMallocManager->Resize(ptr, size, file, line);
	else
#endif
		return realloc(ptr, size);
}

void HQEngineFree (void *ptr, const char *file, int line)
{
#if HQ_DEBUG_MALLOC
	if (g_pMallocManager != NULL)
		g_pMallocManager->Remove(ptr);
	else
#endif
		free(ptr);
}

#if 0
void  _cdecl operator delete (void* p, const char *file, int line)
{
	HQEngineFree(p, file, line);
}

void  _cdecl operator delete[] (void* p, const char *file, int line)
{
	HQEngineFree(p, file, line);
}

void * _cdecl operator new (size_t cbSize, const char *file, int line)
{
    void *p = HQEngineMalloc(cbSize, file, line); 
	if (p == NULL)
		throw std::bad_alloc();

    return p;

}

void * _cdecl operator new[] (size_t cbSize, const char *file, int line)
{
    void *p = HQEngineMalloc(cbSize, file, line); 
	if (p == NULL)
		throw std::bad_alloc();

    return p;

}

void  _cdecl operator delete (void* p)
{
	HQEngineFree(p, NULL, 0);
}

void  _cdecl operator delete[] (void* p)
{
	HQEngineFree(p, NULL, 0);
}

void * _cdecl operator new (size_t cbSize)
{
    void *p = HQEngineMalloc(cbSize, NULL, 0); 
	if (p == NULL)
		throw std::bad_alloc();

    return p;

}

void * _cdecl operator new[] (size_t cbSize)
{
    void *p = HQEngineMalloc(cbSize, NULL, 0); 
	if (p == NULL)
		throw std::bad_alloc();

    return p;

}
#endif
