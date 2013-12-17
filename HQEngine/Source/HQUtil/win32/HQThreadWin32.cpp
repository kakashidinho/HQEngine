#include "../HQUtilPCH.h"
#include "../../HQThread.h"

#include <string.h>


/*------trick to set thread name in win32 debugger-------------*/
static const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
   DWORD dwType; // Must be 0x1000.
   LPCSTR szName; // Pointer to name (in user addr space).
   DWORD dwThreadID; // Thread ID (-1=caller thread).
   DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

static void SetThreadName( DWORD dwThreadID, const char* threadName)
{
   THREADNAME_INFO info;
   info.dwType = 0x1000;
   info.szName = threadName;
   info.dwThreadID = dwThreadID;
   info.dwFlags = 0;

   __try
   {
      RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
   }
   __except(EXCEPTION_EXECUTE_HANDLER)
   {
   }
}

/*-------thread routine--*/
DWORD WINAPI ThreadFunc(void * arg)
{
	HQThread * thread = (HQThread*) arg;
	
	if (thread->GetThreadName() != NULL)
		SetThreadName(GetCurrentThreadId(), thread->GetThreadName());

	thread->Run();

	return 1;
}
/*-------class-----------*/
HQThread::HQThread(const char *threadName)
: m_platformSpecific(NULL)
{
	if (threadName != NULL)
	{
		size_t len = strlen(threadName);
		m_threadName = new char[len + 1];

		strcpy(m_threadName, threadName);
	}
	else
		m_threadName = NULL;
}

HQThread::~HQThread()
{
	if (m_platformSpecific != NULL)
		CloseHandle(m_platformSpecific);

	if (m_threadName != NULL)
		delete[] m_threadName;
}

void HQThread::Start()
{
	HQMutex::ScopeLock scopeLock(m_mutex);
	if (m_platformSpecific != NULL)
	{
		DWORD code;
		GetExitCodeThread(m_platformSpecific , &code);
		if (code == STILL_ACTIVE)
			return;

		CloseHandle(m_platformSpecific);
	}
	
	m_platformSpecific = CreateThread(0 , 0 ,ThreadFunc ,this , 0 , 0 );
}

void HQThread::TempPause()
{
	SwitchToThread();
}

void HQThread::Join()
{
	HQMutex::ScopeLock scopeLock(m_mutex);
	if (m_platformSpecific != NULL)
		WaitForSingleObject(m_platformSpecific , INFINITE);
}