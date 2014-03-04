/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQThread.h"

#include <thread>

/*------trick to set thread name in win32 debugger-------------*/
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM

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

#endif//#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM

/*-------thread routine--*/
void ThreadFunc(HQThread * thread)
{
	if (thread->GetThreadName() != NULL)
	{
		//set thread name
#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM

		SetThreadName(GetCurrentThreadId(), thread->GetThreadName());

#endif//#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
	}
	thread->Run();
}
/*-------class-----------*/
HQThread::HQThread(const char * threadName)
	:m_platformSpecific(NULL)
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
	{
		if (((std::thread*)m_platformSpecific)->joinable())
			((std::thread*)m_platformSpecific)->join();
		delete ((std::thread*)m_platformSpecific);
	}

	if (m_threadName != NULL)
		delete[] m_threadName;
}

void HQThread::Start()
{
	HQMutex::ScopeLock scopeLock(m_mutex);
	if (m_platformSpecific != NULL)
	{
		if (((std::thread*)m_platformSpecific)->joinable())//still running
			return;

		delete ((std::thread*)m_platformSpecific);
		m_platformSpecific = NULL;
	}
	
	m_platformSpecific = HQ_NEW std::thread(ThreadFunc, this);
}

void HQThread::TempPause()
{
	std::this_thread::yield();
}

void HQThread::Join()
{
	HQMutex::ScopeLock scopeLock(m_mutex);
	if (m_platformSpecific != NULL)
	{
		if (((std::thread*)m_platformSpecific)->joinable())
			((std::thread*)m_platformSpecific)->join();
	}
}
