/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQThread.h"
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#ifndef HAS_PTHREAD_SETNAME_NP
#	define pthread_setname_np(...)
#endif

struct ThreadData
{
	ThreadData(HQThread *_thread)
	{
		threadObj = _thread;
		flag = 1;
		refCount = 1;
	}
	
	int GetFlag() {HQMutex::ScopeLock sl(mutex); return flag; }
	void SetFlag(int val) {HQMutex::ScopeLock sl(mutex); flag = val; }
	void AddRef() {HQMutex::ScopeLock sl(mutex); refCount++;}
	void Release()
	{
		HQMutex::ScopeLock sl(mutex);
		refCount--;
		if (refCount == 0)
			delete this;
	}
	
	HQThread * threadObj;
	pthread_t threadID;
private:
	HQMutex mutex;
	int flag;
	hquint32 refCount;
};

/*-------thread routine--*/
void* ThreadFunc(void * arg)
{
	ThreadData * data = (ThreadData*) arg;
	data->threadObj->Run();
	data->SetFlag(1);//thread routine has finished
	data->Release();
	
	return NULL;
}
/*-------class-----------*/
HQThread::HQThread(const char *threadName)
{
	m_platformSpecific = new ThreadData(this);

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
	((ThreadData*) m_platformSpecific)->Release();

	if (m_threadName != NULL)
		delete[] m_threadName;
}

void HQThread::Start()
{
	HQMutex::ScopeLock scopeLock(m_mutex);
	ThreadData * data = (ThreadData*) m_platformSpecific;
	if (data->GetFlag() == 0)//still running
			return;
	data->SetFlag(0);
	data->AddRef();
	pthread_create(&data->threadID, NULL, ThreadFunc, m_platformSpecific);

	if (m_threadName != NULL)
		pthread_setname_np(&data->threadID, m_threadName);
}

void HQThread::TempPause()
{
	sched_yield();
}

void HQThread::Join()
{
	HQMutex::ScopeLock scopeLock(m_mutex);
	pthread_join(((ThreadData*) m_platformSpecific)->threadID, NULL);
}
