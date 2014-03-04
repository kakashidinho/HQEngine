/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_THREAD_H
#define HQ_THREAD_H

#include "HQUtil.h"

#include <stdlib.h>

class HQUTIL_API HQThread
{
public:
	HQThread(const char *threadName = NULL);
	virtual ~HQThread();
	
	///start thread
	void Start();

	///temporarily pause the calling thread and allow other threads to run
	static void TempPause();
	///wait for this thread to die
	void Join();

	const char *GetThreadName() const {return m_threadName;}///the thread name can be NULL

	///implement dependent thread procedure
	virtual void Run() = 0;
private:
	HQMutex m_mutex;
	void *m_platformSpecific;
	char * m_threadName;
};

#endif
