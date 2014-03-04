/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SEMAPHORE_H
#define HQ_SEMAPHORE_H
#include "HQUtil.h"

class HQUTIL_API HQSemaphore
{
public:
	class ScopeLock
	{
	public:
		inline ScopeLock(HQSemaphore &sem) : m_sem(sem) {m_sem.Lock();}
		inline ~ScopeLock() {m_sem.Unlock();}
	private:
		HQSemaphore &m_sem;
	};
	
	///
	///if {initValue} < 1 , it's treated as one 
	///
	HQSemaphore(hq_int32 initValue);
	~HQSemaphore();
	
	///decrease semaphore value by one and if it's negative, return false, and the calling thread does not own the semaphore, the semaphore's state is left unchanged as before calling this method
	bool TryLock();

	///decrease semaphore value by one and if it's negative, lock the calling thread
	void Lock();
	///increase semaphore value by one and if it's zero, one of waiting threads will be resumed
	void Unlock();
private:
	void *m_platformSpecific;
};

#endif
