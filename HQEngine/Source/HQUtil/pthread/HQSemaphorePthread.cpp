/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQSemaphore.h"
#include <semaphore.h>

HQSemaphore::HQSemaphore(hq_int32 initValue)
: m_platformSpecific (new sem_t())

{
	if (initValue < 1)
		initValue = 1;
	int re = sem_init((sem_t*) m_platformSpecific , 0 , (unsigned int) initValue);
	if(re != 0 )
	{
		throw std::bad_alloc();
	}
}

HQSemaphore::~HQSemaphore()
{
	sem_destroy((sem_t*) m_platformSpecific);
	delete ((sem_t*) m_platformSpecific);
}

bool HQSemaphore::TryLock()
{
	return sem_trywait((sem_t*) m_platformSpecific) == 0;
}

void HQSemaphore::Lock()
{
	sem_wait((sem_t*) m_platformSpecific);
}

void HQSemaphore::Unlock()
{
	sem_post((sem_t*) m_platformSpecific);
}
