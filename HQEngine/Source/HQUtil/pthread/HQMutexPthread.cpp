/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQMutex.h"
#include <pthread.h>

HQMutex::HQMutex()
: m_platformSpecific( new pthread_mutex_t ())
{
	if(pthread_mutex_init((pthread_mutex_t*) m_platformSpecific  , NULL) != 0)
		throw std::bad_alloc();
}

HQMutex::~HQMutex()
{
	pthread_mutex_destroy((pthread_mutex_t*) m_platformSpecific);
	delete ((pthread_mutex_t*)m_platformSpecific);
}

bool HQMutex::TryLock()
{
	return pthread_mutex_trylock((pthread_mutex_t*) m_platformSpecific) == 0;
}

void HQMutex::Lock()
{
	pthread_mutex_lock((pthread_mutex_t*) m_platformSpecific);
}

void HQMutex::Unlock()
{
	pthread_mutex_unlock((pthread_mutex_t*) m_platformSpecific);
}
