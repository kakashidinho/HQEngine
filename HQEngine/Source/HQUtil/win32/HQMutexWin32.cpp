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

HQMutex::HQMutex()
{
	m_platformSpecific = HQ_NEW CRITICAL_SECTION();
	InitializeCriticalSection((CRITICAL_SECTION*)m_platformSpecific);
}

HQMutex::~HQMutex()
{
	DeleteCriticalSection((CRITICAL_SECTION*)m_platformSpecific);

	HQ_DELETE ((CRITICAL_SECTION*)m_platformSpecific);
}

bool HQMutex::TryLock()
{
	return TryEnterCriticalSection((CRITICAL_SECTION*)m_platformSpecific) != 0;
}

void HQMutex::Lock()
{
	EnterCriticalSection((CRITICAL_SECTION*)m_platformSpecific);
}

void HQMutex::Unlock()
{
	LeaveCriticalSection((CRITICAL_SECTION*)m_platformSpecific);
}
