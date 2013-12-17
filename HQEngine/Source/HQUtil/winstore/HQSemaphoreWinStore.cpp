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

#include <windows.h>

HQSemaphore::HQSemaphore(hq_int32 initValue)
{
	if (initValue < 1)
		initValue = 1;
	const LONG maxValue = (LONG)((~((ULONG) 0)) >> 1);
	m_platformSpecific = CreateSemaphoreExW(0 , (LONG) initValue , maxValue , 0, 0, SEMAPHORE_MODIFY_STATE);
	if (m_platformSpecific == NULL)
		throw std::bad_alloc();
}

HQSemaphore::~HQSemaphore()
{
	CloseHandle(m_platformSpecific);
}

bool HQSemaphore::TryLock()
{
	return WaitForSingleObjectEx(m_platformSpecific , 0, FALSE) == 0;
}

void HQSemaphore::Lock()
{
	WaitForSingleObjectEx(m_platformSpecific , INFINITE, FALSE);
}

void HQSemaphore::Unlock()
{
	ReleaseSemaphore(m_platformSpecific , 1 , NULL);
}
