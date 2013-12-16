#include "../HQUtilPCH.h"
#include "../../HQSemaphore.h"

HQSemaphore::HQSemaphore(hq_int32 initValue)
{
	if (initValue < 1)
		initValue = 1;
	const LONG maxValue = (LONG)((~((ULONG) 0)) >> 1);
	m_platformSpecific = CreateSemaphore(0 , (LONG) initValue , maxValue , 0);
	if (m_platformSpecific == NULL)
		throw std::bad_alloc();
}

HQSemaphore::~HQSemaphore()
{
	CloseHandle(m_platformSpecific);
}

bool HQSemaphore::TryLock()
{
	return WaitForSingleObject(m_platformSpecific , 0) == 0;
}

void HQSemaphore::Lock()
{
	WaitForSingleObject(m_platformSpecific , INFINITE);
}

void HQSemaphore::Unlock()
{
	ReleaseSemaphore(m_platformSpecific , 1 , NULL);
}