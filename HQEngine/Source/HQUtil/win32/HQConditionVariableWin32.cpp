#include "../HQUtilPCH.h"
#include "../../HQConditionVariable.h"
#include "../../HQAtomic.h"

#include <new>

struct HQWin32CondVarData
{
	HQWin32CondVarData()
		:m_waitCount(0)
	{
	}

	CRITICAL_SECTION m_critSection;
	HANDLE m_autoResetEvent;
	HQAtomic<hquint32> m_waitCount;
};

HQSimpleConditionVar::HQSimpleConditionVar()
{
	m_platformSpecific = HQ_NEW HQWin32CondVarData();

	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;

	InitializeCriticalSection(&data->m_critSection);

	data->m_autoResetEvent = CreateEvent (NULL, FALSE, FALSE, NULL);//auto reset event

	if (data->m_autoResetEvent == NULL)
	{
		DeleteCriticalSection(&data->m_critSection);
		HQ_DELETE (data);

		throw std::bad_alloc();
	}
}

HQSimpleConditionVar::~HQSimpleConditionVar()
{
	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;

	DeleteCriticalSection(&data->m_critSection);

	CloseHandle(data->m_autoResetEvent);

	HQ_DELETE (data);
}

bool HQSimpleConditionVar::TryLock()
{
	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;
	return TryEnterCriticalSection(&data->m_critSection) != 0;
}

void HQSimpleConditionVar::Lock()
{
	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;
	EnterCriticalSection(&data->m_critSection);
}

void HQSimpleConditionVar::Unlock()
{
	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;
	LeaveCriticalSection(&data->m_critSection);
}

void HQSimpleConditionVar::Wait()
{
	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;

	data->m_waitCount ++;

	LeaveCriticalSection(&data->m_critSection);//release mutex


	WaitForSingleObject(data->m_autoResetEvent, INFINITE);//wait for event

	//re acquire mutex
	EnterCriticalSection(&data->m_critSection);

	data->m_waitCount --;
}

void HQSimpleConditionVar::Signal()
{
	HQWin32CondVarData *data = (HQWin32CondVarData*)m_platformSpecific;

	if ( data->m_waitCount > 0)
		SetEvent(data->m_autoResetEvent);
}