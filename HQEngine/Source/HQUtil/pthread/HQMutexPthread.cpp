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