/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQConditionVariable.h"

#include <pthread.h>
#include <new>

struct HQPthreadCondVarData
{
	pthread_mutex_t m_mutex;
	pthread_cond_t m_condVar;
};

HQSimpleConditionVar::HQSimpleConditionVar()
{
	m_platformSpecific = HQ_NEW HQPthreadCondVarData();

	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;

	if(pthread_mutex_init(&data->m_mutex  , NULL) != 0)
		throw std::bad_alloc();

	if (pthread_cond_init(&data->m_condVar, NULL) != 0)
		throw std::bad_alloc();
}

HQSimpleConditionVar::~HQSimpleConditionVar()
{
	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;

	pthread_mutex_destroy(&data->m_mutex);

	pthread_cond_destroy(&data->m_condVar);

	HQ_DELETE (data);
}

bool HQSimpleConditionVar::TryLock()
{
	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;
	return pthread_mutex_trylock(&data->m_mutex) == 0;
}

void HQSimpleConditionVar::Lock()
{
	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;
	pthread_mutex_lock(&data->m_mutex);
}

void HQSimpleConditionVar::Unlock()
{
	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;
	pthread_mutex_unlock(&data->m_mutex);
}

void HQSimpleConditionVar::Wait()
{
	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;

	pthread_cond_wait(&data->m_condVar , &data->m_mutex);
}

void HQSimpleConditionVar::Signal()
{
	HQPthreadCondVarData *data = (HQPthreadCondVarData*)m_platformSpecific;

	pthread_cond_signal(&data->m_condVar);
}
