/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "HQEventSeparateThread.h"


#define INIT_MAX_EVENTS_IN_POOL 5
#define INIT_MAX_EVENTS_POOLS 10

/*------HQEvent---------*/
HQEvent & HQEvent::operator=(const HQEvent &srcEvent)
{
	switch (srcEvent.type)
	{
		case HQ_TOUCH_BEGAN:
		case HQ_TOUCH_MOVED:
		case HQ_TOUCH_ENDED:
		case HQ_TOUCH_CANCELLED:
			this->touchData.numTouches = srcEvent.touchData.numTouches;
			for (hquint32 i = 0 ; i < this->touchData.numTouches ; ++i)
				this->touchData.touches[i] = srcEvent.touchData.touches[i];
			break;
		case HQ_KEY_PRESSED:
		case HQ_KEY_RELEASED:
			this->keyData = srcEvent.keyData;
			break;
		case HQ_MOUSE_MOVED:
		case HQ_MOUSE_RELEASED:
		case HQ_MOUSE_PRESSED:
		case HQ_MOUSE_WHEEL:
			this->mouseData = srcEvent.mouseData;
			break;
	}
	this->type = srcEvent.type;
	return *this;
}

hquint32 HQEvent::GetNumTouches() const
{
	return touchData.numTouches;
}

hqint32 HQEvent::GetTouchID(hquint32 index) const
{
	return touchData.touches[index].touchID;
}

const HQPointf & HQEvent::GetPosition(hquint32 index) const 
{
	return touchData.touches[index].position;
}

const HQPointf & HQEvent::GetPrevPosition(hquint32 index) const 
{
	return touchData.touches[index].prev_position;
}

/*------HQEventQueue--------*/
HQEventQueue::HQEventQueue()
	: m_events(new HQPoolMemoryManager(sizeof(EventListType::LinkedListNodeType), INIT_MAX_EVENTS_IN_POOL, INIT_MAX_EVENTS_POOLS, false)),
	m_acceptEvent(true)
{
	//event list can hold at least (INIT_MAX_EVENTS_IN_POOL * INIT_MAX_EVENTS_POOLS) events
}

HQEvent * HQEventQueue::GetFirstEvent()
{
	if (m_events.GetSize() == 0)
		return NULL;
	return &m_events.GetFront();
}

void HQEventQueue::RemoveFirstEvent()
{
	if (m_events.GetSize() == 0)
		return ;
	
	m_events.PopFront();
}


HQEvent & HQEventQueue::BeginAddEvent() 
{
	EventListType::LinkedListNodeType* newNode = m_events.PushBack();
	if (newNode == NULL)
	{
		//no choice but have to remove the oldest event
		this->RemoveFirstEvent();

		newNode = m_events.PushBack();//retry
	}
	
	return newNode->m_element;
}
void HQEventQueue::EndAddEvent()
{
	if (!m_acceptEvent)
	{
		//discard
		m_events.PopBack();
	}
#if 0 && defined WIN32 && ( defined DEBUG || defined _DEBUG)
	else
	{
		static char buffer [512];
		sprintf(buffer, "number of events = %u\n", m_events.GetSize());

		OutputDebugStringA(buffer);
	}
#endif
}


void HQEventQueue::Lock()
{
	m_mutex.Lock();
}

void HQEventQueue::Unlock()
{
	m_mutex.Unlock();
	HQThread::TempPause();//give game thread a chance to process the event
}

void HQEventQueue::Enable(bool enable)
{
	m_acceptEvent = enable;
}

