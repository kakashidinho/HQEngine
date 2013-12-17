/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_EVENT_QUEUE_SEP_THREAD_INC_H
#define HQ_EVENT_QUEUE_SEP_THREAD_INC_H

#include "../HQ2DMath.h"
#include "../HQEngineAppDelegate.h"
#include "../HQThread.h"
#include "../HQMemoryManager.h"
#include "../HQLinkedList.h"

#define MAX_MULTITOUCHES 10

enum HQEventType
{
	HQ_TOUCH_BEGAN = 0,
	HQ_TOUCH_MOVED = 1,
	HQ_TOUCH_ENDED = 2,
	HQ_TOUCH_CANCELLED = 3,
	HQ_ORIENTATION_PORTRAIT,
	HQ_ORIENTATION_PORTRAIT_UPSIDE_DOWN,
	HQ_ORIENTATION_LANDSCAPE_LEFT,//right side is top
	HQ_ORIENTATION_LANDSCAPE_RIGHT,//left side is right
	HQ_MOUSE_MOVED,
	HQ_MOUSE_RELEASED,
	HQ_MOUSE_PRESSED,
	HQ_MOUSE_WHEEL,
	HQ_KEY_PRESSED,
	HQ_KEY_RELEASED,
};

struct HQTouch
{
	hqint32 touchID;
	HQPointf position;
	HQPointf prev_position;
};

struct HQTouchData
{
	HQTouch touches[MAX_MULTITOUCHES];
	hquint32 numTouches;
};

struct HQMouseData
{
	HQPointi position;
	HQMouseKeyCodeType keyCode;
	hq_float32 wheelDelta;
};

struct HQKeyData
{
	HQKeyCodeType keyCode;
};

struct HQEvent : public HQTouchEvent
{
	HQEventType type;
	union{
		HQTouchData touchData;
		HQKeyData keyData;
		HQMouseData mouseData;
	};

	HQEvent & operator=(const HQEvent &srcEvent);
	
	
	hquint32 GetNumTouches() const;
	hqint32 GetTouchID(hquint32 index) const;
	const HQPointf & GetPosition(hquint32 index) const ;
	const HQPointf & GetPrevPosition(hquint32 index) const ;
};

struct HQEventQueueNode
{
	HQEvent event;
	HQEventQueueNode *nextEvent;
};

class HQEventQueue
{
public:
	class ScopeLock {
	public:
		ScopeLock(HQEventQueue *queue) {queue->Lock(); m_queue = queue;}
		~ScopeLock() {m_queue->Unlock();}
	private:
		HQEventQueue *m_queue;
	};
	
	HQEventQueue();
	
	void Enable(bool enable);//enable event queue to accept new event
	bool IsEnabled() const {return m_acceptEvent;}
	HQEvent * GetFirstEvent();
	void RemoveFirstEvent();//remove first event from queue
	HQEvent & BeginAddEvent() ;
	void EndAddEvent();//discard new event if event queue is disabled
	
	void Lock();//lock access
	void Unlock();//allow other thread to access
private:
	typedef HQLinkedList<HQEvent, HQPoolMemoryManager> EventListType;
	HQMutex m_mutex;
	EventListType m_events;
	bool m_acceptEvent;//does the event queue accept new event? default is true
};

#endif

