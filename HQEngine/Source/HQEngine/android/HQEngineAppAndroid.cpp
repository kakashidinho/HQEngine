/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../../HQEngineApp.h"
#include "../HQEngineWindow.h"
#include "../HQEventSeparateThread.h"
#include "../HQDefaultDataStream.h"

#define LOG_TIME_SPEND_IN_LOCK 0

#if LOG_TIME_SPEND_IN_LOCK
#include <android/log.h>
#endif

extern HQEventQueue* hq_engine_eventQueue_internal;

/*--------HQEngineApp-----------------*/

void HQEngineApp::PlatformInit()
{
}

void HQEngineApp::PlatformRelease()
{
}

//default implementation
HQDataReaderStream* HQEngineApp::OpenFileStream(const char *file)
{
	HQSTDDataReader *stream = HQ_NEW HQSTDDataReader(file);
	if (!stream->Good())
	{
		stream->Release();
		return NULL;
	}

	return stream;
}

bool HQEngineApp::EventHandle()
{
	HQEvent *nextEvent;
#if LOG_TIME_SPEND_IN_LOCK
	HQTime dt;
	HQTimeCheckPoint t1, t2;
	HQTimer::GetCheckPoint(t1);
#endif
	
	hq_engine_eventQueue_internal->Lock();

#if LOG_TIME_SPEND_IN_LOCK
	HQTimer::GetCheckPoint(t2);
	dt = HQTimer::GetElapsedTime(t1 , t2);
#endif	

	bool hasEvent = false;

	while ((nextEvent = hq_engine_eventQueue_internal->GetFirstEvent()) != NULL)
	{

#if LOG_TIME_SPEND_IN_LOCK
		__android_log_print(ANDROID_LOG_DEBUG, "HQEngineApp", "time spend waiting for event queue : %f", dt);
#endif

		HQEvent curEvent = *nextEvent;//copy event
		hq_engine_eventQueue_internal->RemoveFirstEvent();//remove from event queue		
		hq_engine_eventQueue_internal->Unlock();

		switch (curEvent.type) {
			case HQ_TOUCH_BEGAN:
				m_motionListener->TouchBegan(curEvent);
				break;
			case HQ_TOUCH_MOVED:
				m_motionListener->TouchMoved(curEvent);
				break;
			case HQ_TOUCH_ENDED:
				m_motionListener->TouchEnded(curEvent);
				break;
			case HQ_TOUCH_CANCELLED:
				m_motionListener->TouchCancelled(curEvent);
				break;
			case HQ_ORIENTATION_PORTRAIT:
				m_orientListener->ChangedToPortrait();
				break;
			case HQ_ORIENTATION_PORTRAIT_UPSIDE_DOWN:
				m_orientListener->ChangedToPortraitUpsideDown();
				break;
			case HQ_ORIENTATION_LANDSCAPE_LEFT:
				m_orientListener->ChangedToLandscapeLeft();
				break;
			case HQ_ORIENTATION_LANDSCAPE_RIGHT:
				m_orientListener->ChangedToLandscapeRight();
				break;
			default:
				break;
		}
		
		
		hasEvent = true;
	}
	
	hq_engine_eventQueue_internal->Unlock();	

	
	return hasEvent;
}

HQReturnVal HQEngineApp::PlatformEnableMouseCursor(bool enable)
{
	return HQ_FAILED;
}
