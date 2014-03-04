/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../../HQEngineApp.h"
#include "../HQEngineWindow.h"
#include "../HQEventSeparateThread.h"
#include "../HQDefaultDataStream.h"

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
	bool hasEvent = false;
	HQEvent *nextEvent;
	hq_engine_eventQueue_internal->Lock();
	while ((nextEvent = hq_engine_eventQueue_internal->GetFirstEvent()) != NULL)
	{
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
