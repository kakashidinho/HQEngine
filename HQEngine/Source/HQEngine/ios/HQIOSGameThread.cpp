/*
 *  HQIOSGameThread.cpp
 *  HQEngine
 *
 *  Created by Kakashidinho on 3/25/12.
 *  Copyright 2012 LHQ. All rights reserved.
 *
 */
#include "../../HQEngineCommon.h"
#include "../HQEventSeparateThread.h"

HQEventQueue* hq_engine_eventQueue_internal;
HQGameThead hq_engine_GameThread_internal("HQEngine Game Thread");

HQGameThead ::HQGameThead(const char *threadName)
	:HQThread(threadName)
{
}

void HQGameThead :: Run()
{
	hq_engine_eventQueue_internal = HQ_NEW HQEventQueue();
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
	
	m_args->re = m_args->entryFunc(m_args->argc, m_args->argv);
	
	[pool release];
	HQ_DELETE (hq_engine_eventQueue_internal);
}


