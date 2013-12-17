/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


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


