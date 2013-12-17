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

#include <android/log.h>

#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO, "HQGameThread", __VA_ARGS__)

HQEventQueue* hq_engine_eventQueue_internal;
HQGameThead ge_hqGameThread("HQEngine Game Thread");

extern JavaVM *ge_jvm;
extern JNIEnv * AttachCurrenThreadJEnv();

extern void HQAppInternalOnGameThreadExit();

HQGameThead ::HQGameThead(const char *threadName)
	:HQThread(threadName)
{
}

void HQGameThead :: Run()
{
	//attach this thread to jvm
	JNIEnv *env;
	jint re = ge_jvm->GetEnv((void**)&env, JNI_VERSION_1_2);

	if(re == JNI_EDETACHED) 
	{
		AttachCurrenThreadJEnv();
	}

	//create event queue
	hq_engine_eventQueue_internal = new HQEventQueue();
	
	LOGI("before calling m_entryFunc, m_entryFunc=%p", m_entryFunc);

	m_entryFunc(0, NULL);

	LOGI("after calling m_entryFunc");
	
	delete hq_engine_eventQueue_internal;
	
	HQAppInternalOnGameThreadExit();
}


