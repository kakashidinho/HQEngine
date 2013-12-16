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


