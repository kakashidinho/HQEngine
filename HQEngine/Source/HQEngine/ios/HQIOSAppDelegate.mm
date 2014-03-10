/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

//
//  HQIOSAppDelegate.mm
//  HQEngine
//
//  Created by Kakashidinho on 3/25/12.
//  Copyright 2012 LHQ. All rights reserved.
//

#import "HQEngineApp.h"
#include "../../HQConditionVariable.h"
#include "../../HQAtomic.h"

#include <libkern/OSAtomic.h>

#define SLEEP_MAIN_THREAD 0
#define USE_MUTEX 0

int hq_internalAppNormalState = 0;
int hq_internalAppExit = 1;
int hq_internalAppIsPaused = 2;

//for blocking game thread
static HQSimpleConditionVar g_pauseGameThreadCnd;
static HQAtomic<bool> g_game_thread_blocked = false;

#if USE_MUTEX
static int g_appStatusFlag = hq_internalAppNormalState;
static HQMutex g_mutex;
#else
static volatile int32_t g_appStatusFlag = hq_internalAppNormalState;
#endif

static void HQAppInternalSetStatusFlag(int flag)
{
#if USE_MUTEX
	HQMutex::ScopeLock sl(g_mutex);
	g_appStatusFlag = flag;
#else
	OSAtomicCompareAndSwap32(g_appStatusFlag, flag, &g_appStatusFlag);
#endif
}


int HQAppInternalGetStatusFlag()
{
#if USE_MUTEX
	HQMutex::ScopeLock sl(g_mutex);
	return g_appStatusFlag;
#else
	return OSAtomicOr32(0x0, (volatile uint32_t*)&g_appStatusFlag);
#endif
}

void HQAppInternalBlockGameLoopIfNeeded()
{
	if (!g_pauseGameThreadCnd.TryLock())
		return;
    
	if (HQAppInternalGetStatusFlag() == hq_internalAppIsPaused)
	{
		g_game_thread_blocked = true;
        
		g_pauseGameThreadCnd.Wait();
        
	}
    
	g_pauseGameThreadCnd.Unlock();
}

static void HQAppInternalWakeGameLoopIfNeededNoLock()
{
	bool shouldWake = false;
    
	shouldWake = g_game_thread_blocked;
	g_game_thread_blocked = false;
    
	if (shouldWake)
		g_pauseGameThreadCnd.Signal();
    
}


static void HQAppInternalWakeGameLoopIfNeeded()
{
	g_pauseGameThreadCnd.Lock();
	
	HQAppInternalWakeGameLoopIfNeededNoLock();
    
	g_pauseGameThreadCnd.Unlock();
    
}

@implementation HQAppDelegate


- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
	[[UIApplication sharedApplication] setStatusBarHidden:YES];//hide status bar
	
#if SLEEP_MAIN_THREAD
	//schedule main thread to sleep
	self->m_timer = [NSTimer scheduledTimerWithTimeInterval:0.01 
													 target:self
												  selector:@selector (sleepMainThreadTimerFire:)
												   userInfo:nil
													repeats:YES];
#endif
	//listen to orientation change
	[[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
	
	//create window and make key
	self->m_window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
	
	[self->m_window makeKeyWindow];

	
	hq_engine_GameThread_internal.Start();
	
	
	return TRUE;
}

- (void)applicationWillResignActive:(UIApplication *)application
{
	HQAppInternalSetStatusFlag(hq_internalAppIsPaused);

	HQEngineApp *pEngineApp = HQEngineApp::GetInstance();
	
	if (pEngineApp != NULL)
	{
		HQEngineAppListener *appListener = pEngineApp->GetAppListener();
		appListener->OnPause();
	}
}

- (void)applicationDidBecomeActive:(UIApplication *)application
{
	HQAppInternalSetStatusFlag(hq_internalAppNormalState);
    HQAppInternalWakeGameLoopIfNeeded();//wake game loop if needed

	HQEngineApp *pEngineApp = HQEngineApp::GetInstance();
	
	if (pEngineApp != NULL)
	{
		HQEngineAppListener *appListener = pEngineApp->GetAppListener();
		appListener->OnResume();
	}
}

- (void)applicationWillTerminate:(UIApplication *)application
{
#if SLEEP_MAIN_THREAD
	[self->m_timer invalidate];
	[self->m_timer release];
#endif
	
	HQAppInternalSetStatusFlag(hq_internalAppExit);
	
	HQEngineApp *pEngineApp = HQEngineApp::GetInstance();
	
	if (pEngineApp != NULL)
	{
		HQEngineAppListener *appListener = pEngineApp->GetAppListener();
		appListener->OnDestroy();
	}
	hq_engine_GameThread_internal.Join();
	
	[[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
}

- (void)applicationDidEnterBackground:(UIApplication *)application
{
    // Handle any background procedures not related to animation here.
}

- (void)applicationWillEnterForeground:(UIApplication *)application
{
    // Handle any foreground procedures not related to animation here.
}

- (void)sleepMainThreadTimerFire:(NSTimer*)theTimer
{
	[self performSelectorOnMainThread:@selector(sleepMainThread) withObject:nil waitUntilDone:YES];
}

-(void)sleepMainThread
{
	HQTimer::Sleep(0.01f);
}

- (void)dealloc {
	[self->m_window release];
    [super dealloc];
}

@end
