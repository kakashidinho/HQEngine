/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_COMMON_H
#define HQ_ENGINE_COMMON_H

#include "HQPlatformDef.h"
#if defined IOS || defined ANDROID || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#	ifndef _STATIC_RENDERER_LIB_
#		define _STATIC_RENDERER_LIB_
#	endif 
#endif

#ifndef HQENGINE_API
#	ifdef HQ_STATIC_ENGINE
#		define HQENGINE_API
#	else
#		if defined WIN32 || defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
#			ifdef HQENGINE_EXPORTS
#				define HQENGINE_API __declspec(dllexport)
#			else
#				define HQENGINE_API __declspec(dllimport)
#			endif
#		else
#				define HQENGINE_API __attribute__ ((visibility("default")))
#		endif
#	endif
#endif

#if defined IOS || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#include "HQThread.h"

struct HQGameThreadArgs
{
	/*--command line args---*/
	int argc;
	char **argv;
	
	/*----gamethread's entry point----*/
	int (* entryFunc ) (int argc, char ** argv);
	int re;//return value of entry point
};

class HQENGINE_API HQGameThead : public HQThread
{
public:
	HQGameThead(const char *threadName);
	
	void Run();
	
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	bool IsRunning();
	bool IsStarted();
#endif

	HQGameThreadArgs* m_args;
private:
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	volatile unsigned long long m_state;
#endif
};

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
extern HQENGINE_API HQGameThead *hq_engine_GameThread_internal;
#else
extern HQGameThead hq_engine_GameThread_internal;
#endif

#endif//if defined IOS || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)//win32 desktop
/*-------WinMain wrapper's helper functions---------*/

#include <windows.h>
HQENGINE_API char ** HQGetCommandLineWrapper(int &argCount);
HQENGINE_API void HQFreeCommandLineArgs(char **&args, int argCount);

#elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)//winRT
/*-------helper functions ---*/

HQENGINE_API char ** HQWinStoreGetCmdLineWrapper(Platform::Array<Platform::String^>^refArgs, int &argCount);
HQENGINE_API void HQWinStoreFreeCmdLineArgs(char **&args, int argCount);

HQENGINE_API Windows::ApplicationModel::Core::IFrameworkViewSource ^ HQWinStoreCreateFWViewSource();

#elif defined IOS/*----IOS------*/

#import <UIKit/UIKit.h>

///
//app delegate
//
@interface HQAppDelegate : NSObject <UIApplicationDelegate> {
@private
	NSTimer *m_timer;
	UIWindow *m_window;
}


@end

/*--------Android--------------*/
#elif defined ANDROID
#include "HQThread.h"

#include <jni.h>

class HQENGINE_API HQGameThead : public HQThread
{
public:
	HQGameThead(const char *threadName);
	
	void Run();
	
	int (* m_entryFunc ) (int argc, char ** argv);
};

extern HQGameThead ge_hqGameThread HQENGINE_API;


#endif//#ifdef WIN32

#ifndef HQEngineMain
#	if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)//windows desktop
#		define HQEngineMain(argc , argv) \
			HQEngineMainWrapper(argc, argv);\
			int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)\
			{\
				int argCount;\
				char ** args = HQGetCommandLineWrapper(argCount);\
				if (args == NULL)\
					return -1;\
				int re = HQEngineMainWrapper(argCount, args);\
				HQFreeCommandLineArgs(args, argCount);\
				return re;\
			}\
			int HQEngineMainWrapper(argc, argv)

#	elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)//winRT
	#		define HQEngineMain(dargc,dargv) \
			HQEngineMainWrapper(dargc, dargv);\
			[Platform::MTAThread]\
			int main(Platform::Array<Platform::String^>^ refArgs)\
			{\
				int argCount;\
				char ** args = HQWinStoreGetCmdLineWrapper(refArgs, argCount);\
				if (args == NULL)\
					return -1;\
				HQGameThreadArgs threadArgs ;\
				threadArgs.argc = argCount;\
				threadArgs.argv = args;\
				threadArgs.entryFunc = HQEngineMainWrapper;\
				\
				hq_engine_GameThread_internal = HQ_NEW HQGameThead("HQEngine Game Thread");\
				hq_engine_GameThread_internal->m_args = &threadArgs;\
				\
				auto frameworkViewSource = HQWinStoreCreateFWViewSource();\
				Windows::ApplicationModel::Core::CoreApplication::Run(frameworkViewSource);\
				\
				hq_engine_GameThread_internal->Join();\
				\
				HQWinStoreFreeCmdLineArgs(args, argCount);\
				HQ_DELETE (hq_engine_GameThread_internal);\
				\
				return threadArgs.re;\
			}\
			int HQEngineMainWrapper(dargc, dargv)

#	elif defined LINUX || defined APPLE
#		define HQEngineMain main

#	elif defined IOS
#		define HQEngineMain(dargc,dargv) \
			HQEngineMainWrapper(dargc, dargv);\
			int main(int _argc, char **_argv)\
			{\
				HQGameThreadArgs args ;\
				args.argc = _argc;\
				args.argv = _argv;\
				args.entryFunc = HQEngineMainWrapper;\
				\
				hq_engine_GameThread_internal.m_args = &args;\
				\
				NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];\
				UIApplicationMain(_argc, _argv, @"UIApplication", NSStringFromClass([HQAppDelegate class]));\
				\
				hq_engine_GameThread_internal.Join();\
				[pool release];\
				\
				return args.re;\
			}\
			int HQEngineMainWrapper(dargc, dargv)

#	elif defined ANDROID
#		define HQEngineMain(dargc,dargv) \
			HQEngineMainWrapper(dargc, dargv);\
			extern "C"\
			{\
				JNIEXPORT void JNICALL Java_hqengine_java_HQEngineBaseActivity_onCreateNative(JNIEnv *env, jobject jactivity)\
				{\
					ge_hqGameThread.m_entryFunc = &HQEngineMainWrapper;\
				}\
			}\
			int HQEngineMainWrapper(dargc, dargv)

#	else
#		error need implement
#	endif
#endif

#endif
