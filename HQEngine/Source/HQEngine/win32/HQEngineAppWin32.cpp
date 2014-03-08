/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../../HQEngineApp.h"
#include "../HQEngineWindow.h"
#include "../HQDefaultDataStream.h"

#include <Shellapi.h>
#include <stdlib.h>
#include <new>

MSG g_msg={0};

struct AppInitOnce
{
	AppInitOnce()
	{
		DisableProcessWindowsGhosting();
	}
};

void HQEngineApp::PlatformInit()
{
	static AppInitOnce initApp;
}

void HQEngineApp::PlatformRelease()
{
}

HQNativeWindow HQEngineApp::GetNativeWindow()
{
	if (m_window == NULL)
		return NULL;

	return m_window->GetRenderDeviceInitInput();
}

bool HQEngineApp::EventHandle()
{
	bool hasEvent = false;
	while( PeekMessage( &g_msg, NULL, 0U, 0U, PM_REMOVE ) )
	{
		TranslateMessage( &g_msg );
		DispatchMessage( &g_msg );

		hasEvent = true;
	}

	return hasEvent;
}

HQReturnVal HQEngineApp::PlatformEnableMouseCursor(bool enable)
{
	/*------------register/unregister raw mouse input------------*/

	RAWINPUTDEVICE rid;
	rid.usUsagePage = 1; 
	rid.usUsage = 2;//mouse 

	if (!enable)//register raw mouse if cursor is disable
	{
		ShowCursor(FALSE);
	        
		rid.dwFlags = RIDEV_NOLEGACY;   //ignore legacy mouse messages
		rid.hwndTarget = m_window->GetRenderDeviceInitInput();

		RECT rect;
		GetWindowRect(m_window->GetRenderDeviceInitInput() , &rect);
		ClipCursor(&rect);//prevent cursor from moving outside window
	}
	else//unregister 
	{
		
		rid.dwFlags = RIDEV_REMOVE; 
		rid.hwndTarget = NULL;

		ClipCursor(NULL);
	}


	if (RegisterRawInputDevices(&rid, 1, sizeof(RAWINPUTDEVICE)) == FALSE) 
		return HQ_FAILED;

	if (enable)
		ShowCursor(TRUE);
	
	return HQ_OK;
}

/*-------------------*/

char ** HQGetCommandLineWrapper(int &argc)
{
	wchar_t**wargv = CommandLineToArgvW(GetCommandLine(), &argc);//get unicode arguments
	if (wargv == NULL)
		return NULL;
	/*------get ANSI arguments-----------*/
	char **argv = NULL;
	try{
		argv = HQ_NEW char*[argc];
		for (int i = 0 ; i < argc; ++i)
		{
			argv[i] = NULL;//initialize all to null
		}
		for (int i = 0 ; i < argc; ++i)
		{
			size_t len = wcstombs(NULL, wargv[i], 0);
			argv[i] = HQ_NEW char[len + 1];
			wcstombs(argv[i], wargv[i], len + 1);
		}

	}
	catch (std::bad_alloc e)
	{
		HQFreeCommandLineArgs(argv, argc);
		LocalFree(wargv);
		return NULL;
	}


	LocalFree(wargv);

	return argv;
}

void HQFreeCommandLineArgs(char ** &argv, int argc)
{
	if (argv == NULL)
		return;
	for (int i = 0 ; i < argc; ++i)
	{
		if (argv[i] != NULL)
		{
			delete[] argv[i];
			argv[i] = NULL;
		}
	}
	delete[] argv;
	argv = NULL;
}
