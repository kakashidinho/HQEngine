/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../../HQEngineApp.h"
#include "../HQEngineWindow.h"
#include "../HQDefaultDataStream.h"

static XEvent g_msg={0};

//window event handler
extern void HQEngineWndEventHandler(XEvent * event);

struct AppInitOnce
{
	AppInitOnce()
	{
		
	}
};

void HQEngineApp::PlatformInit()
{
	static AppInitOnce initApp;
}

void HQEngineApp::PlatformRelease()
{
}


bool HQEngineApp::EventHandle()
{
	bool hasEvent = false;
	while( this->IsRunning() && XPending( m_window->GetDisplay()) )
	{
		XNextEvent(m_window->GetDisplay(), &g_msg);

		if (g_msg.xany.window == m_window->GetRawWindow())
			HQEngineWndEventHandler(&g_msg);

		hasEvent = true;
	}

	return hasEvent;
}

HQReturnVal HQEngineApp::PlatformEnableMouseCursor(bool enable)
{

	if (m_window == NULL || !m_window->EnableCursor(enable))
		return HQ_FAILED;
	
	return HQ_OK;
}

