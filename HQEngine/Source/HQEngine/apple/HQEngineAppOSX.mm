/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../stdafx.h"
#include "../../HQEngineApp.h"
#include "../HQEngineWindow.h"
#include "../HQDefaultDataStream.h"

/*-------------------------------------------------------------------------
AppInitOnce - init stuffs only at the first time app instance is created
---------------*/
struct AppInitOnce
{
	AppInitOnce()
	{
		[NSApplication sharedApplication];
		m_pool = [[NSAutoreleasePool alloc] init];
		[NSApp finishLaunching];
	}
	~AppInitOnce()
	{
		
		[m_pool release];
	}
	
	NSAutoreleasePool * m_pool;
};



/*--------HQEngineApp-----------------*/

void HQEngineApp::PlatformInit()
{
	static AppInitOnce appInit;
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
	NSEvent *event;
	while ((event = [NSApp nextEventMatchingMask:NSAnyEventMask untilDate:nil inMode:NSDefaultRunLoopMode dequeue:YES]) != nil)
	{
		[NSApp sendEvent: event];
		//[event release];
		hasEvent = true;
	}
	
	
	return hasEvent;
}

HQReturnVal HQEngineApp::PlatformEnableMouseCursor(bool enable)
{
	if (enable)
	{
		if (CGDisplayShowCursor (kCGDirectMainDisplay) != kCGErrorSuccess)
			return HQ_FAILED;
		if (CGAssociateMouseAndMouseCursorPosition (true) != kCGErrorSuccess)
		{
			CGDisplayHideCursor (kCGDirectMainDisplay);
			return HQ_FAILED;
		}
	}
	else
	{
		if (CGDisplayHideCursor (kCGDirectMainDisplay) != kCGErrorSuccess)
			return HQ_FAILED;
		
		if (CGAssociateMouseAndMouseCursorPosition (false) != kCGErrorSuccess)
		{
			CGDisplayShowCursor (kCGDirectMainDisplay);
			return HQ_FAILED;
		}
	}
	
	return HQ_OK;
}
