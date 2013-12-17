/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_UTIL_H
#define HQ_UTIL_H

#include "HQPlatformDef.h"
#include "HQPrimitiveDataType.h"

#if (defined HQ_STATIC_ENGINE) || defined IOS
#	define _HQ_UTIL_STATICLINK
#endif

#ifndef HQUTIL_API
#	ifdef _HQ_UTIL_STATICLINK
#		define HQUTIL_API
#	else
#		if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#			ifdef HQUTIL_EXPORTS
#				define HQUTIL_API __declspec(dllexport)
#			else
#				define HQUTIL_API __declspec(dllimport)
#			endif
#		else
#				define HQUTIL_API __attribute__ ((visibility("default")))
#		endif
#	endif
#endif

#include "HQMutex.h"
#include "HQSemaphore.h"
#include "HQThread.h"
#include "HQTimer.h"

/*-----log stream------*/
#include "HQLogStream.h"

HQUTIL_API HQLogStream * HQCreateFileLogStream(const char *filePath);
HQUTIL_API HQLogStream * HQCreateConsoleLogStream();

#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
HQUTIL_API HQLogStream * HQCreateDebugLogStream();
#endif

#ifdef ANDROID
HQUTIL_API HQLogStream * HQCreateLogCatStream();
#endif


#endif
