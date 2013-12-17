/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ASSERT_H
#define HQ_ASSERT_H
#include <assert.h>
#include "HQPlatformDef.h"

#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#include <crtdbg.h>
#include <windows.h>
#endif

#if	defined ANDROID
#include <android/log.h>
#endif

#ifndef HQ_ASSERT
#	if defined DEBUG || defined _DEBUG
#		if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#			define HQ_ASSERT(e) {_ASSERTE(e);}
#			define HQ_ASSERT_SOURCE_LINE(e, source, line) { _ASSERTE(e);}
#		else
#			if	defined ANDROID
#				define HQ_ASSERT_SOURCE_LINE(e, source, line) { if (!(e)) { __android_log_print(ANDROID_LOG_ERROR, "HQ_ASSERT_SOURCE_LINE", "failed in %s:%d", source, line ); assert(e);}}
#				define HQ_ASSERT(e) (HQ_ASSERT_SOURCE_LINE(e, __FILE__, __LINE__))
#			else
#				define HQ_ASSERT_SOURCE_LINE(e, source, line) { assert(e);}
#				define HQ_ASSERT(e) { assert(e);}
#			endif
#		endif
#	else
#		define HQ_ASSERT(e)
#		define HQ_ASSERT_SOURCE_LINE(e, source, line)
#	endif
#endif


#endif
