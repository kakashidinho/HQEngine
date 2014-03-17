/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_GL_HEADERS_H
#define HQ_GL_HEADERS_H

#include "../HQPlatformDef.h"
/*---------------------------------*/

#if !defined (HQ_MAC_PLATFORM) && !defined (HQ_IPHONE_PLATFORM) && !defined (HQ_ANDROID_PLATFORM)
#include <GL/glew.h>
void GLAPIENTRY DummyProc1(GLenum v);

#ifdef HQ_LINUX_PLATFORM
typedef void (*glxFuncPointer) (void);
typedef glxFuncPointer (*pfnglxgetprocaddress) (const GLubyte* procName);
glxFuncPointer gl_GetProcAddress(const char* procName);
glxFuncPointer DummyProc2(const GLubyte* arg);
#endif


#elif defined (HQ_MAC_PLATFORM) || defined (HQ_IPHONE_PLATFORM)
#import "Apple.h"
#define GLAPIENTRY

#elif defined HQ_ANDROID_PLATFORM
#	ifdef ANDROID_PURE_NATIVE
#		error need implement
#	else
#include "Android.h"
#define GLAPIENTRY
#	endif
#endif

#ifdef WIN32
#include <GL/wglew.h>
#elif defined HQ_LINUX_PLATFORM
#include <GL/glxew.h>
#endif

/*-----------------custom macro----------------------*/

#define _GL_DRAW_BUFFER_USAGE(isDynamic) (isDynamic? GL_DYNAMIC_DRAW : GL_STATIC_DRAW)


#endif
