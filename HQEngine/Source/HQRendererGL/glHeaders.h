/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_GL_HEADERS_H
#define HQ_GL_HEADERS_H

#include "../HQPlatformDef.h"
/*---------------------------------*/

#if !defined (APPLE) && !defined (IOS) && !defined (ANDROID)
#include <GL/glew.h>
void GLAPIENTRY DummyProc1(GLenum v);

#ifdef LINUX
typedef void (*glxFuncPointer) (void);
typedef glxFuncPointer (*pfnglxgetprocaddress) (const GLubyte* procName);
glxFuncPointer gl_GetProcAddress(const char* procName);
glxFuncPointer DummyProc2(const GLubyte* arg);
#endif


#elif defined (APPLE) || defined (IOS)
#import "Apple.h"
#define GLAPIENTRY

#elif defined ANDROID
#	ifdef ANDROID_PURE_NATIVE
#		error need implement
#	else
#include "Android.h"
#define GLAPIENTRY
#	endif
#endif

#ifdef WIN32
#include <GL/wglew.h>
#elif defined LINUX
#include <GL/glxew.h>
#endif

/*-----------------custom macro----------------------*/

#define _GL_DRAW_BUFFER_USAGE(isDynamic) (isDynamic? GL_DYNAMIC_DRAW : GL_STATIC_DRAW)


#endif
