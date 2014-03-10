/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

// HQDeviceGLPCH.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#ifndef _STD_AFX_RENDER_DEVICE_H
#define _STD_AFX_RENDER_DEVICE_H
#ifdef WIN32

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#endif



// TODO: reference additional headers your program requires here
#include <iostream>
#include <fstream>

#include "../HQPlatformDef.h"
#include "../HQRendererPlatformDef.h"
#include "../HQItemManager.h"
#include "../HQLoggableObject.h"
#include "../HQLinkedList.h"
#include "../HQRendererCoreType.h"
#include "../HQRendererCoreTypeInline.h"
#include "../HQReturnVal.h"
#include "../BaseImpl/HQBaseImplCommon.h"

#endif
