/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "HQEngineWindow.h"
#include <stdio.h>
#define __DEFAULT_WIDTH 600
#define __DEFAULT_HEIGHT 600
#define __DEFAULT_WINDOWED true


HQEngineBaseWindow::HQEngineBaseWindow(const char* settingFileDir)
: m_width(__DEFAULT_WIDTH) ,
 m_height(__DEFAULT_HEIGHT),
 m_windowed (__DEFAULT_WINDOWED)
{
	FILE *f = settingFileDir != NULL? fopen(settingFileDir , "r") : NULL;
	if (f != NULL)
	{
		fscanf(f, " Basic Settings");
		fscanf(f, " Width=%u", &m_width);
		fscanf(f, " Height=%u", &m_height);
		int windowed;
		fscanf(f, " Windowed=%d", &windowed);
		m_windowed = windowed != 0;
		fclose(f);
	}
}
