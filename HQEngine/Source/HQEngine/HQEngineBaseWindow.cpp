/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
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
	FILE *f = fopen(settingFileDir , "r");
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
