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