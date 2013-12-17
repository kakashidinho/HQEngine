/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQTimer.h"

void HQTimer :: GetCheckPoint(HQTimeCheckPoint& checkPoint)
{
	QueryPerformanceCounter(&checkPoint);
}
HQTime HQTimer :: GetElapsedTime(const HQTimeCheckPoint& point1 , const HQTimeCheckPoint& point2)
{
	LARGE_INTEGER frequency;
    QueryPerformanceFrequency( &frequency ) ;

	return (HQTime) (double)(point2.QuadPart - point1.QuadPart) / (double) frequency.QuadPart;
}

void HQTimer :: Sleep(HQTime seconds)
{
	::Sleep((DWORD)(seconds * 1000.0f));
}
