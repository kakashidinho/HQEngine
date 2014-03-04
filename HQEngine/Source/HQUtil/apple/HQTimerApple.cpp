/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQUtilPCH.h"
#include "../../HQTimer.h"
#include <math.h>

void HQTimer :: GetCheckPoint(HQTimeCheckPoint& checkPoint)
{
	checkPoint = mach_absolute_time();
}
HQTime HQTimer :: GetElapsedTime(const HQTimeCheckPoint& point1 , const HQTimeCheckPoint& point2)
{
	mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
	
	double time = (double) (point2 - point1) * (double)timebase.numer / 
					(double) timebase.denom / 1e9;

	return (HQTime) time;
}

void HQTimer :: Sleep(HQTime seconds)
{
	timespec sleepTime;
	HQTime intSeconds ;
	HQTime fracSecond = modff(seconds, &intSeconds);
	sleepTime.tv_sec = (time_t)intSeconds;
	sleepTime.tv_nsec = fracSecond * 1e9;
	
	nanosleep(&sleepTime, NULL);
}
