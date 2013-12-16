#include "../HQUtilPCH.h"
#include "../../HQTimer.h"
#include <math.h>

void HQTimer :: GetCheckPoint(HQTimeCheckPoint& checkPoint)
{
	clock_gettime(CLOCK_MONOTONIC , &checkPoint);
}
HQTime HQTimer :: GetElapsedTime(const HQTimeCheckPoint& point1 , const HQTimeCheckPoint& point2)
{
	
	time_t sec = point2.tv_sec - point1.tv_sec;
	long nsec = point2.tv_nsec - point1.tv_nsec;
	
	return (HQTime)((double)sec + (double)nsec / 1e9);
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