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