#include "../HQUtilPCH.h"
#include "../../HQTimer.h"

#include <thread>
#include <chrono>

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
	std::this_thread::sleep_for( std::chrono::milliseconds((DWORD)(seconds * 1000.0f)));
}