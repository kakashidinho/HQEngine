#ifndef BASE_COMMON_H
#define BASE_COMMON_H

#if _MSC_VER >= 1800
#	define __FINAL__ final
#else
#	define __FINAL__
#endif

#include "../../HQEngine/Source/HQSharedPointer.h"
#include "../../HQEngine/Source/HQMeshNode.h"

#include "../../HQEngine/Source/HQ3DMathBasics.h"

#include <stdlib.h>

#ifndef min
#	define min(a,b) (((a) > (b)? (b) : (a))
#endif

#ifndef max
#	define max(a,b) (((a) > (b)? (a) : (b))
#endif
/*----------------------------------*/
struct Transform {
	HQBaseMatrix3x4 worldMat;
	HQBaseMatrix4 viewMat;
	HQBaseMatrix4 projMat;
};

struct ModelViewInfo : public Transform{
	HQFloat4 cameraPosition;
};

struct DispatchComputeArgs {
	hquint32 numGroupX;
	hquint32 numGroupY;
	hquint32 numGroupZ;

	void Set(hquint32 nX, hquint32 nY = 1, hquint32 nZ = 1)
	{
		numGroupX = nX;
		numGroupY = nY;
		numGroupZ = nZ;
	}
};

struct Float2 {
	float x, y;
};

//uniform random variable between [a..b]
inline float randf(float a, float b){
	return (rand() / float(RAND_MAX)) * (b - a) + a;
}

inline double randd(double a, double b){
	return (rand() / double(RAND_MAX)) * (b - a) + a;
}

#endif