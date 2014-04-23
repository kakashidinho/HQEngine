#ifndef BASE_COMMON_H
#define BASE_COMMON_H

#if _MSC_VER >= 1800
#	define __FINAL__ final
#else
#	define __FINAL__
#endif

#include "../../HQEngine/Source/HQ3DMathBasics.h"
/*----------------------------------*/
struct Transform {
	HQBaseMatrix3x4 worldMat;
	HQBaseMatrix4 viewMat;
	HQBaseMatrix4 projMat;
};


#endif