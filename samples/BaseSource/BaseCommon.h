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

#endif