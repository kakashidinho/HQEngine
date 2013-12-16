#ifndef HQ_BOUNDING_VOL_H
#define HQ_BOUNDING_VOL_H

#if 0

#include "HQSceneManagementCommon.h"
#include "math/HQ3DMath.h"

class HQBoudingVolume
{
public:
	enum {
		BT_AABB ,
		BT_OBB ,
		BT_SPHERE ,
		BT_UNKNOWN
	};

	virtual ~HQBoudingVolume(){}

	hq_int32 GetVolumeType() {return type}

	virtual bool IsInside(const HQAABB &box) {return false;}
	virtual bool IsInside(const BoudingVolume *volume) {return false;}

	virtual bool IsCollide(const HQAABB &box) {return false;}
	virtual bool IsCollide(const HQOBB &box) {return false;}
	virtual bool IsCollide(const HQSphere &sphere) {return false;}
	virtual bool IsCollide(const HQPlane &plane) {return false;}
	virtual bool IsCollide(const BoudingVolume *volume) {return false;}

	virtual void Transform(const HQMatrix3x4 &transformation) = 0; 
protected:
	hq_in32 type;
};

#endif

#endif