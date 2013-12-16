#ifndef HQ_CAMERA_H
#define HQ_CAMERA_H

#include "HQSceneNode.h"


class HQSCENEMANAGEMENT_API HQBaseCamera : public HQA16ByteObject
{
public :
	HQBaseCamera() {}
	virtual ~HQBaseCamera() {} 

	const HQMatrix4x4 * GetViewMatrix() {return &m_viewMatrix;}
	const HQMatrix4x4 * GetProjectionMatrix() {return &m_projMatrix;}
	virtual const HQMatrix4x4 * GetViewProjMatrix() = 0;
	const HQPlane * GetViewFrustum() {return m_frustumPlanes;}

protected:
	HQMatrix4x4 m_viewMatrix;
	HQMatrix4x4 m_projMatrix;
	HQPlane m_frustumPlanes[6];

#ifdef WIN32
};
#else
} HQ_ALIGN16 ;
#endif


#endif