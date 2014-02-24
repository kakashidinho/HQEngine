/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_CAMERA_H
#define HQ_CAMERA_H

#include "HQSceneNode.h"


class HQSCENEMANAGEMENT_API HQBaseCamera : public HQA16ByteObject
{
public :
	HQBaseCamera() {}
	virtual ~HQBaseCamera() {} 

	const HQMatrix4 * GetViewMatrix() const {return &m_viewMatrix;}
	const HQMatrix4 * GetProjectionMatrix() const {return &m_projMatrix;}
	virtual const HQMatrix4 * GetViewProjMatrix() const = 0;
	const HQPlane * GetViewFrustum() {return m_frustumPlanes;}

protected:
	HQMatrix4 m_viewMatrix;
	HQMatrix4 m_projMatrix;
	HQPlane m_frustumPlanes[6];//view frustum

#ifdef WIN32
};
#else
} HQ_ALIGN16 ;
#endif


//base perspective camera. Note: the matrices will be Left-Hand and row-major based
class HQSCENEMANAGEMENT_API HQBasePerspectiveCamera : public HQBaseCamera
{
public :
	HQBasePerspectiveCamera(
			hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,//position
			hqfloat32 upX, hqfloat32 upY, hqfloat32 upZ,//up direction
			hqfloat32 directX, hqfloat32 directY, hqfloat32 directZ,//direction
			hqfloat32 fov, //field of view
			hqfloat32 aspect_ratio,//width/height,
			hqfloat32 nearPlane, hqfloat32 farPlane, //near and far plane
			HQRenderAPI renderAPI//renderer API (D3D or OpenGL)
		);
	
	//get multiplcation of view & projection matrix
	const HQMatrix4 * GetViewProjMatrix() const { return &m_viewProjMat;}

private:
	HQMatrix4 m_viewProjMat;//view x projection matrix
	HQRenderAPI m_renderAPI;

#ifdef WIN32
};
#else
} HQ_ALIGN16 ;
#endif

#endif
