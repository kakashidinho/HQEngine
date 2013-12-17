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
