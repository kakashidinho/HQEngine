/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/
#include "HQScenePCH.h"
#include "../HQCamera.h"

HQBasePerspectiveCamera::HQBasePerspectiveCamera(
			hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,//position
			hqfloat32 upX, hqfloat32 upY, hqfloat32 upZ,//up direction
			hqfloat32 directX, hqfloat32 directY, hqfloat32 directZ,//direction
			hqfloat32 fov, //field of view
			hqfloat32 aspect_ratio,//width/height
			hqfloat32 nearPlane, hqfloat32 farPlane, //near and far plane
			HQRenderAPI renderAPI//renderer API (D3D or OpenGL)
		)
		: m_renderAPI(renderAPI)
{
	HQ_DECL_STACK_4VECTOR4( at, pos, direct, up);
	at.Set(directX + posX, directY + posY, directZ + posZ, 1.0f);
	pos.Set(posX, posY, posZ);
	up.Set(upX, upY, upZ);

	//calculate view matrix
	HQMatrix4rLookAtLH(&pos, &at, &up, &this->m_viewMatrix);

	//calculate projection matrix
	HQMatrix4rPerspectiveProjLH(fov, aspect_ratio, nearPlane, farPlane, &this->m_projMatrix, m_renderAPI);

	//calculate view x projection matrix
	HQMatrix4Multiply(&m_viewMatrix, &m_projMatrix, &m_viewProjMat);

	//calculate view frustum
	HQMatrix4rGetFrustum(&m_viewProjMat, m_frustumPlanes, m_renderAPI);
}