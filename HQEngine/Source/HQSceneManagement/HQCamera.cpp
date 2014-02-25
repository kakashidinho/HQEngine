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


/*-----base camera--------*/
HQBaseCamera::HQBaseCamera()
	:m_viewMatrix(HQMatrix4::New()),
	m_projMatrix(HQMatrix4::New()),
	m_frustumPlanes(HQPlane::NewArray(6))
{
}

HQBaseCamera::~HQBaseCamera(){
	delete m_viewMatrix;
	delete m_projMatrix;
	delete[] m_frustumPlanes;
}

/*------base perspective camera-----*/
HQBasePerspectiveCamera::HQBasePerspectiveCamera(
			hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,//position
			hqfloat32 upX, hqfloat32 upY, hqfloat32 upZ,//up direction
			hqfloat32 directX, hqfloat32 directY, hqfloat32 directZ,//direction
			hqfloat32 fov, //field of view
			hqfloat32 aspect_ratio,//width/height
			hqfloat32 nearPlane, hqfloat32 farPlane, //near and far plane
			HQRenderAPI renderAPI//renderer API (D3D or OpenGL)
		)
		:	m_viewProjMat(HQMatrix4::New()),
			m_xAxis(HQVector4::New()),
			m_yAxis(HQVector4::New()),
			m_zAxis(HQVector4::New()),
			m_renderAPI(renderAPI)
{
	HQ_DECL_STACK_4VECTOR4( at, pos, direct, up);
	at.Set(directX + posX, directY + posY, directZ + posZ, 1.0f);
	pos.Set(posX, posY, posZ);
	up.Set(upX, upY, upZ);

	//calculate view matrix
	HQMatrix4rLookAtLH(&pos, &at, &up, this->m_viewMatrix);
	m_xAxis->Set(this->m_viewMatrix->_11, this->m_viewMatrix->_21, this->m_viewMatrix->_31, 0.f);
	m_yAxis->Set(this->m_viewMatrix->_12, this->m_viewMatrix->_22, this->m_viewMatrix->_32, 0.f);
	m_zAxis->Set(this->m_viewMatrix->_13, this->m_viewMatrix->_23, this->m_viewMatrix->_33, 0.f);

	//calculate projection matrix
	HQMatrix4rPerspectiveProjLH(fov, aspect_ratio, nearPlane, farPlane, this->m_projMatrix, m_renderAPI);

	//calculate view x projection matrix
	HQMatrix4Multiply(m_viewMatrix, m_projMatrix, m_viewProjMat);

	//calculate view frustum
	HQMatrix4rGetFrustum(m_viewProjMat, m_frustumPlanes, m_renderAPI);
}

HQBasePerspectiveCamera::~HQBasePerspectiveCamera(){
	delete m_viewProjMat;
	delete m_xAxis;
	delete m_yAxis;
	delete m_zAxis;
}

/*---------HQCamera---------*/
HQCamera::HQCamera(
		    const char* name,
			hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,//position
			hqfloat32 upX, hqfloat32 upY, hqfloat32 upZ,//up direction
			hqfloat32 directX, hqfloat32 directY, hqfloat32 directZ,//direction
			hqfloat32 fov, //field of view
			hqfloat32 aspect_ratio,//width/height
			hqfloat32 nearPlane, hqfloat32 farPlane, //near and far plane
			HQRenderAPI renderAPI//renderer API (D3D or OpenGL)
		)
		: HQSceneNode(name,
					posX, posY, posZ,
					1, 1, 1,
					0, 0, 0, 1
					),
		  HQBasePerspectiveCamera(posX, posY, posZ,
								  upX, upY, upZ,
								  directX, directY, directZ,
								  fov, aspect_ratio, nearPlane, farPlane,
								  renderAPI)
{
}

HQCamera::~HQCamera(){
}

void HQCamera::Update(hqfloat32 dt ,bool updateChilds, bool parentChanged  )
{
	//base class's method
	HQSceneNode::Update(dt, updateChilds, parentChanged);

	HQ_DECL_STACK_4VECTOR4 (newX, newY, newZ, newPos);
	//calculate new local axes
	HQVector4TransformNormal(m_xAxis, &this->GetWorldTransform(), &newX);
	HQVector4TransformNormal(m_yAxis, &this->GetWorldTransform(), &newY);
	HQVector4TransformNormal(m_zAxis, &this->GetWorldTransform(), &newZ);
	newX.Normalize();
	newY.Normalize();
	newZ.Normalize();

	//calculate new position
	this->GetWorldPosition(newPos);

	//update matrices and frustum
	//calculate view matrix
	HQMatrix4rView(&newX, &newY, &newZ, &newPos, m_viewMatrix);

	//calculate view x projection matrix
	HQMatrix4Multiply(m_viewMatrix, m_projMatrix, m_viewProjMat);

	//calculate view frustum
	HQMatrix4rGetFrustum(m_viewProjMat, m_frustumPlanes, m_renderAPI);
}