/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#include "HQScenePCH.h"
#include "../HQCamera.h"

#include <math.h>


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
			m_ori_xAxis(HQVector4::New()),
			m_ori_yAxis(HQVector4::New()),
			m_ori_zAxis(HQVector4::New()),
			m_renderAPI(renderAPI)
{
	HQ_DECL_STACK_4VECTOR4( at, pos, direct, up);
	at.Set(directX + posX, directY + posY, directZ + posZ, 1.0f);
	pos.Set(posX, posY, posZ);
	up.Set(upX, upY, upZ);

	//calculate view matrix
	HQMatrix4cLookAtLH(&pos, &at, &up, this->m_viewMatrix);
	m_ori_xAxis->Set(this->m_viewMatrix->_11, this->m_viewMatrix->_12, this->m_viewMatrix->_13, 0.f);
	m_ori_yAxis->Set(this->m_viewMatrix->_21, this->m_viewMatrix->_22, this->m_viewMatrix->_23, 0.f);
	m_ori_zAxis->Set(this->m_viewMatrix->_31, this->m_viewMatrix->_32, this->m_viewMatrix->_33, 0.f);

	//calculate projection matrix
	HQMatrix4cPerspectiveProjLH(fov, aspect_ratio, nearPlane, farPlane, this->m_projMatrix, m_renderAPI);

	//calculate view x projection matrix
	HQMatrix4Multiply(m_projMatrix, m_viewMatrix, m_viewProjMat);

	//calculate view frustum
	HQMatrix4cGetFrustum(m_viewProjMat, m_frustumPlanes, m_renderAPI);
}

HQBasePerspectiveCamera::~HQBasePerspectiveCamera(){
	delete m_viewProjMat;
	delete m_ori_xAxis;
	delete m_ori_yAxis;
	delete m_ori_zAxis;
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
	HQRenderAPI renderAPI,//renderer API (D3D or OpenGL)
	hqfloat32 maxAngle
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
		renderAPI),
	m_horizontal_angle(0.f), m_vertical_angle(0.f),
	m_maxVerticalAngle(maxAngle)
{
	m_alignedMovement.Set(0, 0, 0);
}

HQCamera::~HQCamera(){
}

void HQCamera::Update(hqfloat32 dt ,bool updateChilds, bool parentChanged  )
{
	//base class's method
	HQSceneNode::Update(dt, updateChilds, parentChanged);

	HQ_DECL_STACK_4VECTOR4 (newX, newY, newZ, newPos);
	
	//calculate camera's new axes in world space
	HQVector4TransformNormal(m_ori_xAxis, &this->GetWorldTransform(), &newX);
	HQVector4TransformNormal(m_ori_zAxis, &this->GetWorldTransform(), &newZ);
	newX.Normalize();
	newZ.Normalize();

	HQVector4Cross(&newZ, &newX, &newY);

	//calculate new position
	this->GetWorldPosition(newPos);

	//update matrices and frustum
	//calculate view matrix
	HQMatrix4cView(&newX, &newY, &newZ, &newPos, m_viewMatrix);

	//calculate view x projection matrix
	HQMatrix4Multiply(m_projMatrix, m_viewMatrix, m_viewProjMat);

	//calculate view frustum
	HQMatrix4cGetFrustum(m_viewProjMat, m_frustumPlanes, m_renderAPI);
}

void HQCamera::UpdateLocalTransform()
{
	//base class's method
	HQSceneNode::UpdateLocalTransform();

	/*--------apply camera's roll pitch yaw rotations------------*/
	//ensuring aligned memory
	HQA16ByteStorage<3 * sizeof(HQVector4)+ 2 * sizeof(HQQuaternion) + sizeof(HQMatrix3x4)> aligned_storage; 
	void *pBuffer_aligned_storage = aligned_storage();
	HQVector4 &newX = *(HQVector4::PlNew(pBuffer_aligned_storage));
	HQVector4 &newY = *(HQVector4::PlNew((char*)pBuffer_aligned_storage + sizeof(HQVector4)));
	HQVector4 &newZ = *(HQVector4::PlNew((char*)pBuffer_aligned_storage + 2 * sizeof(HQVector4)));
	HQQuaternion &preVerticalRotation = *(HQQuaternion::PlUINew((char*)pBuffer_aligned_storage + 3 * sizeof(HQVector4)));
	HQQuaternion &preHorizonRotation = *(HQQuaternion::PlUINew((char*)pBuffer_aligned_storage + 3 * sizeof(HQVector4)+sizeof(HQQuaternion)));
	HQMatrix3x4 &preTransform = *(HQMatrix3x4::PlUINew((char*)pBuffer_aligned_storage + 3 * sizeof(HQVector4)+2 * sizeof(HQQuaternion)));

	//clamp the rotated vertical angle
	if (m_maxVerticalAngle != 0.f)
	{
		if (m_vertical_angle > m_maxVerticalAngle)
		{
			m_vertical_angle = m_maxVerticalAngle;
		}
		else if (m_vertical_angle < -m_maxVerticalAngle)
		{
			m_vertical_angle = -m_maxVerticalAngle;
		}
	}

	//rotate around camera's x axis first
	preVerticalRotation.QuatFromRotAxisUnit(-m_vertical_angle, *m_ori_xAxis);

	//calculate camera's new y axis in local space
	HQVector4Transform(m_ori_yAxis, &preVerticalRotation, &newY);
	newY.Normalize();

	//rotate around camera' y axis next
	preHorizonRotation.QuatFromRotAxisUnit(m_horizontal_angle, newY);

	//pre-multiply with local transformation matrix
	preVerticalRotation *= preHorizonRotation;
	preVerticalRotation.QuatToMatrix3x4c(&preTransform);

	HQMatrix3x4Multiply(m_localTransform, &preTransform, m_localTransform);

	//now calculate camera's axes in local space after applying local transformation
	//calculate camera's new axes in world space
	HQVector4TransformNormal(m_ori_xAxis, m_localTransform, &newX);
	HQVector4TransformNormal(m_ori_zAxis, m_localTransform, &newZ);
	newX.Normalize();
	newZ.Normalize();

	HQVector4Cross(&newZ, &newX, &newY);

	//apply aligned movement 
	newX *= m_alignedMovement.x;
	newY *= m_alignedMovement.y;
	newZ *= m_alignedMovement.z;

	*m_localPosition += newX;
	*m_localPosition += newY;
	*m_localPosition += newZ;

	m_localTransform->_14 = m_localPosition->x;
	m_localTransform->_24 = m_localPosition->y;
	m_localTransform->_34 = m_localPosition->z;

	//reset aligned movement per update
	m_alignedMovement.Duplicate(0.f);
}

void HQCamera::RotateVertical(hqfloat32 angle)
{
	m_vertical_angle += angle;
	if (fabs(m_vertical_angle) > HQPiFamily::_2PI)
	{
		m_vertical_angle = fmod(m_vertical_angle, HQPiFamily::_2PI);
	}
}

void HQCamera::RotateHorizontal(hqfloat32 angle)
{
	m_horizontal_angle += angle;
	if (fabs(m_horizontal_angle) > HQPiFamily::_2PI)
	{
		m_horizontal_angle = fmod(m_horizontal_angle, HQPiFamily::_2PI);
	}
}

void HQCamera::MoveLeftRight(float dx)
{
	m_alignedMovement.x += dx;
}

void HQCamera::MoveUpDown(float dx)
{
	m_alignedMovement.y += dx;

}
void HQCamera::MoveBackForward(float dx)
{
	m_alignedMovement.z += dx;
}

void HQCamera::GetLocalDirectionVec(HQVector4& directionOut) const
{
	//get local transformed direction
	HQVector4TransformNormal(m_ori_zAxis, m_localTransform, &directionOut);
	directionOut.Normalize();
}

void HQCamera::GetLocalDirection(HQFloat4& directionOut) const
{
	HQ_DECL_STACK_VECTOR4(tempVec);
	this->GetLocalDirectionVec(tempVec);
	memcpy(&directionOut, &tempVec, sizeof(HQFloat4));
}

void HQCamera::GetWorldDirectionVec(HQVector4& directionOut) const
{
	//get world transformed direction
	HQVector4TransformNormal(m_ori_zAxis, &this->GetWorldTransform(), &directionOut);
	directionOut.Normalize();
}

void HQCamera::GetWorldDirection(HQFloat4& directionOut) const
{
	HQ_DECL_STACK_VECTOR4(tempVec);
	this->GetWorldDirectionVec(tempVec);
	memcpy(&directionOut, &tempVec, sizeof(HQFloat4));
}