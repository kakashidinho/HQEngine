/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQScenePCH.h"
#include "../HQSceneNode.h"

#ifdef ANDROID
#	include <android/log.h>
#	define TRACE(...)
//#	define TRACE(...) __android_log_print(ANDROID_LOG_DEBUG, "HQSceneNode", __VA_ARGS__)
#else
#	define TRACE(...)
#endif

//TO DO
//handle non uniform scale

/*----------HQSceneNode-------------*/
/*--------static attribute-----------------*/
HQSceneNode::Listener  HQSceneNode::m_sDefaultListener;

/*---------methods------------------------*/

HQSceneNode::HQSceneNode()
: m_localTransformNeedUpdate(false) , m_worldTransformNeedUpdate(false) ,
 m_pParent(NULL) , m_pFirstChild(NULL) , m_pLastChild(NULL) , 
 m_pNextSibling(NULL) , m_pPrevSibling(NULL), 
 m_name(NULL),

 m_localPosition(HQVector4::New()),
 m_localRotation(HQQuaternion :: New()),
 m_worldTransform(HQMatrix3x4 :: New()),
 m_localTransform(HQMatrix3x4 :: New())
{
	m_localPosition->Set(0.0f , 0.0f , 0.0f);
	m_scale.Set(1.0f , 1.0f , 1.0f);
	m_pListener = &m_sDefaultListener;//default listener
}

HQSceneNode::HQSceneNode(const char *name)
: m_localTransformNeedUpdate(false) , m_worldTransformNeedUpdate(false) ,
 m_pParent(NULL) , m_pFirstChild(NULL) , m_pLastChild(NULL) , 
 m_pNextSibling(NULL) , m_pPrevSibling(NULL),

 m_localPosition(HQVector4::New()),
 m_localRotation(HQQuaternion :: New()),
 m_worldTransform(HQMatrix3x4 :: New()),
 m_localTransform(HQMatrix3x4 :: New())
{
	m_localPosition->Set(0.0f , 0.0f , 0.0f);
	m_scale.Set(1.0f , 1.0f , 1.0f);

	m_pListener = &m_sDefaultListener;//default listener

	size_t len = strlen(name);
	m_name = HQ_NEW char[len + 1];
	strcpy(m_name, name);
}


HQSceneNode::HQSceneNode(const char *name, const HQFloat3 &position , const HQFloat3 &scaleFactors , const HQQuaternion &rotation )
:  m_scale(scaleFactors) , 
 m_localTransformNeedUpdate(false) , m_worldTransformNeedUpdate(false) ,
 m_pParent(NULL) , m_pFirstChild(NULL) , m_pLastChild(NULL) , 
 m_pNextSibling(NULL) , m_pPrevSibling(NULL),

 m_localPosition(HQVector4::New(position.x, position.y, position.z)),
 m_localRotation(HQQuaternion :: New(rotation)),
 m_worldTransform(HQMatrix3x4 :: New()),
 m_localTransform(HQMatrix3x4 :: New())
{
	m_pListener = &m_sDefaultListener;//default listener

	size_t len = strlen(name);
	m_name = HQ_NEW char[len + 1];
	strcpy(m_name, name);
}

HQSceneNode::HQSceneNode(const char *name, 
		hqfloat32 positionX, hqfloat32 positionY, hqfloat32 positionZ , 
		hqfloat32 scaleX, hqfloat32 scaleY, hqfloat32 scaleZ , 
		hqfloat32 rotationX, hqfloat32 rotationY, hqfloat32 rotationZ, hqfloat32 rotationW //quaternion
		)
:
 m_localTransformNeedUpdate(false) , m_worldTransformNeedUpdate(false) ,
 m_pParent(NULL) , m_pFirstChild(NULL) , m_pLastChild(NULL) , 
 m_pNextSibling(NULL) , m_pPrevSibling(NULL),

 m_localPosition(HQVector4::New(positionX, positionY, positionZ)),
 m_localRotation(HQQuaternion :: New(rotationX, rotationY, rotationZ, rotationW)),
 m_worldTransform(HQMatrix3x4 :: New()),
 m_localTransform(HQMatrix3x4 :: New())
{
	m_scale.Set(scaleX, scaleY, scaleZ);

	m_pListener = &m_sDefaultListener;//default listener

	size_t len = strlen(name);
	m_name = HQ_NEW char[len + 1];
	strcpy(m_name, name);
}


HQSceneNode::~HQSceneNode()
{
	delete[] m_name;

	RemoveFromParent();

	delete m_localPosition;
	delete m_localRotation;
	delete m_worldTransform;
	delete m_localTransform;
}

void HQSceneNode::AddChild(HQSceneNode *pNode)
{
	pNode->RemoveFromSiblingChain();

	pNode->m_pParent = this;
	pNode->m_pNextSibling = NULL;
	pNode->m_pPrevSibling = m_pLastChild;

	pNode->m_worldTransformNeedUpdate = true;

	if (m_pLastChild == NULL)
	{
		m_pFirstChild = m_pLastChild = pNode;
	}
	else
	{
		m_pLastChild->m_pNextSibling = pNode;
		m_pLastChild = pNode;
	}

	m_pListener->OnNodeAddedToParent(this);

}

void HQSceneNode::AddSibling(HQSceneNode * pSibling)
{
	HQSceneNode *curNextSibling = this->m_pNextSibling;
	
	
	if (m_pParent != NULL)
		m_pParent->AddChild(pSibling);
	else
	{
		pSibling->RemoveFromSiblingChain();

		pSibling->m_pParent = this->m_pParent;

		this->m_pNextSibling = pSibling;
		pSibling->m_pPrevSibling = this;
		pSibling->m_pNextSibling = curNextSibling;

		if (curNextSibling != NULL)
			curNextSibling->m_pPrevSibling = pSibling;
	}
}

void HQSceneNode::RemoveFromParent()
{
	this->RemoveFromSiblingChain();

	this->m_pParent = NULL;
	this->m_pNextSibling = NULL;
	this->m_pPrevSibling = NULL;

	m_pListener->OnNodeRemovedFromParent(this);
}

void HQSceneNode::RemoveFromSiblingChain()
{
	if (this->m_pPrevSibling != NULL)
		this->m_pPrevSibling->m_pNextSibling = this->m_pNextSibling;
	else if (this->m_pParent != NULL)//this node is the first child of its parent
		this->m_pParent->m_pFirstChild = this->m_pNextSibling;

	if (this->m_pNextSibling != NULL)
		this->m_pNextSibling->m_pPrevSibling = this->m_pPrevSibling;
}


void HQSceneNode::SetName(const char *name)
{
	delete[] m_name;
	
	size_t len = strlen(name);
	m_name = HQ_NEW char[len + 1];
	strcpy(m_name, name);
}

void HQSceneNode::SetUniformScale(float factor)
{
	m_localTransformNeedUpdate = true;
	m_scale.Duplicate(factor);
}
void HQSceneNode::SetNonUniformScale(const HQFloat3 &factor)
{
	m_localTransformNeedUpdate = true;
	m_scale = factor;
}
void HQSceneNode::SetPosition(const HQFloat3 &position)//set postion of this node in parent space
{
	m_localTransformNeedUpdate = true;
	m_localPosition->Set( position.x, position.y, position.z);
}
void HQSceneNode::Translate(const HQFloat3 &offset)//translate this node relative to parent node
{
	m_localTransformNeedUpdate = true;
	m_localPosition->x += offset.x;
	m_localPosition->y += offset.y;
	m_localPosition->z += offset.z;
}
void HQSceneNode::RotateX(float angle)//rotate around Ox axis of node local space
{
	m_localTransformNeedUpdate = true;
	HQ_DECL_STACK_QUATERNION_CTOR_PARAMS( newRotation, (NULL));
	newRotation.QuatFromRotAxisOx(angle);

	*m_localRotation *= newRotation;
}
void HQSceneNode::RotateY(float angle)//rotate around Oy axis of node local space
{
	m_localTransformNeedUpdate = true;
	HQ_DECL_STACK_QUATERNION_CTOR_PARAMS( newRotation, (NULL));
	newRotation.QuatFromRotAxisOy(angle);

	*m_localRotation *= newRotation;
}
void HQSceneNode::RotateZ(float angle)//rotate around Oz axis of node local space
{
	m_localTransformNeedUpdate = true;
	HQ_DECL_STACK_QUATERNION_CTOR_PARAMS( newRotation, (NULL));
	newRotation.QuatFromRotAxisOz(angle);

	*m_localRotation *= newRotation;
}

void HQSceneNode::RotateAxisUnit(float angle ,const HQVector4& axis)//rotate around arbitrary axis of node local space.<axis> must be unit length vector
{
	m_localTransformNeedUpdate = true;
	HQ_DECL_STACK_QUATERNION_CTOR_PARAMS( newRotation, (NULL));
	newRotation.QuatFromRotAxisUnit(angle , axis);

	*m_localRotation *= newRotation;
}

void HQSceneNode::SetRotation(const HQQuaternion & rotation)
{
	m_localTransformNeedUpdate = true;
	
	*m_localRotation = rotation;
}

void HQSceneNode::SetListener (HQSceneNode::Listener * listener)
{
	if (listener == NULL)
		m_pListener = &m_sDefaultListener;
	else
		m_pListener = listener;
}

void HQSceneNode::Update(hqfloat32 dt , bool updateChilds , bool parentChanged)
{
	bool thisNodeChanged = this->UpdateWorldTransform(parentChanged);

	m_pListener->OnUpdated(this, dt);//delegate

	if (updateChilds)
	{
		this->UpdateChilds(dt , updateChilds , thisNodeChanged);
	}
}

bool HQSceneNode::UpdateWorldTransform( bool parentChanged)
{
	TRACE("here %s %d", __FILE__, __LINE__);

	if (m_localTransformNeedUpdate)
	{
		this->UpdateLocalTransform();
		if (this->m_pParent != NULL)//multiply with parent matrix
		{
			HQMatrix3x4Multiply(m_pParent->m_worldTransform , m_localTransform , m_worldTransform);
		}
		else
		{
			*m_worldTransform = *m_localTransform;
		}

		TRACE("here %s %d", __FILE__, __LINE__);

		m_localTransformNeedUpdate = false;

		return true;
	}
	else if (parentChanged)//when parent has updated its transform state but its childs haven't
	{
		if (this->m_pParent != NULL)//multiply with parent matrix
		{
			HQMatrix3x4Multiply(m_pParent->m_worldTransform , m_localTransform , m_worldTransform);
		}
		
		TRACE("here %s %d", __FILE__, __LINE__);

		return true;
	}
	else if (m_worldTransformNeedUpdate)//when this node has changed parent
	{
		if (this->m_pParent != NULL)//multiply with parent matrix
		{
			HQMatrix3x4Multiply(m_pParent->m_worldTransform , m_localTransform , m_worldTransform);
		}

		m_worldTransformNeedUpdate = false;
		
		TRACE("here %s %d", __FILE__, __LINE__);

		return true;
	}

	TRACE("here %s %d", __FILE__, __LINE__);

	return false;
}
void HQSceneNode::UpdateChilds(float dt , bool updateGrandChilds , bool thisNodeChanged)
{
	HQSceneNode::ChildIterator childIte; 
	this->GetChildIterator(childIte);
	while(childIte.IsValid())
	{
		childIte->Update(dt , updateGrandChilds , thisNodeChanged);

		++childIte;
	}
}


void HQSceneNode::UpdateLocalTransform()
{
	TRACE("here %s %d", __FILE__, __LINE__);

	HQMatrix3x4Scale(m_scale.f , m_localTransform);

	TRACE("here %s %d", __FILE__, __LINE__);

	HQ_DECL_STACK_MATRIX3X4_CTOR_PARAMS( rotation , (NULL));
	m_localRotation->QuatToMatrix3x4c(&rotation);
	(*m_localTransform) *= rotation;

	TRACE("here %s %d", __FILE__, __LINE__);

	m_localTransform->_14 = m_localPosition->x;
	m_localTransform->_24 = m_localPosition->y;
	m_localTransform->_34 = m_localPosition->z;
	
}


void HQSceneNode::GetWorldPosition(HQVector4 &positionOut) const
{
	if (this->m_pParent != NULL)
	{
		HQVector4TransformNormal(m_localPosition, &m_pParent->GetWorldTransform(), &positionOut);
	}
	else
		memcpy(&positionOut, m_localPosition,sizeof(HQVector4));
}
