/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SCENE_NODE_H
#define HQ_SCENE_NODE_H


#include "HQSceneManagementCommon.h"
#include "HQPrimitiveDataType.h"
#include "HQMiscDataType.h"
#include "HQ3DMath.h"
#include "HQMemAlignment.h"


/*-----------HQSceneNode-----------*/

class HQSCENEMANAGEMENT_API HQSceneNode
{
public:
	class ChildIterator{
		friend class HQSceneNode;
	public:
		ChildIterator& operator ++ () ;//prefix addition , jump to next child
		ChildIterator operator ++ (hq_int i); //suffix addition , jump to next child
		bool HasNextChild() {return m_pChild->m_pNextSibling != NULL;}
		bool IsValid() {return m_pChild != NULL;}
		HQSceneNode * operator->() {return m_pChild;}
		HQSceneNode & operator *() {return *m_pChild;}
		HQSceneNode * GetChild() {return m_pChild;}
	private:
		HQSceneNode *m_pChild;
	};
	
	class Listener {
	public:
		virtual void OnUpdated(const HQSceneNode * node, hqfloat32 dt) {} //call when node's transformation has updated
		virtual void OnNodeAddedToParent(const HQSceneNode * node) {} //call when node has added to parent
		virtual void OnNodeRemovedFromParent(const HQSceneNode * node) {} //call when node has removed from parent
	};

	HQSceneNode(const char *name);
	HQSceneNode(const char *name, const HQFloat3 &position , const HQFloat3 &scaleFactors , const HQQuaternion &rotation );
	HQSceneNode(const char *name, 
		hqfloat32 positionX, hqfloat32 positionY, hqfloat32 positionZ , 
		hqfloat32 scaleX, hqfloat32 scaleY, hqfloat32 scaleZ , 
		hqfloat32 rotationX, hqfloat32 rotationY, hqfloat32 rotationZ, hqfloat32 rotationW //quaternion
		);
	
	virtual ~HQSceneNode() ;


	HQSceneNode * GetParent() {return m_pParent;}
	bool HasChild() {return m_pFirstChild != NULL;}
	
	void GetChildIterator(ChildIterator & ite) {ite.m_pChild = m_pFirstChild;}
	
	void AddChild(HQSceneNode * pChild);
	void AddSibling(HQSceneNode * pSibling);
	void RemoveFromParent();
	
	const char *GetName() const {return m_name;}
	void SetName(const char *name);
	///
	///internal use
	///
	char *& GetNamePointer() {return m_name;}

	const HQVector4 & GetLocalPosition() const {return *m_localPosition;}
	const HQFloat3 & GetScaleFactors() const {return m_scale;}
	const HQQuaternion & GetLocalRotation() const {return *m_localRotation;}
	void GetWorldPosition(HQVector4 &positionOut) const ;
	const HQMatrix3x4 & GetWorldTransform() const {return *m_worldTransform;}

	///
	///{pLocalPosition} points to local position of node. 
	///{pScale} points to scale factor of node. 
	///{pLocalRotaion} points to local rotation of node
	///
	inline void WantToChangeLocalTransform(HQVector4 *&pLocalPosition, HQFloat3 *&pScale, HQQuaternion *&pLocalRotaion) {
		pLocalPosition = m_localPosition;
		pScale = &m_scale;
		pLocalRotaion = m_localRotation;

		m_localTransformNeedUpdate = true;
	}

	void SetListener (Listener * listener);

	virtual void SetUniformScale(hqfloat32 factor);
	virtual void SetNonUniformScale(const HQFloat3 &factor);
	virtual void SetPosition(const HQFloat3 &position);//set postion of this node in parent space
	virtual void Translate(const HQFloat3 &offset);//translate this node relative to parent node
	virtual void RotateX(hqfloat32 angle);//rotate around Ox axis of node local space
	virtual void RotateY(hqfloat32 angle);//rotate around Oy axis of node local space
	virtual void RotateZ(hqfloat32 angle);//rotate around Oz axis of node local space
	virtual void RotateAxisUnit(hqfloat32 angle ,const HQVector4& axis);//rotate around arbitrary axis of node local space.<axis> must be unit length vector
	virtual void SetRotation(const HQQuaternion & rotation);

	virtual void Update(hqfloat32 dt ,bool updateChilds = true, bool parentChanged = false );//default : call UpdateWorldTransform() then call UpdateChilds()

protected:
	HQSceneNode();

	void RemoveFromSiblingChain();
	//calculate world transform of this node from parent world transform and this node local transform .
	//return true if this node's trasform state is changed in this method 
	bool UpdateWorldTransform(bool parentChanged);
	void UpdateChilds(hqfloat32 dt , bool updateGrandChilds , bool thisNodeChanged);//call Update(dt , updateGrandChilds , thisNodeChanged) on all of this node's childs

	void UpdateLocalTransform();//calculate local transform matrix from local position , scale factor and , local rotation.This method is called in method UpdateWorldTransform()

	char *m_name;

	HQSceneNode * m_pParent;
	HQSceneNode * m_pFirstChild;
	HQSceneNode * m_pLastChild;
	HQSceneNode * m_pNextSibling;
	HQSceneNode * m_pPrevSibling;

#ifdef WIN32
#	pragma warning(push)
#	pragma warning(disable:4251)
#endif
	
	HQFloat3 m_scale;
	HQVector4 *m_localPosition;
	HQQuaternion *m_localRotation;

	HQMatrix3x4 *m_worldTransform;
	HQMatrix3x4 *m_localTransform;
	
	Listener * m_pListener;

	static Listener m_sDefaultListener;

	bool m_localTransformNeedUpdate;
	bool m_worldTransformNeedUpdate;
#ifdef WIN32
#	pragma warning(pop)
#endif

};
 
/*-------ChildIterator--------------*/
inline HQSceneNode::ChildIterator& HQSceneNode::ChildIterator::operator ++ ()
{
	HQ_ASSERT(m_pChild != NULL);

	m_pChild = m_pChild->m_pNextSibling; 
	return *this;
}

inline HQSceneNode::ChildIterator HQSceneNode::ChildIterator::operator ++ (hq_int i)
{
	HQ_ASSERT(m_pChild != NULL);

	ChildIterator pre = *this ; 
	m_pChild = m_pChild->m_pNextSibling; 
	return pre;
}



#endif
