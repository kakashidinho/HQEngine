/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_MESH_INTERNAL_H
#define HQ_MESH_INTERNAL_H

#include "../HQReferenceCountObj.h"
#include "../HQSceneNode.h"
#include "../HQMeshNode.h"

/*-----------Geometric Info------------*/
struct HQGeometricGroup
{
	hquint32 startIndex;//start vertex index
	hquint32 numIndices;//number of vertex indices 

	HQSubMeshInfo material;
};

struct HQMeshNode::GeometricInfo : public HQReferenceCountObj
{
	GeometricInfo() :
		groups(NULL) 
	{
	}
	~GeometricInfo() {
		delete[] groups;
	}

	hquint32 vertexInputLayout;
	hquint32 vertexBuffer;
	hquint32 numVertices;
	hquint32 vertexBufferStride;
	hquint32 indexBuffer;
	HQIndexDataType indexDataType;

	hquint32 numGroups;
	
	HQGeometricGroup * groups;
};

/*------------Animation Info-----------*/


class HQAnimationSceneNode : public HQSceneNode
{
public:
	HQAnimationSceneNode() : HQSceneNode() {}
	~HQAnimationSceneNode() {}
};

struct HQAnimationBone
{
	HQMatrix3x4 boneOffsetMatrices;

	HQSceneNode *node;
};

//each key set is for one node
struct HQAnimationKeySet
{
	HQAnimationKeySet() 
		:numTransKeys (0), numScaleKeys(0), numRotKeys(0), 
		transKeyTimes(NULL), transKeys(NULL), fixTransKeyPeriod(0.0f),
		scaleKeyTimes(NULL), scaleKeys(NULL), fixScaleKeyPeriod(0.0f),
		rotKeyTimes(NULL), rotKeys(NULL), fixRotKeyPeriod(0.0f)
	{
	}

	~HQAnimationKeySet()
	{
		delete[] transKeyTimes;
		delete[] transKeys;
		
		delete[] scaleKeyTimes;
		delete[] scaleKeys;
		
		delete[] rotKeyTimes;
		delete[] rotKeys;
	}

	void AllocTransKeys(hquint32 numKeys);
	void AllocScaleKeys(hquint32 numKeys);
	void AllocRotKeys(hquint32 numKeys);

	inline hqfloat32 *GetTransKeyTimePtr(hquint32 keyIndex) {return &transKeyTimes[keyIndex];} 
	inline hqfloat32 GetTransKeyTime(hquint32 keyIndex) const {return transKeyTimes[keyIndex];} 
	inline HQFloat3 &GetTransKey(hquint32 keyIndex) {return transKeys[keyIndex];} 
	inline const HQFloat3 &GetTransKey(hquint32 keyIndex) const {return transKeys[keyIndex];} 

	inline hqfloat32 *GetScaleKeyTimePtr(hquint32 keyIndex) {return &scaleKeyTimes[keyIndex];} 
	inline hqfloat32 GetScaleKeyTime(hquint32 keyIndex) const {return scaleKeyTimes[keyIndex];} 
	inline HQFloat3 &GetScaleKey(hquint32 keyIndex) {return scaleKeys[keyIndex];} 
	inline const HQFloat3 &GetScaleKey(hquint32 keyIndex) const {return scaleKeys[keyIndex];} 

	inline hqfloat32 *GetRotKeyTimePtr(hquint32 keyIndex) {return &rotKeyTimes[keyIndex];} 
	inline hqfloat32 GetRotKeyTime(hquint32 keyIndex) const {return rotKeyTimes[keyIndex];} 
	inline const HQQuaternion &GetRotKey(hquint32 keyIndex) const {return rotKeys[keyIndex];}
	inline HQQuaternion &GetRotKey(hquint32 keyIndex) {return rotKeys[keyIndex];}
	
	void ComputeTranslation(HQFloat3 &translation, hqfloat32 time, hquint32 &lastKey);
	void ComputeScale(HQFloat3 &scale, hqfloat32 time, hquint32 &lastKey);
	void ComputeRotation(HQQuaternion &rotation, hqfloat32 time, hquint32 &lastKey);

	void ComputeTranslationFixedPeriod(HQFloat3 &translation, hqfloat32 time, hquint32 &lastKey);
	void ComputeScaleFixedPeriod(HQFloat3 &scale, hqfloat32 time, hquint32 &lastKey);
	void ComputeRotationFixedPeriod(HQQuaternion &rotation, hqfloat32 time, hquint32 &lastKey);

	hquint32 nodeIndex;
	hquint32 numTransKeys;
	hquint32 numScaleKeys;
	hquint32 numRotKeys;

	//translation keys
	hqfloat32 fixTransKeyPeriod;//0.0f means no fixed period between key
	hqfloat32 *transKeyTimes;
	HQFloat3* transKeys;

	//scale keys
	hqfloat32 fixScaleKeyPeriod;//0.0f means no fixed period between key
	hqfloat32 *scaleKeyTimes;
	HQFloat3* scaleKeys;

	//rotation keys
	hqfloat32 fixRotKeyPeriod;//0.0f means no fixed period between key
	hqfloat32 *rotKeyTimes;
	HQQuaternion* rotKeys;

	typedef void (HQAnimationKeySet::*ComputeTranslationFuncPtr) (HQFloat3 &translation, hqfloat32 time, hquint32 &lastKey);
	typedef void (HQAnimationKeySet::*ComputeScaleFuncPtr) (HQFloat3 &scale, hqfloat32 time, hquint32 &lastKey);
	typedef void (HQAnimationKeySet::*ComputeRotationFuncPtr) (HQQuaternion &rotation, hqfloat32 time, hquint32 &lastKey);

	ComputeTranslationFuncPtr computeTranslationFunc;
	ComputeScaleFuncPtr computeScaleFunc;
	ComputeRotationFuncPtr computeRotationFunc;

};


struct HQAnimation
{
	HQAnimation() : name(NULL), numKeySets(0), keySets(NULL), duration(0.f)
	{}
	~HQAnimation() {
		delete[] name;
		delete[] keySets;
	}


	char *name;

	hquint32 numKeySets;
	HQAnimationKeySet *keySets;
	hqfloat32 duration;
};


struct HQAnimationStaticInfo : public HQReferenceCountObj
{
	HQAnimationStaticInfo();
	~HQAnimationStaticInfo();

	hquint32 numBones;
	hquint32 *boneToNodeIndices;
	HQMatrix3x4 *boneOffsetMatrices;

	hquint32 numAnimations;
	HQAnimation *animations;
};

struct HQMeshNode::AnimationInfo
{
	AnimationInfo();
	~AnimationInfo();

	hquint32 numNodes;
	HQAnimationSceneNode * nodes;
	
	HQMatrix3x4 *boneTransformMatrices;

	HQAnimationStaticInfo * staticInfo;//this can be shared between multiple instance

	HQAnimation *activeAnimation;
	hqfloat32 currentTime;

	hquint32 *currentTransKeys;//for keeping track of last largest key that have time smaller than current time
	hquint32 *currentScaleKeys;
	hquint32 *currentRotKeys;
};

#endif
