/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQScenePCH.h"
#include "../HQMeshNode.h"
#include "HQMeshNodeInternal.h"

#include "../HQEngine/HQEngineCommonInternal.h"

using namespace HQEngineHelper;

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef ANDROID
#	include <android/log.h>
//#	define TRACE(...) __android_log_print(ANDROID_LOG_DEBUG, "HQMeshNode", __VA_ARGS__)
#	define TRACE(...)
#else
#	define TRACE(...)
#endif

#define HQANIM_MAGIC_STR "HQEngineAnimFile"

#define FLOAT_EPSILON 0.00001f

struct HQAnimationFileHeader
{
	char magicString[16];
	hquint32 numNodes;
	hquint32 numBones;
	hquint32 numAnimations;
};

#define LERP(a, b, dt)  (a * dt + b * (1 - dt))
HQ_FORCE_INLINE void Lerp(const HQFloat3& in1, const HQFloat3 &in2, HQFloat3 & out, hqfloat32 dt)
{
	out.x = LERP(in1.x, in2.x, dt);
	out.y = LERP(in1.y, in2.y, dt);
	out.z = LERP(in1.z, in2.z, dt);
}

HQ_FORCE_INLINE void Lerp(const HQQuaternion& in1, const HQQuaternion &in2, HQQuaternion & out, hqfloat32 dt)
{
	out.x = LERP(in1.x, in2.x, dt);
	out.y = LERP(in1.y, in2.y, dt);
	out.z = LERP(in1.z, in2.z, dt);
	out.w = LERP(in1.w, in2.w, dt);
}


/*-------HQAnimationKeySet--------*/
void HQAnimationKeySet::AllocTransKeys(hquint32 numKeys)
{
	transKeyTimes = HQ_NEW hqfloat32[numKeys];
	transKeys = HQ_NEW HQFloat3[numKeys];
}
void HQAnimationKeySet::AllocScaleKeys(hquint32 numKeys)
{
	scaleKeyTimes = HQ_NEW hqfloat32[numKeys];
	scaleKeys = HQ_NEW HQFloat3[numKeys];
}
void HQAnimationKeySet::AllocRotKeys(hquint32 numKeys)
{
	rotKeyTimes = HQ_NEW hqfloat32[numKeys];
	rotKeys = HQQuaternion::NewArray(numKeys);
}


void HQAnimationKeySet::ComputeTranslation(HQFloat3 &translation, hqfloat32 time, hquint32 &lastKey)
{
	hquint32 lastIndex = this->numTransKeys - 1;
	if (time >= this->GetTransKeyTime(lastIndex) - FLOAT_EPSILON)//use last key
	{
		translation = this->GetTransKey(lastIndex);
		lastKey = lastIndex;
	}
	else
	{
		hquint32 key;
		hqfloat32 dt;

		for (key = lastKey; this->GetTransKeyTime(key) < time && key < this->numTransKeys; ++key )
		{
		}
		
		lastKey = key - 1;
		
		dt = (time - this->GetTransKeyTime(lastKey)) / (this->GetTransKeyTime(lastKey + 1) - this->GetTransKeyTime(lastKey)) ;
		Lerp(this->GetTransKey(lastKey), this->GetTransKey(lastKey + 1), translation, dt);
	}
}

void HQAnimationKeySet::ComputeScale(HQFloat3 &scale, hqfloat32 time, hquint32 &lastKey)
{
	hquint32 lastIndex = this->numScaleKeys - 1;
	if (time >= this->GetScaleKeyTime(lastIndex) - FLOAT_EPSILON)//use last key
	{
		scale = this->GetScaleKey(lastIndex);
		lastKey = lastIndex;
	}
	else
	{
		hquint32 key;
		hqfloat32 dt;

		for (key = lastKey; this->GetScaleKeyTime(key) < time && key < this->numScaleKeys; ++key )
		{
		}
		
		lastKey = key - 1;
		
		dt = (time - this->GetScaleKeyTime(lastKey)) / (this->GetScaleKeyTime(lastKey + 1) - this->GetScaleKeyTime(lastKey)) ;
		Lerp(this->GetScaleKey(lastKey), this->GetScaleKey(lastKey + 1), scale, dt);
	}
}

void HQAnimationKeySet::ComputeRotation(HQQuaternion &rotation, hqfloat32 time, hquint32 &lastKey)
{
	hquint32 lastIndex = this->numRotKeys - 1;
	if (time >= this->GetRotKeyTime(lastIndex) - FLOAT_EPSILON)//use last key
	{
		rotation = this->GetRotKey(lastIndex);
		lastKey = lastIndex;
	}
	else
	{
		hquint32 key;
		hqfloat32 dt;

		for (key = lastKey; this->GetRotKeyTime(key) < time && key < this->numRotKeys; ++key )
		{
		}
		
		lastKey = key - 1;
		
		dt = (time - this->GetRotKeyTime(lastKey)) / (this->GetRotKeyTime(lastKey + 1) - this->GetRotKeyTime(lastKey)) ;

		rotation.Slerp(this->GetRotKey(lastKey), this->GetRotKey(lastKey + 1), dt);
		//Lerp(this->GetRotKey(lastKey), this->GetRotKey(lastKey + 1), rotation, dt);
	}
}


void HQAnimationKeySet::ComputeTranslationFixedPeriod(HQFloat3 &translation, hqfloat32 time, hquint32 &lastKey)
{
	time -= this->GetTransKeyTime(0);
	if (time < FLOAT_EPSILON)//use first key
	{
		lastKey = 0;
		translation = this->GetTransKey(0);
		return;
	}

	hqfloat32 indexf = floorf(time / this->fixTransKeyPeriod);
	hquint32 index = (hquint32) indexf;

	if (index >= this->numTransKeys)//use the last key
	{
		lastKey = this->numTransKeys - 1;
		translation = this->GetTransKey(lastKey);
	}
	else
	{
		hqfloat32 dt = (time - this->GetTransKeyTime(index)) / this->fixTransKeyPeriod;

		Lerp(this->GetTransKey(index), this->GetTransKey(index + 1), translation, dt);

		lastKey = index;
	}
}

void HQAnimationKeySet::ComputeScaleFixedPeriod(HQFloat3 &scale, hqfloat32 time, hquint32 &lastKey)
{
	time -= this->GetScaleKeyTime(0);
	if (time < FLOAT_EPSILON)//use first key
	{
		lastKey = 0;
		scale = this->GetScaleKey(0);
		return;
	}

	hqfloat32 indexf = floorf(time / this->fixScaleKeyPeriod);
	hquint32 index = (hquint32) indexf;

	if (index > this->numScaleKeys)//use the last key
	{
		lastKey = this->numScaleKeys - 1;
		scale = this->GetScaleKey(lastKey);
	}
	else
	{
		hqfloat32 dt = (time - this->GetScaleKeyTime(index)) / this->fixScaleKeyPeriod;

		Lerp(this->GetScaleKey(index), this->GetScaleKey(index + 1), scale, dt);

		lastKey = index;
	}
}

void HQAnimationKeySet::ComputeRotationFixedPeriod(HQQuaternion &rotation, hqfloat32 time, hquint32 &lastKey)
{
	time -= this->GetRotKeyTime(0);
	if (time < FLOAT_EPSILON)//use first key
	{
		lastKey = 0;
		rotation = this->GetRotKey(0);
		return;
	}

	hqfloat32 indexf = floorf(time / this->fixRotKeyPeriod);
	hquint32 index = (hquint32) indexf;

	if (index > this->numRotKeys)//use the last key
	{
		lastKey = this->numRotKeys - 1;
		rotation = this->GetRotKey(lastKey);
	}
	else
	{
		hqfloat32 dt = (time - this->GetRotKeyTime(index)) / this->fixRotKeyPeriod;

		rotation.Slerp(this->GetRotKey(index), this->GetRotKey(index + 1), dt);

		lastKey = index;
	}
}

/*-------HQAnimationStaticInfo---------*/
HQAnimationStaticInfo::HQAnimationStaticInfo()
:numBones(0), boneToNodeIndices(NULL), boneOffsetMatrices(NULL) ,
numAnimations (0), animations(NULL)
{
}

HQAnimationStaticInfo::~HQAnimationStaticInfo()
{
	delete[] boneToNodeIndices;
	delete[] boneOffsetMatrices;

	delete[] animations;
}

/*-----HQMeshNode::AnimationInfo-----------*/
HQMeshNode::AnimationInfo::AnimationInfo()
:numNodes (0), nodes(NULL), boneTransformMatrices(NULL), activeAnimation(NULL), 
currentTime(0.0f), currentTransKeys(NULL), currentScaleKeys(NULL), currentRotKeys(NULL)
{
	staticInfo = HQ_NEW HQAnimationStaticInfo();
}

HQMeshNode::AnimationInfo::~AnimationInfo()
{
	delete[] currentTransKeys;
	delete[] currentScaleKeys;
	delete[] currentRotKeys;
	delete[] nodes;
	delete[] boneTransformMatrices;
	staticInfo->Release();
}

/*------HQMeshNode-------*/
bool HQMeshNode::LoadAnimationInfo(const char *fileName)
{
	HQDataReaderStream* f = HQEngineApp::GetInstance()->OpenFileForRead(fileName);

	HQAnimationFileHeader header;

	fread(&header, sizeof(HQAnimationFileHeader), 1, f);

	if (strncmp(header.magicString, HQANIM_MAGIC_STR, strlen(HQANIM_MAGIC_STR)))
	{
		fclose(f);
		return false;
	}

	this->m_animInfo->numNodes = header.numNodes;
	this->m_animInfo->staticInfo->numBones = header.numBones;
	this->m_animInfo->staticInfo->numAnimations = header.numAnimations;

	this->m_animInfo->nodes = HQ_NEW HQAnimationSceneNode[header.numNodes];
	this->m_animInfo->boneTransformMatrices = HQMatrix3x4 :: NewArray(header.numBones);
	this->m_animInfo->staticInfo->boneToNodeIndices = HQ_NEW hquint32[header.numBones];
	this->m_animInfo->staticInfo->boneOffsetMatrices = HQMatrix3x4 :: NewArray(header.numBones);

	this->m_animInfo->staticInfo->animations = HQ_NEW HQAnimation[header.numAnimations];
	if (this->m_animInfo->staticInfo->animations > 0)
		this->m_animInfo->activeAnimation = &this->m_animInfo->staticInfo->animations[0];

	//load nodes data
	for (hquint32 i = 0 ; i < header.numNodes; ++i)
	{
		hqubyte8 nameLength;
		hquint32 relIndex;
		char *& nodeName = m_animInfo->nodes[i].GetNamePointer();
		
		fread(&nameLength, 1, 1, f);//name length
		nodeName = HQ_NEW char[nameLength + 1];
		fread(nodeName, nameLength, 1, f);
		nodeName[nameLength] = '\0';

		fread(&relIndex, sizeof(hquint32), 1, f);//sibling index
		if (relIndex != header.numNodes)
			m_animInfo->nodes[i].AddSibling(&m_animInfo->nodes[relIndex]);
		
		fread(&relIndex, sizeof(hquint32), 1, f);//first child index
		if (relIndex != header.numNodes)
			m_animInfo->nodes[i].AddChild(&m_animInfo->nodes[relIndex]);
	}
	
	//load bones data
	for (hquint32 i = 0 ; i < header.numBones; ++i)
	{
		fread(&m_animInfo->staticInfo->boneToNodeIndices[i], sizeof(hquint32), 1, f);//read node index
		
		fread(&m_animInfo->staticInfo->boneOffsetMatrices[i], sizeof(HQMatrix3x4), 1, f);//read offset matric

	}

	//load animation data
	hquint32 largestNumberOfKeySets = 0;
	for (hquint32 i = 0 ; i < header.numAnimations; ++i)
	{
		HQAnimation &animation = this->m_animInfo->staticInfo->animations[i];
		unsigned char nameLen;//animation name length
		fread(&nameLen, 1, 1, f);//read name length
		
		animation.name = HQ_NEW char[nameLen + 1];
		fread(animation.name, nameLen, 1, f);//read name
		animation.name[nameLen] = '\0';

		fread(&animation.numKeySets, sizeof(hquint32), 1, f);//read number of key sets
		animation.keySets = HQ_NEW HQAnimationKeySet[animation.numKeySets];
		

		if (largestNumberOfKeySets < animation.numKeySets)
			largestNumberOfKeySets = animation.numKeySets;
	
		for (hquint32 j = 0 ; j < animation.numKeySets; ++j)
		{
			bool fixedPeriod;
			HQAnimationKeySet &keySet = animation.keySets[j];

			fread(&keySet.nodeIndex, sizeof(hquint32), 1, f);
			fread(&keySet.numTransKeys, sizeof(hquint32), 1, f);//read number of translation keys
			fread(&keySet.numScaleKeys, sizeof(hquint32), 1, f);//read number of scale keys
			fread(&keySet.numRotKeys, sizeof(hquint32), 1, f);//read number of rotation keys

			keySet.AllocTransKeys(keySet.numTransKeys);
			keySet.AllocScaleKeys(keySet.numScaleKeys);
			keySet.AllocRotKeys(keySet.numRotKeys);

			//read translation keys
			fixedPeriod = true;
			for (hquint32 k = 0; k < keySet.numTransKeys; ++k)
			{
				fread(keySet.GetTransKeyTimePtr(k), sizeof(hqfloat32), 1, f);//read key time
				fread(&keySet.GetTransKey(k), sizeof(HQFloat3), 1, f);//read key value
				
				if (animation.duration < keySet.GetTransKeyTime(k))
					animation.duration = keySet.GetTransKeyTime(k);

				//check if all keys have fixed period
				if (fixedPeriod && k > 0)
				{
					hqfloat32 period = keySet.GetTransKeyTime(k) - keySet.GetTransKeyTime(k - 1);
					if (k == 1)
						keySet.fixTransKeyPeriod = period;
					else if (period != keySet.fixTransKeyPeriod)
					{
						fixedPeriod = false;
						keySet.fixTransKeyPeriod = 0.0f;
					}
				}
			}
			if (fixedPeriod)
			{
				keySet.computeTranslationFunc = &HQAnimationKeySet::ComputeTranslationFixedPeriod;
			}
			else
				keySet.computeTranslationFunc = &HQAnimationKeySet::ComputeTranslation;

			//read scale keys
			fixedPeriod = true;
			for (hquint32 k = 0; k < keySet.numScaleKeys; ++k)
			{
				fread(keySet.GetScaleKeyTimePtr(k), sizeof(hqfloat32), 1, f);//read key time
				fread(&keySet.GetScaleKey(k), sizeof(HQFloat3), 1, f);//read key value
				
				if (animation.duration < keySet.GetScaleKeyTime(k))
					animation.duration = keySet.GetScaleKeyTime(k);

				//check if all keys have fixed period
				if (fixedPeriod && k > 0)
				{
					hqfloat32 period = keySet.GetScaleKeyTime(k) - keySet.GetScaleKeyTime(k - 1);
					if (k == 1)
						keySet.fixScaleKeyPeriod = period;
					else if (period != keySet.fixScaleKeyPeriod)
					{
						fixedPeriod = false;
						keySet.fixScaleKeyPeriod = 0.0f;
					}
				}
			}

			if (fixedPeriod)
			{
				keySet.computeScaleFunc = &HQAnimationKeySet::ComputeScaleFixedPeriod;
			}
			else
				keySet.computeScaleFunc = &HQAnimationKeySet::ComputeScale;

			//read rotation keys
			fixedPeriod = true;
			for (hquint32 k = 0; k < keySet.numRotKeys; ++k)
			{
				fread(keySet.GetRotKeyTimePtr(k), sizeof(hqfloat32), 1, f);//read key time
				fread(&keySet.GetRotKey(k), sizeof(HQQuaternion), 1, f);//read key value
				
				if (animation.duration < keySet.GetRotKeyTime(k))
					animation.duration = keySet.GetRotKeyTime(k);

				//check if all keys have fixed period
				if (fixedPeriod && k > 0)
				{
					hqfloat32 period = keySet.GetRotKeyTime(k) - keySet.GetRotKeyTime(k - 1);
					if (k == 1)
						keySet.fixRotKeyPeriod = period;
					else if (period != keySet.fixRotKeyPeriod)
					{
						fixedPeriod = false;
						keySet.fixRotKeyPeriod = 0.0f;
					}
				}
			}

			if (fixedPeriod)
			{
				keySet.computeRotationFunc = &HQAnimationKeySet::ComputeRotationFixedPeriod;
			}
			else
				keySet.computeRotationFunc = &HQAnimationKeySet::ComputeRotation;

		}//for (hquint32 j = 0 ; j < animation.numKeySets; ++j)
	}//for (hquint32 i = 0 ; i < header.numAnimations; ++i)

	fclose(f);

	m_animInfo->currentTransKeys = HQ_NEW hquint32 [largestNumberOfKeySets];
	m_animInfo->currentScaleKeys = HQ_NEW hquint32 [largestNumberOfKeySets];
	m_animInfo->currentRotKeys = HQ_NEW hquint32 [largestNumberOfKeySets];

	return true;
}

hquint32 HQMeshNode::GetNumBones() const
{
	return m_animInfo->staticInfo->numBones;
}

const HQMatrix3x4 * HQMeshNode::GetBoneTransformMatrices() const
{
	return m_animInfo->boneTransformMatrices;
}

void HQMeshNode::AdvanceAnimationTime(hqfloat32 dt)
{
	TRACE("here %s %d", __FILE__, __LINE__);
	if (dt < FLOAT_EPSILON)//too small
		return;
	m_animInfo->currentTime += dt;

	if (m_animInfo->currentTime > m_animInfo->activeAnimation->duration && m_animInfo->activeAnimation->duration > FLOAT_EPSILON)
	{
		hqfloat32 currentTime = m_animInfo->currentTime;

		this->Reset();
		
		m_animInfo->currentTime = fmod(currentTime, m_animInfo->activeAnimation->duration);
	}

	TRACE("here %s %d", __FILE__, __LINE__);

	for (hquint32 i = 0; i < m_animInfo->activeAnimation->numKeySets; ++i)
	{
		HQAnimationKeySet &keySet = m_animInfo->activeAnimation->keySets[i];
		HQAnimationSceneNode *node = m_animInfo->nodes + keySet.nodeIndex;

		HQVector4 *pTrans;
		HQFloat3 *pScale;
		HQQuaternion *pRot;
		
		node->WantToChangeLocalTransform(pTrans, pScale, pRot);

		
		(keySet.*keySet.computeTranslationFunc)(*pTrans, m_animInfo->currentTime, m_animInfo->currentTransKeys[i]);
		(keySet.*keySet.computeScaleFunc)(*pScale, m_animInfo->currentTime, m_animInfo->currentScaleKeys[i]);
		(keySet.*keySet.computeRotationFunc)(*pRot, m_animInfo->currentTime, m_animInfo->currentRotKeys[i]);
	}

	TRACE("here %s %d", __FILE__, __LINE__);
}

void HQMeshNode::Reset()
{
	m_animInfo->currentTime = 0.0f;

	for (hquint32 i = 0; i < m_animInfo->activeAnimation->numKeySets; ++i)
	{
		HQAnimationKeySet &keySet = m_animInfo->activeAnimation->keySets[i];
		HQAnimationSceneNode *node = m_animInfo->nodes + keySet.nodeIndex;

		HQVector4 *pTrans;
		HQFloat3 *pScale;
		HQQuaternion *pRot;
		
		node->WantToChangeLocalTransform(pTrans, pScale, pRot);
		
		HQFloat3 & transKey = keySet.GetTransKey(0);
		pTrans->Set(transKey.x, transKey.y, transKey.z);
		*pScale = keySet.GetScaleKey(0);
		*pRot = keySet.GetRotKey(0);

		m_animInfo->currentTransKeys[i] = 0;
		m_animInfo->currentScaleKeys[i] = 0;
		m_animInfo->currentRotKeys[i] = 0;
	}
}
