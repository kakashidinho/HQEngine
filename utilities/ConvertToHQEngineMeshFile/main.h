/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef MAIN_H
#define MAIN_H
#include "../../HQEngine/Source/HQPrimitiveDataType.h"
#include "../../HQEngine/Source/HQRendererCoreType.h"
#include "../../HQEngine/Source/HQVertexAttribute.h"
#include "../../HQEngine/Source/HQUtilMath.h"
#include "AllocMeshX.h"

#include <d3d9.h>
#include <d3dx9.h>
#include <string.h>
#include <string>
#include <map>

#define HQMESH_MAGIC_STR "HQEngineMeshFile"
#define HQANIM_MAGIC_STR "HQEngineAnimFile"
#define MAX_NUM_BONES_SUPPORTED 128

#define FLAG_FLAT_FACES 0x1
#define FLAG_FORCE_32BIT_INDICES 0x2
#define FLAG_OUTPUT_ADDITIONAL_INFO 0x4
#define FLAG_FORCE_WHITE_TEXTURE 0x8

struct HQMeshFileHeader
{
	char magicString[16];
	hquint32 numVertices;
	hquint32 numVertexAttribs;
	hquint32 vertexSize;//size of 1 vertex's data
	hquint32 numIndices;
	HQIndexDataType indexDataType;
	hquint32 numSubMeshes;
};

struct HQAnimationFileHeader
{
	char magicString[16];
	hquint32 numNodes;
	hquint32 numBones;
	hquint32 numAnimations;
};

struct Node {
	Node(): nameLength(0), name(NULL)
	{
	}
	~Node() {
		delete[] name;
	}

	unsigned char nameLength;
	char *name;
	hquint32 siblingIndex;
	hquint32 firstChildIndex;
};

struct Bone {
	hquint32 nodeIndex;

	hqfloat32 offsetMatrix[12];//matrix 3 x 4
};

struct Float3Key
{
	hqfloat32 time;
	HQFloat3 key;
};

struct QuaternionKey
{
	QuaternionKey() {
		pKey = HQQuaternion::New();
	}
	~QuaternionKey()
	{
		delete pKey;
	}

	hqfloat32 time;
	HQQuaternion *pKey;
};

struct AnimationKeySet
{
	AnimationKeySet() 
		:transKeys(NULL), scaleKeys(NULL), rotKeys(NULL)
	{
	}

	~AnimationKeySet()
	{
		delete[] transKeys;
		delete[] scaleKeys;
		delete[] rotKeys;
	}

	hquint32 nodeIndex;
	hquint32 numTransKeys;
	hquint32 numScaleKeys;
	hquint32 numRotKeys;

	Float3Key* transKeys;
	Float3Key* scaleKeys;
	QuaternionKey* rotKeys;

};


struct Animation
{
	Animation() : nameLength(0), name(NULL), numKeySets(0), keySets(NULL)
	{}
	~Animation() {
		delete[] name;
		delete[] keySets;
	}


	unsigned char nameLength;
	char *name;

	hquint32 numKeySets;
	AnimationKeySet *keySets;
};

struct MeshAdditionalInfo {
	HQFloat3 bboxMin, bboxMax;
	hqfloat32 meshSurfArea;//mesh's surface area
};


void CreateWindowHandle();
void CreateDirect3dDevice();
void CleanUpDirectX();

void ConvertXToHQMeshFile(const char *dest, const char* source, int flags);

void ConvertToHQMeshFile(const char *dest, const char* source, int flags);

char* GetAnimationFileName(const char* destMeshFile);
char* GetMoreInfoFileName(const char* destMeshFile);//get name of destination file containing additional info of the mesh

void WriteMoreInfo(const char* destMeshFile, const MeshAdditionalInfo &info);
void WriteWhiteBMPImage(const char* name);
#endif
