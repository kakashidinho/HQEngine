#ifndef MAIN_H
#define MAIN_H
#include "../HQPrimitiveDataType.h"
#include "../HQRendererCoreType.h"
#include "../HQVertexAttribute.h"
#include "../HQUtilMath.h"
#include "AllocMeshX.h"

#include <d3d9.h>
#include <d3dx9.h>
#include <string.h>
#include <string>
#include <map>

#define HQMESH_MAGIC_STR "HQEngineMeshFile"
#define HQANIM_MAGIC_STR "HQEngineAnimFile"
#define MAX_NUM_BONES_SUPPORTED 128

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
		pKey = new HQQuaternion();
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


void CreateWindowHandle();
void CreateDirect3dDevice();
void CleanUpDirectX();

void ConvertXToHQMeshFile(const char *dest, const char* source);

void ConvertToHQMeshFile(const char *dest, const char* source);

char* GetAnimationFileName(const char* destMeshFile);

#endif