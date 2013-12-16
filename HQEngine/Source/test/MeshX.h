#pragma once

#include "AllocMeshX.h"
#include "HQUtilMath.h"

class MeshX
{
public:
	MeshX(const char *xFileName);
	~MeshX();

	AllocMeshHierarchy &GetAllocMeshHierarchy() {return allocMeshHierarchy;}
	
	ID3DXAnimationController *GetAnimationController() {return animCtrl;}

	FrameEx *GetRootNode() {return root;}

	const HQMatrix3x4* GetBoneMatrices() const {return boneMatrices3x4;}

	void Update();
private:
	void UpdateNode(FrameEx *node, FrameEx *parent);

	void CreateWindowHandle();
	void CreateDirect3dDevice();

	
	void BuildSkinnedMesh();
	bool HasNormals(ID3DXMesh* mesh);

	AllocMeshHierarchy allocMeshHierarchy;
	
	FrameEx * root;
	ID3DXAnimationController *animCtrl;
	ID3DXSkinInfo *skinInfo;
	ID3DXMesh *mesh;

	D3DXMATRIX *boneMatrices;
	HQMatrix3x4 *boneMatrices3x4;

	HWND            hwnd;
	LPDIRECT3DDEVICE9 pDevice;
	LPDIRECT3D9 pD3D;

};