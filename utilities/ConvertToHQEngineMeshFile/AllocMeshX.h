//=============================================================================
// AllocMeshHierarchy.h by Frank Luna (C) 2005 All Rights Reserved.
//=============================================================================

#ifndef ALLOC_MESH_HIERARCHY_H
#define ALLOC_MESH_HIERARCHY_H

#include <d3dx9.h>
#include <list>
// Implements the ID3DXAllocateHierarchy interface.  In order to create and destroy an animation 
// hierarchy using the D3DXLoadMeshHierarchyFromX and D3DXFrameDestroy functions, we must implement
// the ID3DXAllocateHierarchy interface, which defines how meshes and frames are created and 
// destroyed, thereby giving us some flexibility in the construction and destruction process.
#define SafeRelease(p) {if (p) {p->Release(); p = 0;}}


struct FrameEx : public D3DXFRAME
{
	DWORD index;
};

class AllocMeshHierarchy : public ID3DXAllocateHierarchy 
{
private:
	::LPDIRECT3DDEVICE9 d3ddev;
	std::list<LPD3DXMESHCONTAINER> meshes;
	//DWORD nMat;;
	DWORD numFrames;
public:
	AllocMeshHierarchy();

	void setDevice(LPDIRECT3DDEVICE9 _d3ddev){
		d3ddev=_d3ddev;
	};
	std::list<LPD3DXMESHCONTAINER> GetMeshList() {return meshes;}

	DWORD GetNumFrames() {return numFrames;}

	HRESULT STDMETHODCALLTYPE CreateFrame(LPCSTR Name, D3DXFRAME** ppNewFrame);                     

	HRESULT STDMETHODCALLTYPE CreateMeshContainer(LPCSTR Name, const D3DXMESHDATA* pMeshData,               
		const D3DXMATERIAL* pMaterials, const D3DXEFFECTINSTANCE* pEffectInstances, DWORD NumMaterials, 
		const DWORD *pAdjacency, ID3DXSkinInfo* pSkinInfo, D3DXMESHCONTAINER** ppNewMeshContainer);     

	HRESULT STDMETHODCALLTYPE DestroyFrame(THIS_ D3DXFRAME* pFrameToFree);              
	HRESULT STDMETHODCALLTYPE DestroyMeshContainer(THIS_ D3DXMESHCONTAINER* pMeshContainerBase);
	//~AllocMeshHierarchy();
};

#endif // ALLOC_MESH_HIERARCHY_H
