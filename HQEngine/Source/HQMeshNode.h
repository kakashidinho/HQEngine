/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_MESH_H
#define HQ_MESH_H

#include "HQSceneManagementCommon.h"
#include "HQSceneNode.h"
#include "HQRenderDevice.h"
#include "HQLogStream.h"

struct HQSubMeshInfo
{
	HQColorMaterial colorMaterial;
	hquint32 numTextures;
	hquint32 *textures;
};

///
///This class represents both static mesh and skinned mesh
///
class HQSCENEMANAGEMENT_API HQMeshNode : public HQSceneNode 
{
public:
	///
	///{vertexShaderID} is used for creating vertex input layout
	///
	HQMeshNode(const char *name,
		const char *hqMeshFileName, 
		HQRenderDevice *pDevice, 
		hquint32 vertexShaderID,
		HQLogStream *pLogStream = NULL);
	~HQMeshNode();
	
	hquint32 GetNumSubMeshes();//number of sub-meshes. Each uses different material, textures ..
	const HQSubMeshInfo & GetSubMeshInfo(hquint32 submeshIndex);

	hquint32 GetNumBones() const;
	const HQMatrix3x4 * GetBoneTransformMatrices() const;

	void AdvanceAnimationTime(hqfloat32 dt);
	///
	///reset animation
	///
	void Reset();

	void Update(hqfloat32 dt ,bool updateChilds = true, bool parentChanged = false );

	void BeginRender();
	void EndRender();
	void SetSubMeshTextures(hquint32 submeshIndex);
	void DrawSubMesh(hquint32 submeshIndex);//only draw with vertex & index buffer

	void DrawInOneCall();

	void OnResetDevice();//need to call this when render device restore

private:
	struct MeshFileHeader;
	struct GeometricInfo;
	struct AnimationInfo;

	bool LoadGeometricInfo(void *f, MeshFileHeader &header, hquint32 vertexShaderID);
	bool LoadAnimationInfo(const char *fileName);
	

	GeometricInfo *m_geoInfo;
	AnimationInfo *m_animInfo;

	HQRenderDevice *m_pRenderDevice;

	char *m_hqMeshFileName;
	bool m_reloading;
};

#endif
