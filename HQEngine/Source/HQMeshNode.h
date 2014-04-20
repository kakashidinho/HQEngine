/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_MESH_H
#define HQ_MESH_H

#include "HQSceneManagementCommon.h"
#include "HQEngineResManager.h"
#include "HQSceneNode.h"
#include "HQRenderDevice.h"
#include "HQLogStream.h"

struct HQSubMeshInfo
{
	HQColorMaterial colorMaterial;
	hquint32 numTextures;
	HQEngineTextureResource **textures;
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
		HQShaderObject* vertexShaderID,
		HQLogStream *pLogStream = NULL);
	///
	///{vertexShaderName} is name of shader resource used for creating vertex input layout
	///
	HQMeshNode(const char *name,
		const char *hqMeshFileName, 
		HQRenderDevice *pDevice, 
		const char* vertexShaderName,
		HQLogStream *pLogStream = NULL);
	~HQMeshNode();
	
	hquint32 GetNumSubMeshes();//number of sub-meshes. Each uses different material, textures ..
	const HQSubMeshInfo & GetSubMeshInfo(hquint32 submeshIndex);

	hquint32 GetNumBones() const;
	const HQMatrix3x4 * GetBoneTransformMatrices() const;

	///
	///may return NULL if indirect draw is not supported. Don't modify index count & first index & first vertex in buffer or 
	///it will produce wrong drawing
	///
	HQDrawIndexedIndirectArgsBuffer* GetDrawIndirectArgs();

	void AdvanceAnimationTime(hqfloat32 dt);
	///
	///reset animation
	///
	void Reset();

	//override HQSceneNode's update
	void Update(hqfloat32 dt ,bool updateChilds = true, bool parentChanged = false );

	void BeginRender();
	void EndRender();
	void SetSubMeshTextures(hquint32 submeshIndex);
	void DrawSubMesh(hquint32 submeshIndex);//only draw with vertex & index buffer

	//only draw with vertex & index buffer & indirect buffer
	void DrawSubMeshIndirect(hquint32 submeshIndex);

	void DrawInOneCall();

	void DrawInOneCallIndirect();

	void OnResetDevice();//need to call this when render device restore

private:
	void Init(const char *name,
		const char *hqMeshFileName ,
		HQShaderObject* vertexShaderID,
		HQLogStream *pLogStream); 

	struct MeshFileHeader;
	struct GeometricInfo;
	struct AnimationInfo;

	bool LoadGeometricInfo(void *f, MeshFileHeader &header, HQShaderObject* vertexShaderID);
	bool LoadAnimationInfo(const char *fileName);

	void CreateIndirectBuffer();
	

	GeometricInfo *m_geoInfo;
	AnimationInfo *m_animInfo;

	HQRenderDevice *m_pRenderDevice;

	char *m_hqMeshFileName;
	bool m_reloading;
};

#endif
