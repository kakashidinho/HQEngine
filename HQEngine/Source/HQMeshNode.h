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
	hquint32 minIndex;//minimum vertex index
	hquint32 startIndex;//starting vertex index in index buffer
	hquint32 numIndices;//number of vertex indices 

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
	///{vertexShaderID} is used for creating vertex input layout. If it is NULL => no vertex layout will be created. 
	///{uavVertexBuffer} = true if creating UAV vertex & index buffer is desired
	///
	HQMeshNode(const char *name,
		const char *hqMeshFileName, 
		HQRenderDevice *pDevice, 
		HQShaderObject* vertexShaderID,
		HQLogStream *pLogStream = NULL,
		bool uavVertexBuffer = false);
	///
	///{vertexShaderName} is name of shader resource used for creating vertex input layout. If it is NULL => no vertex layout will be created. 
	///{uavVertexBuffer} = true if creating UAV vertex & index buffer is desired
	///
	HQMeshNode(const char *name,
		const char *hqMeshFileName, 
		HQRenderDevice *pDevice, 
		const char* vertexShaderName,
		HQLogStream *pLogStream = NULL,
		bool uavVertexBuffer = false);
	~HQMeshNode();

	//TO DO: give an option to preprocess to know vertex layout, etc

	hquint32 GetNumSubMeshes();//number of sub-meshes. Each uses different material, textures ..
	const HQSubMeshInfo & GetSubMeshInfo(hquint32 submeshIndex);

	hquint32 GetNumBones() const;
	const HQMatrix3x4 * GetBoneTransformMatrices() const;

	///
	///may return NULL if indirect draw is not supported. Don't modify index count & first index & first vertex in buffer or 
	///it will produce wrong drawing
	///
	HQDrawIndexedIndirectArgsBuffer* GetDrawIndirectArgs();

	HQVertexBuffer * GetVertexBuffer();
	HQIndexBuffer * GetIndexBuffer();
	HQVertexLayout * GetVertexLayout();//get graphics vertex input layout that matches mesh file's vertex descriptions
	hquint32 GetVertexSize();//get size (bytes) of single vertex
	HQIndexDataType GetIndexBufferDataType();
	const HQVertexAttribDesc * GetVertexAttribDescs();//get mesh file's vertex descriptions
	hquint32 GetNumVertexAttribs();//get number of vertex attributes in mesh file's vertex descriptions

	void SetVertexLayout(HQVertexLayout *vertexLayout);//tell this mesh to use the specified vertex layout whenever BeginRender() is called

	void AdvanceAnimationTime(hqfloat32 dt);
	///
	///reset animation
	///
	void Reset();

	//override HQSceneNode's update
	void Update(hqfloat32 dt ,bool updateChilds = true, bool parentChanged = false );

	void BeginRender();//set necessary buffers, vertex layout  
	void EndRender();
	void BindSubMeshTextures(hquint32 submeshIndex);//bind the corresponding textures specified in submesh's info to pixel shader stage

	//call draw method
	void DrawSubMesh(hquint32 submeshIndex);

	//call indirect draw method
	void DrawSubMeshIndirect(hquint32 submeshIndex);

	void DrawInOneCall();

	void DrawInOneCallIndirect();

	void OnResetDevice();//need to call this when render device restore

private:
	void Init(const char *name,
		const char *hqMeshFileName ,
		HQShaderObject* vertexShaderID,
		HQLogStream *pLogStream,
		bool uavVertexBuffer);

	struct MeshFileHeader;
	struct GeometricInfo;
	struct AnimationInfo;

	void GetContainingFolder(void* folder);
	bool LoadGeometricInfo(void *f, void *containingFolderName, MeshFileHeader &header, HQShaderObject* vertexShaderID);
	bool LoadAnimationInfo(const char *fileName);

	void CreateIndirectBuffer();
	

	GeometricInfo *m_geoInfo;
	AnimationInfo *m_animInfo;

	HQRenderDevice *m_pRenderDevice;

	char *m_hqMeshFileName;
	bool m_reloading;
	bool m_uavBuffers;
};

#endif
