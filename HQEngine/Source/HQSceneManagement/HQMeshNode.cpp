/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQScenePCH.h"
#include "../HQMeshNode.h"
#include "../HQEngineApp.h"
#include "../HQEngine/HQEngineResManagerImpl.h"
#include "helperFunctions.h"
#include "HQMeshNodeInternal.h"

#include <stdio.h>
#include <string.h>

#ifdef ANDROID
#	include <android/log.h>
//#	define TRACE(...) __android_log_print(ANDROID_LOG_DEBUG, "HQMeshNode", __VA_ARGS__)
#	define TRACE(...)
#else
#	define TRACE(...)
#endif

#include "../HQEngine/HQEngineCommonInternal.h"

using namespace HQEngineHelper;

#define HQMESH_MAGIC_STR "HQEngineMeshFile"

struct HQMeshNode::MeshFileHeader
{
	char magicString[16];
	hquint32 numVertices;
	hquint32 numVertexAttribs;
	hquint32 vertexSize;//size of 1 vertex's data
	hquint32 numIndices;
	HQIndexDataType indexDataType;
	hquint32 numSubMeshes;
};


/*-----------HQMeshNode---------------*/
HQMeshNode::HQMeshNode(const char *name,
		const char *hqMeshFileName, 
		HQRenderDevice *pDevice, 
		hquint32 vertexShaderID, 
		HQLogStream *pLogStream)
:	HQSceneNode(name),
	m_pRenderDevice(pDevice), 
	m_geoInfo(new GeometricInfo()),
	m_animInfo(new AnimationInfo()),
	m_reloading(false)
{

	this->Init(name, hqMeshFileName, vertexShaderID, pLogStream);

}

HQMeshNode::HQMeshNode(const char *name,
		const char *hqMeshFileName, 
		HQRenderDevice *pDevice, 
		const char* vertexShaderName, 
		HQLogStream *pLogStream)
:	HQSceneNode(name),
	m_pRenderDevice(pDevice), 
	m_geoInfo(new GeometricInfo()),
	m_animInfo(new AnimationInfo()),
	m_reloading(false)
{

	HQEngineShaderResource* shaderRes = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource(vertexShaderName);
	HQEngineShaderResImpl* shaderResImpl = (HQEngineShaderResImpl*) shaderRes;
	hquint32 vertexShaderID = shaderResImpl != NULL ? shaderResImpl->GetShaderID(): HQ_NOT_USE_VSHADER;


	this->Init(name, hqMeshFileName, vertexShaderID, pLogStream);

}


void HQMeshNode::Init(const char *name,
		const char *hqMeshFileName ,
		hquint32 vertexShaderID,
		HQLogStream *pLogStream)
{
	m_hqMeshFileName = HQ_NEW char[strlen(hqMeshFileName) + 1];
	strcpy(m_hqMeshFileName, hqMeshFileName);

	//load binary file
	HQDataReaderStream* f = HQEngineApp::GetInstance()-> OpenFileForRead(hqMeshFileName);
	if (f == NULL)
	{
		if (pLogStream != NULL)
		{
			const char prefix[] = "could not open file";
			char *message = HQ_NEW char[strlen(prefix) + strlen(hqMeshFileName) + 2];
			
			sprintf(message, "%s %s", prefix, hqMeshFileName);

			pLogStream->Log("HQMeshNode:", message);
		
			delete[] message;
		}

		throw std::bad_alloc();
	}


	/*----start read file---------*/
	MeshFileHeader header;
	fread(&header, sizeof(MeshFileHeader), 1, f);
	if (strncmp(header.magicString, HQMESH_MAGIC_STR, strlen(HQMESH_MAGIC_STR)))
	{
		if (pLogStream != NULL)
		{
			const char prefix[] = "file";
			const char postfix[] = "is not HQEngine Mesh File";
			char *message = HQ_NEW char[strlen(prefix) + strlen(hqMeshFileName) + strlen(postfix) + 3];
			
			sprintf(message, "%s %s %s", prefix, hqMeshFileName, postfix);

			pLogStream->Log("HQMeshNode:", message);

			size_t len = strlen(HQMESH_MAGIC_STR);

			char *currentMagicString = HQ_NEW char[len + 1];
			memcpy(currentMagicString, header.magicString, len);
			currentMagicString[len] = 0;

			pLogStream->Log("HQMeshNode: this is the magic string read from the file: %s", currentMagicString);

			delete[] currentMagicString;
		
			delete[] message;
		}

		throw std::bad_alloc();
	}

	//load geometric data
	if (!this->LoadGeometricInfo(f, header, vertexShaderID))
	{
		if (pLogStream != NULL)
		{
			const char prefix[] = "could not load geometric info from";
			char *message = HQ_NEW char[strlen(prefix) + strlen(hqMeshFileName) + 2];
			
			sprintf(message, "%s %s", prefix, hqMeshFileName);

			pLogStream->Log("HQMeshNode:", message);
		
			delete[] message;
		}
		throw std::bad_alloc();
	}
	
	//load animation data
	unsigned char c;
	if ((c = fgetc(f)) != 0)
	{
		//c is length of animation file name
		char *animFile = HQ_NEW char[c + 1];
		fread(animFile, c, 1, f);
		animFile[c] = '\0';

		fclose(f);

		if (!this->LoadAnimationInfo(animFile))
		{
			if (pLogStream != NULL)
			{
				const char prefix[] = "could not load animation info from";
				char *message = HQ_NEW char[strlen(prefix) + strlen(hqMeshFileName) + 2];
				
				sprintf(message, "%s %s", prefix, hqMeshFileName);

				pLogStream->Log("HQMeshNode:", message);
			
				delete[] message;
			}
			delete[] animFile;

			throw std::bad_alloc();
		}

		delete[] animFile;

		this->AddChild(&m_animInfo->nodes[0]);
	}
	else
	{
		fclose(f);

		//create dummy animation info
		m_animInfo->staticInfo->animations = HQ_NEW HQAnimation[1];
		m_animInfo->activeAnimation = m_animInfo->staticInfo->animations;
	}

	this->Reset();
}

HQMeshNode::~HQMeshNode()
{
	m_pRenderDevice->GetVertexStreamManager()->RemoveVertexInputLayout(m_geoInfo->vertexInputLayout);
	m_pRenderDevice->GetVertexStreamManager()->RemoveVertexBuffer(m_geoInfo->vertexBuffer);
	m_pRenderDevice->GetVertexStreamManager()->RemoveIndexBuffer(m_geoInfo->indexBuffer);

	for (hquint32 i = 0; i < m_geoInfo->numGroups; ++i)
	{
		for (hquint32 j = 0; j < this->m_geoInfo->groups[i].material.numTextures; ++j)
		{
			HQEngineApp::GetInstance()->GetResourceManager() ->RemoveTextureResource(this->m_geoInfo->groups[i].material.textures[j]);
		}
		delete[] this->m_geoInfo->groups[i].material.textures;
	}

	m_geoInfo->Release();
	HQ_DELETE (m_animInfo);
	delete[] m_hqMeshFileName;
}

hquint32 HQMeshNode::GetNumSubMeshes()
{
	return m_geoInfo->numGroups;
}

const HQSubMeshInfo & HQMeshNode::GetSubMeshInfo(hquint32 submeshIndex)
{
	return m_geoInfo->groups[submeshIndex].material;
}


void HQMeshNode::OnResetDevice()
{
	if (m_hqMeshFileName == NULL)
		return;
	
	m_reloading = true;

	HQDataReaderStream* f = HQEngineApp::GetInstance()-> OpenFileForRead(m_hqMeshFileName);

	MeshFileHeader header;
	fread(&header, sizeof(MeshFileHeader), 1, f);


	this->LoadGeometricInfo(f, header, 0);

	fclose(f);
}

bool HQMeshNode::LoadGeometricInfo(void *fptr, MeshFileHeader &header, hquint32 vertexShaderID)
{
	HQDataReaderStream *f = (HQDataReaderStream*) fptr;
	HQReturnVal re;
	
	if (!m_reloading)
	{//only do this on first loading
		this->m_geoInfo->numGroups = header.numSubMeshes;
		this->m_geoInfo->groups = HQ_NEW HQGeometricGroup[header.numSubMeshes];

		//create vertex input layout
		HQVertexAttribDesc *vAttribDescs = HQ_NEW HQVertexAttribDesc[header.numVertexAttribs];

		for (hquint32 i = 0; i < header.numVertexAttribs; ++i)
		{
			vAttribDescs[i].stream = 0;
			fread(&vAttribDescs[i].offset, sizeof(hquint32), 1, f);
			fread(&vAttribDescs[i].dataType, sizeof(hquint32), 1, f);
			fread(&vAttribDescs[i].usage, sizeof(hquint32), 1, f);
		}
		
		re = m_pRenderDevice->GetVertexStreamManager()->CreateVertexInputLayout(
			vAttribDescs, 
			header.numVertexAttribs, 
			vertexShaderID, 
			&this->m_geoInfo->vertexInputLayout);

		delete[] vAttribDescs;

		if (HQFailed(re))
			return false;
	}
	else
	{
		//in reloading we ignore vertex input layout
		fseek(f, 3 * sizeof(hquint32) * header.numVertexAttribs, SEEK_CUR);
	}

	//create vertex buffer
	this->m_geoInfo->vertexBufferStride = header.vertexSize;
	this->m_geoInfo->numVertices = header.numVertices;

	hquint32 vertexBufferSize = header.vertexSize * header.numVertices;
	void *verticesData = malloc(vertexBufferSize);
	fread(verticesData, vertexBufferSize, 1, f);

	if (!m_reloading)
		re = m_pRenderDevice->GetVertexStreamManager()->CreateVertexBuffer(
			verticesData, 
			vertexBufferSize, 
			false, false,
			&this->m_geoInfo->vertexBuffer);
	else
		re = m_pRenderDevice->GetVertexStreamManager()->UpdateVertexBuffer(
			this->m_geoInfo->vertexBuffer,
			0, 
			vertexBufferSize,
			verticesData);

	free(verticesData);

	if (HQFailed(re))
		return false;

	//create index buffer
	this->m_geoInfo->indexDataType = header.indexDataType;

	hquint32 indexBufferSize;
	switch (header.indexDataType)
	{
	case HQ_IDT_UINT:
		indexBufferSize = header.numIndices * 4;
		break;
	default:
		indexBufferSize = header.numIndices * 2;
		break;
	}

	void *indicesData = malloc(indexBufferSize);
	fread(indicesData, indexBufferSize, 1, f);
	
	if (!m_reloading)
		re = m_pRenderDevice->GetVertexStreamManager()->CreateIndexBuffer(
			indicesData,
			indexBufferSize,
			false,
			header.indexDataType,
			&this->m_geoInfo->indexBuffer);
	else
		re = m_pRenderDevice->GetVertexStreamManager()->UpdateIndexBuffer(
				this->m_geoInfo->indexBuffer,
				0,
				indexBufferSize,
				indicesData);

	free(indicesData);

	if (HQFailed(re))
		return false;

#ifndef ANDROID
	//android need textures to be reloaded too
	if (m_reloading) 
		return true;
#endif
	//read sub mesh info
	for (hquint32 i = 0; i < header.numSubMeshes; ++i)
	{
		fread(&this->m_geoInfo->groups[i].startIndex, sizeof(hquint32), 1, f);
		fread(&this->m_geoInfo->groups[i].numIndices, sizeof(hquint32), 1, f);
		
		//material
		fread(&this->m_geoInfo->groups[i].material.colorMaterial, sizeof(HQColorMaterial), 1, f);

		//textures
		fread(&this->m_geoInfo->groups[i].material.numTextures, sizeof(hquint32), 1, f);
		if (this->m_geoInfo->groups[i].material.numTextures == 0)
			this->m_geoInfo->groups[i].material.textures = NULL;
		else
		{
			if (!m_reloading)
				this->m_geoInfo->groups[i].material.textures = HQ_NEW HQEngineTextureResource* [this->m_geoInfo->groups[i].material.numTextures];
			char *textureName;

			for (hquint32 j = 0; j < this->m_geoInfo->groups[i].material.numTextures; ++j)
			{
				hqubyte8 textureNameSize ;
				fread(&textureNameSize, 1, 1, f);

				textureName = HQ_NEW char[textureNameSize + 1];
				fread(textureName, textureNameSize, 1, f);
				textureName[textureNameSize] = '\0';

				//check if texture resource is already loaded
				HQEngineTextureResource* textureRes = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource(textureName);
				if (textureRes == NULL)
				{
					//create texture resource
					HQEngineApp::GetInstance()->GetResourceManager()->AddTextureResource(
						textureName, 
						textureName,
						true,
						HQ_TEXTURE_2D);

					textureRes = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource(textureName);
				}

				this->m_geoInfo->groups[i].material.textures[j] = textureRes;
		
				delete[] textureName;
			}//for (hquint32 j = 0; j < this->m_geoInfo->groups[i].material.numTextures; ++j)
		}
	}//for (hquint32 i = 0; i < header.numSubMeshes; ++i)

	return true;
}

void HQMeshNode::Update(hqfloat32 dt ,bool updateChilds , bool parentChanged)
{
	TRACE("here %s %d", __FILE__, __LINE__);

	HQSceneNode::Update(dt, updateChilds, parentChanged);

	TRACE("here %s %d", __FILE__, __LINE__);

	//update bone transformation matrix
	HQAnimationSceneNode *node;
	HQAnimationStaticInfo *staticAnimInfo = m_animInfo->staticInfo;
	for (hquint32 i = 0; i < staticAnimInfo->numBones ; ++i)
	{
		node = m_animInfo->nodes + staticAnimInfo->boneToNodeIndices[i];
		HQMatrix3x4Multiply(&node->GetWorldTransform(), &staticAnimInfo->boneOffsetMatrices[i], &m_animInfo->boneTransformMatrices[i]);
	}

	TRACE("here %s %d", __FILE__, __LINE__);
}


void HQMeshNode::BeginRender()
{
	m_pRenderDevice->SetPrimitiveMode(HQ_PRI_TRIANGLES);
	m_pRenderDevice->GetVertexStreamManager()->SetVertexInputLayout(m_geoInfo->vertexInputLayout);
	m_pRenderDevice->GetVertexStreamManager()->SetVertexBuffer(m_geoInfo->vertexBuffer, 0,  m_geoInfo->vertexBufferStride);
	m_pRenderDevice->GetVertexStreamManager()->SetIndexBuffer(m_geoInfo->indexBuffer);
}

void HQMeshNode::EndRender()
{
}

void HQMeshNode::DrawSubMesh(hquint32 submeshIndex)
{
	HQGeometricGroup &subMeshInfo = m_geoInfo->groups[submeshIndex];
	m_pRenderDevice->DrawIndexed(m_geoInfo->numVertices, subMeshInfo.numIndices, subMeshInfo.startIndex);
}

void HQMeshNode::SetSubMeshTextures(hquint32 submeshIndex)
{
	HQGeometricGroup &subMeshInfo = m_geoInfo->groups[submeshIndex];
	for (hquint32 i = 0; i < subMeshInfo.material.numTextures ; ++i)
	{
		HQEngineApp::GetInstance()->GetEffectManager() ->SetTextureForPixelShader(i, subMeshInfo.material.textures[i]);
	}
}

void HQMeshNode::DrawInOneCall()
{
	BeginRender();
	for (hquint32 i = 0 ; i < m_geoInfo->numGroups; ++i)
	{
		SetSubMeshTextures(i);
		DrawSubMesh(i);
	}
	EndRender();
}
