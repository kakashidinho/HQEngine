/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#include "main.h"
#include "Assimp/include/assimp.h"
#include "Assimp/include/aiPostProcess.h"
#include "Assimp/include/aiScene.h"

#include <float.h>

#pragma comment(lib, "Assimp/lib/assimp_release-dll_win32/assimp.lib")

//limitations for now:
//-only support static model
//-only support one vertex color set
//-only support one texcoords set
//-only support one texture per vertex
// TO DO: remove limits

void AssimpWriteGemetricDataToFile(FILE *f, const aiScene * scene, MeshAdditionalInfo& additionalInfo, int flags);
void PrintVertexAttrDesc(HQVertexAttribDesc &desc);

void ConvertToHQMeshFile(const char *dest, const char * source, int flags)
{

	unsigned int aiFlags = aiProcess_Triangulate | aiProcess_MakeLeftHanded | aiProcess_FlipWindingOrder ;
	if (flags & FLAG_FLAT_FACES)
		aiFlags |= aiProcess_GenNormals;
	else
		aiFlags |= aiProcess_JoinIdenticalVertices | aiProcess_GenSmoothNormals;

	const aiScene * scene = aiImportFile(source , aiFlags );
	
	if (scene == NULL)
		return;

	//open file
	FILE *f = fopen(dest, "wb");
	
	//write gemeotry data
	MeshAdditionalInfo additionalInfo;
	AssimpWriteGemetricDataToFile(f, scene, additionalInfo, flags);

	//TO DO: add animation file later
	fputc(0, f);//for now, no animation

	//write bounding box info
	if (flags & FLAG_OUTPUT_ADDITIONAL_INFO)
		WriteMoreInfo(dest, additionalInfo);

	//clean up
	fclose(f);
	aiReleaseImport(scene);
}

void AssimpWriteGemetricDataToFile(FILE *f, const aiScene * scene, MeshAdditionalInfo& additionalInfo, int flags)
{
	HQMeshFileHeader header;

	strncpy(header.magicString, HQMESH_MAGIC_STR, strlen(HQMESH_MAGIC_STR));

	/*-----compute file header------*/
	header.numSubMeshes = scene->mNumMeshes;

	//compute num indices
	header.numIndices = 0;
	for (unsigned int i = 0; i < scene->mNumMeshes; ++i){
		header.numIndices += scene->mMeshes[i]->mNumFaces * 3;
	}
	
	//compute num vertices
	header.numVertices = 0;

	for (unsigned int i = 0; i < scene->mNumMeshes; ++i){
		header.numVertices += scene->mMeshes[i]->mNumVertices;
	}

	//set indices data type.
	if (header.numIndices <= 0xffff && (flags & FLAG_FORCE_32BIT_INDICES) == 0)
		header.indexDataType = HQ_IDT_USHORT;
	else
		header.indexDataType = HQ_IDT_UINT;

	//compute per-vertex data size and vertex attribute description
	header.numVertexAttribs = 2;//number of attributes is 2, enough for normal and position
	header.vertexSize = 6 * sizeof(float);//enough for normal and position
	if (scene->mMeshes[0]->HasVertexColors(0))
	{
		header.vertexSize += 4 * sizeof(float);
		header.numVertexAttribs ++;
	}
	if (scene->mMeshes[0]->HasTextureCoords(0))
	{
		header.vertexSize += 2 * sizeof(float);
		header.numVertexAttribs ++;
	}


	HQVertexAttribDescArray2 vaDescs(header.numVertexAttribs);
	//vertex layout in output file: <position> <normal> <texcoords> <color>
	vaDescs.SetPosition(0, 0, 0, HQ_VADT_FLOAT3);
	vaDescs.SetNormal(1, 0, 3 * sizeof(float), HQ_VADT_FLOAT3);
	if (scene->mMeshes[0]->HasTextureCoords(0))
		vaDescs.SetTexcoord(2, 0, 6 * sizeof(float), HQ_VADT_FLOAT2, 0);
	if (scene->mMeshes[0]->HasVertexColors(0))
	{
		vaDescs.SetColor(3, 0, 8 * sizeof(float), HQ_VADT_FLOAT4);
	}

	//write header and vertex desciption array
	fwrite(&header, sizeof(HQMeshFileHeader), 1, f);

	for (unsigned int i = 0; i < header.numVertexAttribs; ++i)
	{
		PrintVertexAttrDesc(vaDescs[i]);//print the desciption to the screen

		fwrite(&vaDescs[i].offset, sizeof(hquint32), 1, f);
		fwrite(&vaDescs[i].dataType, sizeof(hquint32), 1, f);
		fwrite(&vaDescs[i].usage, sizeof(hquint32), 1, f);
	}

	//bounding box
	additionalInfo.bboxMin.Set(FLT_MAX, FLT_MAX, FLT_MAX);
	additionalInfo.bboxMax.Set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	//total surface area
	additionalInfo.meshSurfArea = 0.f;
	//write vertex data
	for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
	{
		for (unsigned v = 0; v < scene->mMeshes[i]->mNumVertices; ++v){
			fwrite(&scene->mMeshes[i]->mVertices[v], sizeof(aiVector3D), 1, f);//position
			fwrite(&scene->mMeshes[i]->mNormals[v], sizeof(aiVector3D), 1, f);//normal
			if (scene->mMeshes[0]->HasTextureCoords(0))//texcoords
				fwrite(&scene->mMeshes[i]->mTextureCoords[0][v], sizeof(aiVector2D), 1, f);
			if (scene->mMeshes[0]->HasVertexColors(0))//vertex color
				fwrite(&scene->mMeshes[i]->mColors[0][v], sizeof(aiColor4D), 1, f);

			//compute bounding box
			aiVector3D& position = scene->mMeshes[i]->mVertices[v];
			if (additionalInfo.bboxMin.x > position.x)
				additionalInfo.bboxMin.x = position.x;
			if (additionalInfo.bboxMin.y > position.y)
				additionalInfo.bboxMin.y = position.y;
			if (additionalInfo.bboxMin.z > position.z)
				additionalInfo.bboxMin.z = position.z;

			if (additionalInfo.bboxMax.x < position.x)
				additionalInfo.bboxMax.x = position.x;
			if (additionalInfo.bboxMax.y < position.y)
				additionalInfo.bboxMax.y = position.y;
			if (additionalInfo.bboxMax.z < position.z)
				additionalInfo.bboxMax.z = position.z;

		}
	}

	//write index data
	unsigned int prevMeshVertices = 0;
	for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
	{
		for (unsigned t = 0; t < scene->mMeshes[i]->mNumFaces; ++t){
			aiVector3D* pTriVertex[3];
			for (int v = 0; v < 3; ++v)
			{
				pTriVertex[v] = &scene->mMeshes[i]->mVertices[scene->mMeshes[i]->mFaces[t].mIndices[v]];
				unsigned idx = scene->mMeshes[i]->mFaces[t].mIndices[v] + prevMeshVertices;
				if (header.indexDataType == HQ_IDT_USHORT)
				{
					//convert to 16 bit 
					unsigned short idx16 = (unsigned short) (idx & 0xffff);
					fwrite(&idx16, sizeof(unsigned short), 1, f);
				}
				else
					fwrite(&idx, sizeof(unsigned int), 1, f);

			}//for (int v = 0; v < 3; ++v)

			//compute surface area
			aiVector3D v0v1 = *pTriVertex[1] - *pTriVertex[0];
			aiVector3D v0v2 = *pTriVertex[2] - *pTriVertex[0];
			aiVector3D v0v1v2c = v0v1 ^ v0v2;
			hqfloat32 triArea = 0.5f * v0v1v2c.Length();
			additionalInfo.meshSurfArea += triArea;
			
		}//for (unsigned t = 0; t < scene->mMeshes[i]->mNumFaces; ++t)

		prevMeshVertices += scene->mMeshes[i]->mNumVertices;
	}

	//write sub mesh info
	unsigned int prevMeshIndices = 0;
	for (unsigned int i = 0; i < header.numSubMeshes; ++i)
	{
		hquint32 numTextures = 0;
		HQColorMaterial colorMat;

		aiMesh * mesh = scene->mMeshes[i];
		aiMaterial * material = scene->mMaterials[mesh->mMaterialIndex];
		
		//start index and num indices
		unsigned int startIndex = prevMeshIndices;
		unsigned int numIndices = mesh->mNumFaces * 3;

		fwrite(&startIndex, sizeof(hquint32), 1, f);
		fwrite(&numIndices, sizeof(hquint32), 1, f);

		//color material
		aiColor3D color;
		material->Get<aiColor3D>(AI_MATKEY_COLOR_DIFFUSE , color );
		memcpy(colorMat.diffuse , &color , 3 * sizeof(float));
		//OPACITY
		material->Get<hqfloat32>(AI_MATKEY_OPACITY , colorMat.diffuse.a );

		material->Get<aiColor3D>(AI_MATKEY_COLOR_AMBIENT , color );
		memcpy(colorMat.ambient , &color , 3 * sizeof(float));
		colorMat.ambient.a = colorMat.diffuse.a;

		material->Get<aiColor3D>(AI_MATKEY_COLOR_SPECULAR , color );
		memcpy(colorMat.specular , &color , 3 * sizeof(float));
		colorMat.specular.a = colorMat.diffuse.a;

		material->Get<aiColor3D>(AI_MATKEY_COLOR_EMISSIVE , color );
		memcpy(colorMat.emissive , &color , 3 * sizeof(float));
		colorMat.emissive.a = colorMat.diffuse.a;

		material->Get<float>(AI_MATKEY_SHININESS , colorMat.power);
		colorMat.power *= 0.25f;//assimp multiplies this value by 4 after model loaded. we need to convert it back
		fwrite(&colorMat, sizeof(HQColorMaterial), 1, f);

		//texture
		aiString textureName; 
		material->Get<aiString>(AI_MATKEY_TEXTURE_DIFFUSE(0) , textureName);
		if (textureName.length > 0)
			numTextures = 1;
		fwrite(&numTextures, sizeof(hquint32), 1, f);
		if (textureName.length > 0)
		{
			unsigned char textureFileNameLen = textureName.length;
			fwrite(&textureFileNameLen, 1, 1, f);
			fwrite(textureName.data, textureFileNameLen, 1, f);
		}


		//shift the starting index of next sub mesh
		prevMeshIndices += mesh->mNumFaces * 3;
	}
}


void PrintVertexAttrDesc(HQVertexAttribDesc &vAttrDesc)
{
	printf ("Element: ");
	printf ("offset=%u usage=", vAttrDesc.offset);
	
	switch (vAttrDesc.usage)
	{
	case  HQ_VAU_POSITION:
		printf("HQ_VAU_POSITION ");
		break;
	case HQ_VAU_COLOR:
		printf("HQ_VAU_COLOR ");
		break;
	case HQ_VAU_NORMAL:
		printf("HQ_VAU_NORMAL ");
		break;
	case HQ_VAU_TEXCOORD0: case HQ_VAU_TEXCOORD1: case HQ_VAU_TEXCOORD2: case HQ_VAU_TEXCOORD3:
	case HQ_VAU_TEXCOORD4: case HQ_VAU_TEXCOORD5: case HQ_VAU_TEXCOORD6: case HQ_VAU_TEXCOORD7: 
		printf("HQ_VAU_TEXCOORD%d ",vAttrDesc.usage - HQ_VAU_TEXCOORD0);
		break;
		break;
	case HQ_VAU_TANGENT:
		printf("HQ_VAU_TANGENT ");
		break;
	case HQ_VAU_BINORMAL:
		printf("HQ_VAU_BINORMAL ");
		break;
	case  HQ_VAU_BLENDWEIGHT:
		printf("HQ_VAU_BLENDWEIGHT ");
		break;
	case HQ_VAU_BLENDINDICES:
		printf("HQ_VAU_BLENDINDICES ");
		break;
	case HQ_VAU_PSIZE:
		printf("HQ_VAU_PSIZE ");
		break;
	}

	printf("type=");

	switch (vAttrDesc.dataType)
	{
	case HQ_VADT_FLOAT:
		printf("HQ_VADT_FLOAT");
		break;
	case HQ_VADT_FLOAT2:
		printf("HQ_VADT_FLOAT2");
		break;
	case HQ_VADT_FLOAT3:
		printf("HQ_VADT_FLOAT3");
		break;
	case HQ_VADT_FLOAT4:
		printf("HQ_VADT_FLOAT4");
		break;
	case HQ_VADT_UBYTE4 :
		printf("HQ_VADT_UBYTE4");
		break;
	case HQ_VADT_SHORT2 :
		printf("HQ_VADT_SHORT2");
		break;
	case HQ_VADT_SHORT4 :
		printf("HQ_VADT_SHORT4");
		break;
	case HQ_VADT_USHORT2N :
		printf("HQ_VADT_USHORT2N");
		break;
	case HQ_VADT_USHORT4N :
		printf("HQ_VADT_USHORT4N");
		break;
	case HQ_VADT_UBYTE4N:
		printf("HQ_VADT_UBYTE4N");
		break;
	}
	

	printf("\n");
}