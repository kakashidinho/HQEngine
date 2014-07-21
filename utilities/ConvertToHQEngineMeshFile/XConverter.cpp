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

#include <string.h>
#include <map>

#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9.lib")

HWND            hwnd;
LPDIRECT3DDEVICE9 pDevice = NULL;
LPDIRECT3D9 pD3D = NULL;

void XWriteGemetricDataToFile(FILE *f, ID3DXMesh *mesh, LPD3DXMESHCONTAINER meshContainer, int flags);
void XWriteAnimationData(const char *dest, const char * source, LPD3DXMESHCONTAINER pMeshContainer, D3DXFRAME *root, DWORD numFrames);

void XBuildStaticMesh(const char *source, LPD3DXMESHCONTAINER *ppMeshContainerOut);
DWORD XBuildSkinnedMesh(ID3DXMesh * pMeshIn, ID3DXSkinInfo *pSkinInfo, ID3DXMesh ** ppMeshOut);
bool XHasNormals(ID3DXMesh* mesh);
void XConvertToHQVerAttribDesc(const D3DVERTEXELEMENT9 &element, HQVertexAttribDesc &vAttrDesc);
void XMapFrame2PointerArray(FrameEx *frame, FrameEx **pFrames);
void XCreateName2NodeMap(std::map<std::string, FrameEx*> &nodeMap, FrameEx **pFrames, DWORD numFrames);
void XCreateBonesList(std::map<std::string, FrameEx*> &nodeMap, Bone *bones, ID3DXSkinInfo *pSkinInfo);
void XCreateAnimations(Animation **ppAnims, std::map<std::string, FrameEx*> &nodeMap, const aiScene *scene);

LRESULT CALLBACK DummyWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	return DefWindowProc(hwnd, message, wParam, lParam);
}


void CleanUpDirectX()
{
	//clean up
	pDevice->Release();
	pD3D->Release();
}

void CreateWindowHandle()
{
	WNDCLASSEX      wndclass;
	// initialize the window 
	wndclass.hIconSm       = LoadIcon(NULL,IDI_APPLICATION);
	wndclass.hIcon         = LoadIcon(NULL,IDI_APPLICATION);
	wndclass.cbSize        = sizeof(wndclass);
	wndclass.lpfnWndProc   = DummyWndProc;
	wndclass.cbClsExtra    = 0;
	wndclass.cbWndExtra    = 0;
	wndclass.hInstance     = GetModuleHandle(NULL);
	wndclass.hCursor       = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW);
	wndclass.lpszMenuName  = NULL;
	wndclass.lpszClassName = L"Dummy Class";
	wndclass.style         = CS_HREDRAW | CS_VREDRAW ;
	
	RegisterClassEx(&wndclass) ;


	hwnd = CreateWindowEx( NULL, L"Dummy Class",
					L"Dummy Window",
					WS_OVERLAPPEDWINDOW ,
					CW_USEDEFAULT,
					CW_USEDEFAULT,
					300, 300, NULL, NULL, GetModuleHandle(NULL), NULL);
}

void CreateDirect3dDevice()
{
	pD3D = Direct3DCreate9(D3D_SDK_VERSION);
	if(!pD3D)
		exit(-3);

	D3DPRESENT_PARAMETERS d3dp = {0};

	d3dp.AutoDepthStencilFormat= D3DFMT_D16;
	d3dp.BackBufferCount=1;
	d3dp.BackBufferFormat=D3DFMT_X8R8G8B8;
	d3dp.BackBufferWidth=300;
	d3dp.BackBufferHeight=300;
	d3dp.EnableAutoDepthStencil=FALSE;
	d3dp.hDeviceWindow=hwnd;

	d3dp.PresentationInterval= D3DPRESENT_INTERVAL_DEFAULT;
	
	d3dp.SwapEffect=D3DSWAPEFFECT_DISCARD;
	d3dp.Windowed=TRUE;
	
	HRESULT hr=pD3D->CreateDevice(0,
						  D3DDEVTYPE_REF,
						  hwnd,
						  D3DCREATE_SOFTWARE_VERTEXPROCESSING,
						  &d3dp,
						  &pDevice);
	
	if (FAILED(hr))
	{
		pD3D->Release();
		exit(-3);
	}
}

void ConvertXToHQMeshFile(const char *dest, const char* source, int flags)
{
	//TO DO: flags 

	FILE *f;
	HRESULT hr;
	AllocMeshHierarchy allocMeshHierarchy;
	D3DXFRAME *root = 0;
	ID3DXAnimationController *animCtrl = 0;
	ID3DXMesh *mesh = 0;
	LPD3DXMESHCONTAINER meshContainer ;
	DWORD maxBoneInf;
	bool animation = true;
	char *animFile = 0;

	
	allocMeshHierarchy.setDevice(pDevice);

	//try to load skinned mesh
	hr = D3DXLoadMeshHierarchyFromXA(source, D3DXMESH_SYSTEMMEM,
		pDevice, &allocMeshHierarchy, 0, /* ignore user data */ 
		&root,	&animCtrl);

	if (FAILED(hr))
	{
		if (hr == D3DERR_INVALIDCALL)//no animation info
		{
			XBuildStaticMesh(source, &meshContainer);
			animation = false;
		}
		else
		{
			pDevice->Release();
			pD3D->Release();
			exit(-3);
		}
	}

	if (animation)
	{
		std::list<LPD3DXMESHCONTAINER> meshes = allocMeshHierarchy.GetMeshList();
		//only export first mesh
		meshContainer = meshes.front();

		//convert to skinned mesh
		maxBoneInf = XBuildSkinnedMesh(meshContainer->MeshData.pMesh, meshContainer->pSkinInfo, &mesh);
		
		printf("max bone influencies: %u\n", maxBoneInf);

		//get animation file name
		animFile = GetAnimationFileName(dest);

		//Write animation data
		XWriteAnimationData(animFile, source, meshContainer, root, allocMeshHierarchy.GetNumFrames());
	}
	else
	{
		mesh = meshContainer->MeshData.pMesh;
	}

	//open file
	f = fopen(dest, "wb");

	XWriteGemetricDataToFile(f, mesh, meshContainer, flags);


	if (animation)
	{
		unsigned char animFileNameLen = strlen(animFile);
		fwrite(&animFileNameLen, 1, 1, f);//write animation file name length
		fwrite(animFile, animFileNameLen, 1, f);//write animation file name at the end of mesh file
	}
	else//only do this when loading static mesh
	{
		fputc(0, f);

		free(meshContainer->pAdjacency);
		free(meshContainer->pMaterials);

		delete meshContainer;
	}
	fclose(f);


	
	D3DXFrameDestroy(root, &allocMeshHierarchy);
	SafeRelease(animCtrl);
	SafeRelease(mesh);
	delete[] animFile;

}

void XWriteGemetricDataToFile(FILE *f, ID3DXMesh *mesh, LPD3DXMESHCONTAINER meshContainer, int flags)
{
	HQMeshFileHeader header;
	D3DXATTRIBUTERANGE *attrRanges = NULL;
	DWORD attrTableSize;

	strncpy(header.magicString, HQMESH_MAGIC_STR, strlen(HQMESH_MAGIC_STR));

	//computer file header
	header.numSubMeshes = meshContainer->NumMaterials;
	header.numIndices = mesh->GetNumFaces() * 3;
	header.numVertices = mesh->GetNumVertices();
	if (header.numIndices > 0xffff || (flags & FLAG_FORCE_32BIT_INDICES) != 0)
		header.indexDataType = HQ_IDT_UINT;
	else
		header.indexDataType = HQ_IDT_USHORT;
	header.vertexSize = mesh->GetNumBytesPerVertex();
	//compute vertex attribute descs
	header.numVertexAttribs = 0;
	D3DVERTEXELEMENT9 elements[MAX_FVF_DECL_SIZE];
	mesh->GetDeclaration(elements);
	while(elements[header.numVertexAttribs].Stream != 0xff)
	{
		header.numVertexAttribs ++;
	}

	fwrite(&header, sizeof(HQMeshFileHeader), 1, f);

	HQVertexAttribDesc *descs = new HQVertexAttribDesc[header.numVertexAttribs];
	for (unsigned int i = 0; i < header.numVertexAttribs; ++i)
	{
		XConvertToHQVerAttribDesc(elements[i], descs[i]);
		fwrite(&descs[i].offset, sizeof(hquint32), 1, f);
		fwrite(&descs[i].dataType, sizeof(hquint32), 1, f);
		fwrite(&descs[i].usage, sizeof(hquint32), 1, f);
	}

	//write vertex data
	void *pData;
	mesh->LockVertexBuffer(D3DLOCK_READONLY, &pData);
	fwrite(pData, header.vertexSize, header.numVertices, f);
	mesh->UnlockVertexBuffer();

	//write index data
	mesh->LockIndexBuffer(D3DLOCK_READONLY, &pData);
	if (header.numIndices <= 0xffff && (flags & FLAG_FORCE_32BIT_INDICES) != 0)
	{
		for (hquint32 i = 0; i < header.numIndices; ++i)
		{
			hquint32 index = (hquint32) (*(((hqushort16*)pData) + i));
			fwrite(&index, 4, 1, f);
		}
	}
	else
	{
		if (header.numIndices > 0xffff)
			fwrite(pData, 4, header.numIndices, f);
		else
			fwrite(pData, 2, header.numIndices, f);
	}
	mesh->UnlockIndexBuffer();

	//compute sub mesh info
	mesh->GetAttributeTable(NULL, &attrTableSize);
	attrRanges = new D3DXATTRIBUTERANGE[attrTableSize];
	mesh->GetAttributeTable(attrRanges, &attrTableSize);

	for (unsigned int i = 0; i < header.numSubMeshes; ++i)
	{
		hquint32 numTextures = 0;
		HQColorMaterial colorMat;
		hquint32 startIndex = attrRanges[i].FaceStart * 3;
		hquint32 numIndices = attrRanges[i].FaceCount * 3;

		fwrite(&startIndex, sizeof(hquint32), 1, f);
		fwrite(&numIndices, sizeof(hquint32), 1, f);

		//color material
		memcpy(&colorMat, &meshContainer->pMaterials[i].MatD3D, sizeof(HQColorMaterial));
		fwrite(&colorMat, sizeof(HQColorMaterial), 1, f);

		//texture
		if (meshContainer->pMaterials[i].pTextureFilename)
			numTextures = 1;
		fwrite(&numTextures, sizeof(hquint32), 1, f);
		if (meshContainer->pMaterials[i].pTextureFilename)
		{
			unsigned char textureFileNameLen = strlen(meshContainer->pMaterials[i].pTextureFilename);
			fwrite(&textureFileNameLen, 1, 1, f);
			fwrite(meshContainer->pMaterials[i].pTextureFilename, textureFileNameLen, 1, f);
		}
	}

	delete[] descs;
	delete[] attrRanges;
}

void XWriteAnimationData(const char *dest, const char * source, LPD3DXMESHCONTAINER meshContainer, D3DXFRAME *root, DWORD numFrames)
{
	HQAnimationFileHeader header;
	FrameEx **pFrames = 0;
	Bone *bones = 0;
	Node *nodes = 0;
	std::map<std::string, FrameEx*> nodeMap;
	Animation *anims = 0;
	const aiScene *scene;//load animation keys using assimp
	scene = aiImportFile(source,aiProcess_MakeLeftHanded);
	
	strncpy(header.magicString, HQANIM_MAGIC_STR, strlen(HQANIM_MAGIC_STR));


	//create nodes list
	FrameEx *rootEx = (FrameEx*)root;
	pFrames = new FrameEx *[numFrames];
	XMapFrame2PointerArray(rootEx, pFrames);

	//move root node to the first position
	if (rootEx->index != 0)
	{
		FrameEx *first = pFrames[0];
		
		first->index = rootEx->index;
		pFrames[first->index] = first;

		pFrames[0] = rootEx;
		rootEx->index = 0;
	}

	nodes = new Node [numFrames];
	
	for (hquint32 i = 0; i < numFrames ; ++i)
	{
		nodes[i].nameLength = strlen (pFrames[i]->Name);
		nodes[i].name = new char[nodes[i].nameLength];
		strncpy(nodes[i].name, pFrames[i]->Name, nodes[i].nameLength);

		FrameEx *frame = (FrameEx*)pFrames[i]->pFrameSibling;
		nodes[i].siblingIndex = (frame)? frame->index : numFrames;

		frame = (FrameEx*)pFrames[i]->pFrameFirstChild;
		nodes[i].firstChildIndex = (frame)? frame->index : numFrames;
	}

	//create bones list
	bones = new Bone[meshContainer->pSkinInfo->GetNumBones()];
	XCreateName2NodeMap(nodeMap, pFrames, numFrames);
	XCreateBonesList(nodeMap, bones, meshContainer->pSkinInfo);

	//create animation keys
	XCreateAnimations(&anims, nodeMap, scene);

	//computer file header
	header.numNodes = numFrames;
	header.numBones = meshContainer->pSkinInfo->GetNumBones();
	header.numAnimations = scene->mNumAnimations;

	FILE * f = fopen(dest, "wb");

	fwrite(&header, sizeof(HQAnimationFileHeader), 1, f);
	//write nodes data
	for (hquint32 i = 0; i < numFrames ; ++i)
	{
		fwrite(&nodes[i].nameLength, 1, 1, f);//write name length
		fwrite(nodes[i].name, nodes[i].nameLength, 1, f);//write name
		fwrite(&nodes[i].siblingIndex, sizeof(hquint32), 1, f);//write sibling index
		fwrite(&nodes[i].firstChildIndex, sizeof(hquint32), 1, f);//write first child index
	}

	//write bones data
	fwrite(bones, sizeof(Bone), header.numBones, f);
	
	//write animation keys
	//animations
	for (hquint32 i = 0; i < header.numAnimations; ++i)
	{
		fwrite(&anims[i].nameLength, 1, 1, f);
		fwrite(anims[i].name, anims[i].nameLength, 1, f);
		fwrite(&anims[i].numKeySets, sizeof(hquint32), 1, f);

		//key sets, each set for each node
		for (hquint32 j = 0; j < anims[i].numKeySets; ++j)
		{
			AnimationKeySet &keySet = anims[i].keySets[j];

			fwrite(&keySet.nodeIndex, sizeof(hquint32), 1, f);
			fwrite(&keySet.numTransKeys, sizeof(hquint32), 1, f);
			fwrite(&keySet.numScaleKeys, sizeof(hquint32), 1, f);
			fwrite(&keySet.numRotKeys, sizeof(hquint32), 1, f);

			//write translation keys
			for (unsigned int k = 0; k < keySet.numTransKeys ; ++k)
			{
				fwrite(&keySet.transKeys[k], sizeof(Float3Key), 1, f);
			}

			//write scaling keys
			for (unsigned int k = 0; k < keySet.numScaleKeys ; ++k)
			{
				fwrite(&keySet.scaleKeys[k], sizeof(Float3Key), 1, f);
			}

			//write rotation keys
			for (unsigned int k = 0; k < keySet.numRotKeys ; ++k)
			{
				fwrite(&keySet.rotKeys[k].time, sizeof(hqfloat32), 1, f);
				fwrite(keySet.rotKeys[k].pKey, sizeof(HQQuaternion), 1, f);
			}
		}
	}


	fclose(f);

	delete[] nodes;
	delete[] pFrames;
	delete[] bones;
	delete[] anims;
	aiReleaseImport(scene);

}

void XBuildStaticMesh(const char *source, LPD3DXMESHCONTAINER *ppMeshContainerOut)
{
	*ppMeshContainerOut = new D3DXMESHCONTAINER();
	ID3DXBuffer * adjacency = 0;
	ID3DXBuffer * materials = 0;
	ID3DXMesh *pMesh = 0;

	HRESULT hr = D3DXLoadMeshFromXA(source, D3DXMESH_SYSTEMMEM, pDevice, &adjacency, &materials, NULL,
		&(*ppMeshContainerOut)->NumMaterials, &pMesh);
	
	if (FAILED(hr))
	{
		pDevice->Release();
		pD3D->Release();
		exit(-3);
	}

	(*ppMeshContainerOut)->pAdjacency = (DWORD*) malloc(adjacency->GetBufferSize());
	 
	if( !XHasNormals(pMesh ))
		D3DXComputeNormals(pMesh, 0);

	//optimize the mesh
	pMesh->Optimize(D3DXMESHOPT_ATTRSORT | D3DXMESHOPT_VERTEXCACHE, 
		(DWORD*)adjacency->GetBufferPointer(), (*ppMeshContainerOut)->pAdjacency, 0, 0, &(*ppMeshContainerOut)->MeshData.pMesh);

	(*ppMeshContainerOut)->pMaterials = (D3DXMATERIAL*) malloc(materials->GetBufferSize());
	memcpy( (*ppMeshContainerOut)->pMaterials, materials->GetBufferPointer(), materials->GetBufferSize() );

	SafeRelease(adjacency);
	SafeRelease(materials);
	SafeRelease(pMesh);
}

DWORD XBuildSkinnedMesh(ID3DXMesh * mesh, ID3DXSkinInfo *pSkinInfo, ID3DXMesh ** pMeshOut)
{
	//compute normal
	D3DVERTEXELEMENT9 elements[MAX_FVF_DECL_SIZE];
	mesh->GetDeclaration(elements);

	ID3DXMesh* tempMesh = 0;
	ID3DXMesh* tempOpMesh = 0;
	mesh->CloneMesh(D3DXMESH_SYSTEMMEM, elements, pDevice, &tempMesh);
	 
	if( !XHasNormals(tempMesh) )
		D3DXComputeNormals(tempMesh, 0);

	//optimize the mesh
	DWORD* adj = new DWORD[tempMesh->GetNumFaces()*3];
	ID3DXBuffer* remap = 0;
	tempMesh->GenerateAdjacency(0.00001f, adj);
	tempMesh->Optimize(D3DXMESH_SYSTEMMEM | D3DXMESHOPT_VERTEXCACHE | 
		D3DXMESHOPT_ATTRSORT, adj, 0, 0, &remap, &tempOpMesh);

	SafeRelease(tempMesh);

	// In the .X file (specifically the array DWORD vertexIndices[nWeights]
	// data member of the SkinWeights template) each bone has an array of
	// indices which identify the vertices of the mesh that the bone influences.
	// Because we have just rearranged the vertices (from optimizing), the vertex 
	// indices of a bone are obviously incorrect (i.e., they index to vertices the bone
	// does not influence since we moved vertices around).  In order to update a bone's 
	// vertex indices to the vertices the bone _does_ influence, we simply need to specify
	// where we remapped the vertices to, so that the vertex indices can be updated to 
	// match.  This is done with the ID3DXSkinInfo::Remap method.
	pSkinInfo->Remap(tempOpMesh->GetNumVertices(), 
		(DWORD*)remap->GetBufferPointer());
	SafeRelease(remap); // Done with remap info.

	DWORD        numBoneComboEntries = 0;
	ID3DXBuffer* boneComboTable      = 0;
	DWORD maxVertInfluences;
	pSkinInfo->ConvertToIndexedBlendedMesh(tempOpMesh, 0,  
		MAX_NUM_BONES_SUPPORTED, 0, 0, 0, 0, &maxVertInfluences,
		&numBoneComboEntries, &boneComboTable, pMeshOut);

	SafeRelease(tempOpMesh);
	SafeRelease(boneComboTable);
	delete[] adj;

	return maxVertInfluences;
}

bool XHasNormals(ID3DXMesh* mesh)
{
	D3DVERTEXELEMENT9 elems[MAX_FVF_DECL_SIZE];
	mesh->GetDeclaration(elems);
	
	bool hasNormals = false;
	for(int i = 0; i < MAX_FVF_DECL_SIZE; ++i)
	{
		// Did we reach D3DDECL_END() {0xFF,0,D3DDECLTYPE_UNUSED, 0,0,0}?
		if(elems[i].Stream == 0xff)
			break;

		if( elems[i].Type == D3DDECLTYPE_FLOAT3 &&
			elems[i].Usage == D3DDECLUSAGE_NORMAL &&
			elems[i].UsageIndex == 0 )
		{
			hasNormals = true;
			break;
		}
	}
	return hasNormals;
}

void XConvertToHQVerAttribDesc(const D3DVERTEXELEMENT9 &element, HQVertexAttribDesc &vAttrDesc)
{
	printf ("Element: ");
	vAttrDesc.offset = element.Offset;
	printf ("offset=%u usage=", element.Offset);
	
	switch (element.Usage)
	{
	case  D3DDECLUSAGE_POSITION:
		vAttrDesc.usage = HQ_VAU_POSITION;
		printf("HQ_VAU_POSITION ");
		break;
	case D3DDECLUSAGE_COLOR:
		vAttrDesc.usage = HQ_VAU_COLOR;
		printf("HQ_VAU_COLOR ");
		break;
	case D3DDECLUSAGE_NORMAL	:
		vAttrDesc.usage = HQ_VAU_NORMAL;
		printf("HQ_VAU_NORMAL ");
		break;
	case D3DDECLUSAGE_TEXCOORD	:
		vAttrDesc.usage = (HQVertAttribUsage)(HQ_VAU_TEXCOORD0 + element.UsageIndex);
		printf("HQ_VAU_TEXCOORD%d ", element.UsageIndex);
		break;
		break;
	case D3DDECLUSAGE_TANGENT	:
		vAttrDesc.usage = HQ_VAU_TANGENT;
		printf("HQ_VAU_TANGENT ");
		break;
	case D3DDECLUSAGE_BINORMAL:
		vAttrDesc.usage = HQ_VAU_BINORMAL;
		printf("HQ_VAU_BINORMAL ");
		break;
	case  D3DDECLUSAGE_BLENDWEIGHT:
		vAttrDesc.usage = HQ_VAU_BLENDWEIGHT;
		printf("HQ_VAU_BLENDWEIGHT ");
		break;
	case D3DDECLUSAGE_BLENDINDICES:
		vAttrDesc.usage = HQ_VAU_BLENDINDICES;
		printf("HQ_VAU_BLENDINDICES ");
		break;
	case D3DDECLUSAGE_PSIZE:
		vAttrDesc.usage = HQ_VAU_PSIZE	;
		printf("HQ_VAU_PSIZE ");
		break;
	}

	printf("type=");

	switch (element.Type)
	{
	case D3DDECLTYPE_FLOAT1 :
		vAttrDesc.dataType = HQ_VADT_FLOAT;
		printf("HQ_VADT_FLOAT");
		break;
	case D3DDECLTYPE_FLOAT2 :
		vAttrDesc.dataType = HQ_VADT_FLOAT2;
		printf("HQ_VADT_FLOAT2");
		break;
	case D3DDECLTYPE_FLOAT3 :
		vAttrDesc.dataType = HQ_VADT_FLOAT3;
		printf("HQ_VADT_FLOAT3");
		break;
	case D3DDECLTYPE_FLOAT4 :
		vAttrDesc.dataType = HQ_VADT_FLOAT4;
		printf("HQ_VADT_FLOAT4");
		break;
	case D3DDECLTYPE_UBYTE4 :
		vAttrDesc.dataType = HQ_VADT_UBYTE4;
		printf("HQ_VADT_UBYTE4");
		break;
	case D3DDECLTYPE_SHORT2 :
		vAttrDesc.dataType = HQ_VADT_SHORT2;
		printf("HQ_VADT_SHORT2");
		break;
	case D3DDECLTYPE_SHORT4 :
		vAttrDesc.dataType = HQ_VADT_SHORT4;
		printf("HQ_VADT_SHORT4");
		break;
	case D3DDECLTYPE_USHORT2N :
		vAttrDesc.dataType = HQ_VADT_USHORT2N;
		printf("HQ_VADT_USHORT2N");
		break;
	case D3DDECLTYPE_USHORT4N :
		vAttrDesc.dataType = HQ_VADT_USHORT4N;
		printf("HQ_VADT_USHORT4N");
		break;
	case D3DDECLTYPE_D3DCOLOR:
		vAttrDesc.dataType = HQ_VADT_UBYTE4N;
		printf("HQ_VADT_UBYTE4N");
		break;
	}
	

	printf("\n");
}

void XMapFrame2PointerArray(FrameEx *frame, FrameEx **pFrames)
{
	if (frame == NULL)
		return;
	pFrames[frame->index] = frame;

	XMapFrame2PointerArray((FrameEx *)frame->pFrameSibling, pFrames);
	XMapFrame2PointerArray((FrameEx *)frame->pFrameFirstChild, pFrames);
}

void XCreateName2NodeMap(std::map<std::string, FrameEx*> &nodeMap, FrameEx **pFrames, DWORD numFrames)
{
	for (DWORD i = 0; i < numFrames; ++i)
	{
		std::string cppName = pFrames[i]->Name;
		nodeMap[cppName] = pFrames[i];
	}
}

void XCreateBonesList(std::map<std::string, FrameEx*> &nodeMap, Bone *bones, ID3DXSkinInfo *pSkinInfo)
{

	for (hquint32 i = 0; i < pSkinInfo->GetNumBones() ; ++i)
	{
		const char *boneName = pSkinInfo->GetBoneName(i);
		std::string cppName = boneName;
		const FrameEx *frame = nodeMap[cppName];
		bones[i].nodeIndex = frame->index;

		D3DXMATRIX transOffsetMatrix;//convert to column major matrix
		D3DXMatrixTranspose(&transOffsetMatrix, pSkinInfo->GetBoneOffsetMatrix(i));

		memcpy(bones[i].offsetMatrix, &transOffsetMatrix, 12 * sizeof(hqfloat32));
	}
}

void XCreateAnimations(Animation **ppAnims, std::map<std::string, FrameEx*> &nodeMap, const aiScene *scene)
{
	hquint32 nAnims = scene->mNumAnimations;
	*ppAnims = new Animation[nAnims];
	Animation *anims = *ppAnims;


	for (hquint32 i = 0; i < nAnims; ++i)
	{
		aiAnimation *animation = scene->mAnimations[i];

		const char* animName = animation->mName.data;
		anims[i].nameLength = strlen(animName);
		if (anims[i].nameLength == 0)
		{
			anims[i].nameLength = strlen("<no name>");
			anims[i].name = new char[anims[i].nameLength];
			strncpy(anims[i].name, "<no name>", anims[i].nameLength);
		}
		else {
			anims[i].name = new char[anims[i].nameLength];
			strncpy(anims[i].name, animName, anims[i].nameLength);
		}

		anims[i].numKeySets = animation->mNumChannels;
		anims[i].keySets = new AnimationKeySet[anims[i].numKeySets];

		for (hquint32 j = 0; j < anims[i].numKeySets ; ++j)
		{
			AnimationKeySet &keySet = anims[i].keySets[j];
			const char *nodeName;
			aiNodeAnim *channel = animation->mChannels[j];
			nodeName = channel->mNodeName.data;

			FrameEx * node = nodeMap[nodeName];
			keySet.nodeIndex = node->index;
			keySet.numTransKeys = channel->mNumPositionKeys;
			keySet.numScaleKeys = channel->mNumScalingKeys;
			keySet.numRotKeys = channel->mNumRotationKeys;

			keySet.transKeys = new Float3Key[keySet.numTransKeys];
			keySet.scaleKeys = new Float3Key[keySet.numScaleKeys];
			keySet.rotKeys = new QuaternionKey[keySet.numRotKeys];

			for (unsigned int k = 0; k < keySet.numRotKeys ; ++k)
			{
				aiQuatKey &rotKey = 	channel->mRotationKeys[k];
				keySet.rotKeys[k].time = (float)(rotKey.mTime / 1000.0);
				
				keySet.rotKeys[k].pKey->x = rotKey.mValue.x;
				keySet.rotKeys[k].pKey->y = rotKey.mValue.y;
				keySet.rotKeys[k].pKey->z = rotKey.mValue.z;
				keySet.rotKeys[k].pKey->w = rotKey.mValue.w;
				keySet.rotKeys[k].pKey->Normalize();
			}
			for (unsigned int k = 0; k < keySet.numScaleKeys ; ++k)
			{
				aiVectorKey &scaleKey = 	channel->mScalingKeys[k];
				keySet.scaleKeys[k].time = (float)(scaleKey.mTime / 1000.0);
				
				keySet.scaleKeys[k].key.x = scaleKey.mValue.x;
				keySet.scaleKeys[k].key.y = scaleKey.mValue.y;
				keySet.scaleKeys[k].key.z = scaleKey.mValue.z;
			}
			for (unsigned int k = 0; k < keySet.numTransKeys ; ++k)
			{
				aiVectorKey &transKey = 	channel->mPositionKeys[k];
				keySet.transKeys[k].time = (float)(transKey.mTime / 1000.0);
				
				keySet.transKeys[k].key.x = transKey.mValue.x;
				keySet.transKeys[k].key.y = transKey.mValue.y;
				keySet.transKeys[k].key.z = transKey.mValue.z;
			}
		}
	}
}