#include "MeshX.h"

#include <string.h>
#include <map>

#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9.lib")

LRESULT CALLBACK DummyWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	return DefWindowProc(hwnd, message, wParam, lParam);
}

MeshX::MeshX(const char *xFileName)
{
	this->mesh = NULL;
	HRESULT hr;
	animCtrl = 0;

	CreateWindowHandle();
	CreateDirect3dDevice();

	
	allocMeshHierarchy.setDevice(pDevice);

	//try to load skinned mesh
	hr = D3DXLoadMeshHierarchyFromXA(xFileName, D3DXMESH_SYSTEMMEM,
		pDevice, &allocMeshHierarchy, 0, /* ignore user data */ 
		(D3DXFRAME**)&root,	&animCtrl);

	if (FAILED(hr))
	{
		throw std::bad_alloc();
	}

	skinInfo = allocMeshHierarchy.GetMeshList().front()->pSkinInfo;

	BuildSkinnedMesh();

	boneMatrices = new D3DXMATRIX[skinInfo->GetNumBones()];
	boneMatrices3x4 = HQMatrix3x4::NewArray(skinInfo->GetNumBones());

}

MeshX::~MeshX()
{
	//clean up
	delete[] boneMatrices;
	delete[] boneMatrices3x4;

	pDevice->Release();
	pD3D->Release();

	SafeRelease(this->mesh);
	
	D3DXFrameDestroy(root, &allocMeshHierarchy);
	SafeRelease(animCtrl);

	DestroyWindow(hwnd);
}


void MeshX::Update()
{
	UpdateNode(root, NULL);

	
	for (DWORD i = 0; i < skinInfo->GetNumBones(); ++i)
	{
		FrameEx * node = allocMeshHierarchy.GetNode(skinInfo->GetBoneName(i));
		boneMatrices[i] = *skinInfo->GetBoneOffsetMatrix(i) * node->toRoot;

		D3DXMatrixTranspose(&boneMatrices[i], &boneMatrices[i]);
		memcpy(&boneMatrices3x4[i], &boneMatrices[i], sizeof(HQMatrix3x4));
	}
}

void MeshX::UpdateNode(FrameEx *node, FrameEx *parent)
{
	if (node == NULL)
		return;

	if (parent != NULL)
		node->toRoot = node->TransformationMatrix * parent->toRoot;
	else
		node->toRoot = node->TransformationMatrix;

	UpdateNode((FrameEx *)node->pFrameSibling, (FrameEx *)parent);
	UpdateNode((FrameEx *)node->pFrameFirstChild, (FrameEx *)node);
}

void MeshX::CreateWindowHandle()
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

void MeshX::CreateDirect3dDevice()
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

void MeshX::BuildSkinnedMesh()
{
	ID3DXMesh* oldMesh = allocMeshHierarchy.GetMeshList().front()->MeshData.pMesh;

	//compute normal
	D3DVERTEXELEMENT9 elements[MAX_FVF_DECL_SIZE];
	oldMesh->GetDeclaration(elements);

	ID3DXMesh* tempMesh = 0;
	ID3DXMesh* tempOpMesh = 0;
	oldMesh->CloneMesh(D3DXMESH_SYSTEMMEM, elements, pDevice, &tempMesh);
	 
	if( !HasNormals(tempMesh) )
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
	skinInfo->Remap(tempOpMesh->GetNumVertices(), 
		(DWORD*)remap->GetBufferPointer());
	SafeRelease(remap); // Done with remap info.

	DWORD        numBoneComboEntries = 0;
	ID3DXBuffer* boneComboTable      = 0;
	DWORD maxVertInfluences;
	skinInfo->ConvertToIndexedBlendedMesh(tempOpMesh, 0,  
		128, 0, 0, 0, 0, &maxVertInfluences,
		&numBoneComboEntries, &boneComboTable, &this->mesh);

	SafeRelease(tempOpMesh);
	SafeRelease(boneComboTable);
	delete[] adj;
}

bool MeshX::HasNormals(ID3DXMesh* mesh)
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

