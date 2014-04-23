/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "RenderLoop.h"

#define DEBUG_DX 0

#if _MSC_VER >= 1800 && defined _DEBUG && defined USE_D3D11 && DEBUG_DX
#	define DONT_SAVE_VSGLOG_TO_TEMP
#	define VSG_DEFAULT_RUN_FILENAME L"graphics-capture.vsglog"
//#	define VSG_NODEFAULT_INSTANCE
#	include <vsgcapture.h> 
#	define DX_FRAME_CAPTURE
#endif


/*----------RenderLoop----------*/
//constructor
RenderLoop::RenderLoop(const char* rendererAPI,
					HQLogStream *logStream,
					const char * additionalAPISettings)
					: BaseApp(rendererAPI, logStream, additionalAPISettings)
{
	bool use_compute_shader = false;

	/*-------loading resources--------*/
	char apiResNamXML[256];
	if (strcmp(m_renderAPI_name, "GL") == 0)
	{
#ifndef HQ_USE_CUDA
		if (this->m_pRDevice->IsShaderSupport(HQ_COMPUTE_SHADER, "4.3"))
		{
			use_compute_shader = true;
			strcpy(apiResNamXML, "rsm_resourcesGL_compute_shader.script");
		}
		else
#endif
			strcpy(apiResNamXML, "rsm_resourcesGL.script");
	}
	else if (strcmp(m_renderAPI_name, "D3D11") == 0)
	{
#ifndef HQ_USE_CUDA
		if (this->m_pRDevice->IsShaderSupport(HQ_COMPUTE_SHADER, "5.0"))
		{
			use_compute_shader = true;
			strcpy(apiResNamXML, "rsm_resourcesD3D11_compute_shader.script");
		}
		else
#endif
			strcpy(apiResNamXML, "rsm_resourcesD3D11.script");
	}
	else
	{
		strcpy(apiResNamXML, "rsm_resourcesD3D9.script");
	}

	//add resource search paths
	HQEngineApp::GetInstance()->AddFileSearchPath("../Data");
	HQEngineApp::GetInstance()->AddFileSearchPath("../Data/reflective_shadow_map");
	HQEngineApp::GetInstance()->AddFileSearchPath("../../Data");
	HQEngineApp::GetInstance()->AddFileSearchPath("../../Data/reflective_shadow_map");


	//init resources
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile(apiResNamXML);
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile("rsm_resourcesCommon.script");
	HQEngineApp::GetInstance()->GetEffectManager()->AddEffectsFromFile("rsm_effects.script");

	//retrieve main effect
	rsm_effect = HQEngineApp::GetInstance()->GetEffectManager()->GetEffect("rsm");

	//create model
	m_model = new HQMeshNode(
		"cornell_box", 
		"cornell_box.hqmesh",
		m_pRDevice,
		"final-gathering_vs",
		NULL);

	//init camera
	m_camera = new HQCamera(
			"camera",
			278,273, 800,
			0, 1, 0,
			0, 0, -1,
			HQToRadian(37),
			1.0f,
			1.0f, 1000.0f,
			this->m_renderAPI_type
		);

	//create light object
	m_light = new DiffuseSpotLight(
			HQColorRGBA(1, 1, 1, 1),
			343.f, 547.8f , -227.f,
			0, -1, 0,
			HQToRadian(90),
			2.0f,
			1000.f,
			this->m_renderAPI_type);

	//add all to containers
	m_scene->AddChild(m_model);
	m_scene->AddChild(m_camera);
	m_scene->AddChild(&m_light->lightCam());

#ifdef HQ_USE_CUDA
	CudaGenerateNoiseMap();
#else
	//decode noise map from RGBA image to float texture
	if (use_compute_shader)
		ComputeShaderDecodeNoiseMap();
	else
		DecodeNoiseMap();
#endif

	//init render device
	m_pRDevice->SetClearColorf(1, 1, 1, 1);
	//m_pRDevice->GetStateManager()->SetFillMode(HQ_FILL_WIREFRAME);

	/*---------create and bind uniform buffers--*/
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(Transform), NULL, true, &this->m_uniformTransformBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(DiffuseMaterial), NULL, true, &this->m_uniformMaterialBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(DiffuseLightProperties), NULL, true, &this->m_uniformLightProtBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(LightView), NULL, true, &this->m_uniformLightViewBuffer);

	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 0, m_uniformTransformBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 1, m_uniformLightViewBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 2, m_uniformMaterialBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 3, m_uniformLightProtBuffer);


}

//destructor
RenderLoop::~RenderLoop()
{
	delete m_model;
	delete m_light;
}

#ifdef HQ_USE_CUDA

extern "C" void cudaGenerateNoiseMapKernel(cudaArray_t outputArray, unsigned int width, unsigned int height);

void RenderLoop::CudaGenerateNoiseMap() //generate noise map using cuda
{
	HQEngineTextureResource* hqNoiseMapRes = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("decoded_random_factors_img");
	hquint32 width, height;
	hqNoiseMapRes->GetTexture2DSize(width, height);

	//setup cuda
	cudaError_t err;
	cudaGraphicsResource_t cudaNoiseMapRes;

	cudaArray_t cudaNoiseMapArray;
	//set device and read properties
	{
		cudaDeviceProp prop;
		err = cudaSetDevice(0);
		err = cudaGetDeviceProperties(&prop, 0);

		char buffer[512];
		sprintf(buffer, "cuda device '%s'\n\tcompute capability = %d.%d\n", prop.name, prop.major, prop.minor);
#ifdef _MSC_VER
		OutputDebugStringA(buffer);
#else
		printf(buffer);
#endif
	}

	err = HQCudaBinding::cudaGraphicsSetDevice(m_pRDevice);
	err = HQCudaBinding::cudaGraphicsRegisterResource(&cudaNoiseMapRes, hqNoiseMapRes, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	//map cuda resource
	err = cudaGraphicsMapResources(1, &cudaNoiseMapRes);
	//get cuda resource arrray
	err = cudaGraphicsSubResourceGetMappedArray(&cudaNoiseMapArray, cudaNoiseMapRes, 0, 0);
	
	//now run cuda kernel
	cudaGenerateNoiseMapKernel(cudaNoiseMapArray, width, height);

	err = cudaGraphicsUnmapResources(1, &cudaNoiseMapRes);

	//clean up
	err = cudaGraphicsUnregisterResource(cudaNoiseMapRes);

	err = cudaDeviceReset();
}
#else
void RenderLoop::DecodeNoiseMap()//decode noise map from RGBA image to float texture
{
	hquint32 width, height;//encoded noise map's size
	//encoded noise map
	HQEngineTextureResource *encoded_noise_map = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("random_factors_img");
	HQVertexBuffer* vBuffer;
	HQVertexLayout* vInputLayout;//vertex buffer and input layout

	encoded_noise_map->GetTexture2DSize(width, height);

	//create screen quad vertex buffer. vertex data contains width and height of noise map. This info is needed for half pixel offset in Direct3D9 device
	float vertices[] = {
		-1, 1, (float)width, (float)height, 0, 0,
		1, 1, (float)width, (float)height, 1, 0,
		-1, -1, (float)width, (float)height, 0, 1,
		1, -1, (float)width, (float)height, 1, 1
	};

	m_pRDevice->GetVertexStreamManager()->CreateVertexBuffer(vertices, sizeof(vertices), false, false, &vBuffer);

	//create vertex input layout
	HQVertexAttribDescArray<2> vAttrDescs;
	vAttrDescs.SetPosition(0, 0, 0, HQ_VADT_FLOAT4);
	vAttrDescs.SetTexcoord(1, 0, 4 * sizeof(float), HQ_VADT_FLOAT2, 0);
	HQEngineApp::GetInstance()->GetResourceManager()->CreateVertexInputLayout(vAttrDescs, 2, "noise_decoding_vs", &vInputLayout);
	
	//now begin the decoding process. read the encoded noise factors from texture and render the decoded factors to a float render target.
	HQEngineRenderEffect* effect = HQEngineApp::GetInstance()->GetEffectManager()->GetEffect("decode_random_factors");
	effect->GetPass(0)->Apply();

	m_pRDevice->GetVertexStreamManager()->SetVertexBuffer(vBuffer, 0, 6 * sizeof(float));
	m_pRDevice->GetVertexStreamManager()->SetVertexInputLayout(vInputLayout);

	m_pRDevice->SetPrimitiveMode(HQ_PRI_TRIANGLE_STRIP);
	//draw full screen quad
	HQViewPort vp = {0, 0, width, height};
	m_pRDevice->SetViewPort(vp);
	m_pRDevice->BeginRender(HQ_TRUE, HQ_FALSE, HQ_FALSE, HQ_TRUE);
	m_pRDevice->DrawPrimitive(2, 0);
	m_pRDevice->EndRender();

	//clean up
	m_pRDevice->GetVertexStreamManager()->RemoveVertexBuffer(vBuffer);
	HQEngineApp::GetInstance()->GetResourceManager()->RemoveTextureResource(encoded_noise_map);
	HQEngineApp::GetInstance()->GetEffectManager()->RemoveEffect(effect);	

	//for debugging
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(NULL);
	m_pRDevice->DisplayBackBuffer();
}
void RenderLoop::ComputeShaderDecodeNoiseMap()
{
#ifdef DX_FRAME_CAPTURE
	VsgDbg vsgDbg(true);
	vsgDbg.BeginCapture();
#endif

	HQEngineTextureResource * encoded = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("random_factors_img");
	HQEngineTextureResource * decoded = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("decoded_random_factors_img");
	HQEngineShaderResource* computeShader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("noise_decoding_cs");

	hquint32 width, height;
	encoded->GetTexture2DSize(width, height);

	if (m_renderAPI_type == HQ_RA_OGL)
	{
		//set encoded texture
		m_pRDevice->GetTextureManager()->SetTexture(0, encoded->GetTexture());
		//set decoded texture UAV
		m_pRDevice->GetTextureManager()->SetTextureUAV(0, decoded->GetTexture());
	}
	else{
		//set encoded texture
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER | 0, encoded->GetTexture());
		//set decoded texture UAV
		m_pRDevice->GetTextureManager()->SetTextureUAV(HQ_COMPUTE_SHADER | 0, decoded->GetTexture());
	}

	//active compute shader
	m_pRDevice->GetShaderManager()->ActiveComputeShader(computeShader->GetShader());

	//run compute shader
	m_pRDevice->DispatchCompute(1, 1, 1);

	//clean up
	if (m_renderAPI_type == HQ_RA_OGL)
	{
		m_pRDevice->GetTextureManager()->SetTexture(0, NULL);
	}
	else{
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER | 0, NULL);
	}
	m_pRDevice->GetShaderManager()->ActiveComputeShader(NULL);
	HQEngineApp::GetInstance()->GetResourceManager()->RemoveTextureResource(encoded);
	HQEngineApp::GetInstance()->GetResourceManager()->RemoveShaderResource(computeShader);

	m_pRDevice->TextureUAVBarrier();

#ifdef DX_FRAME_CAPTURE
	vsgDbg.EndCapture();
#endif
}

#endif//#ifdef HQ_USE_CUDA

//rendering loop
void RenderLoop::Update(HQTime dt)
{
	//update scene
	m_scene->SetUniformScale(0.01f);
	m_scene->Update(dt);

	//send scene data to shader
	Transform * transform;
	LightView * lightView;
	DiffuseLightProperties * lightProt;

	//send transform data
	m_uniformTransformBuffer->Map(&transform);
	transform->worldMat = m_model->GetWorldTransform();
	transform->viewMat = m_camera->GetViewMatrix();
	transform->projMat = m_camera->GetProjectionMatrix();
	m_uniformTransformBuffer->Unmap();

	//send light camera's matrices data
	m_uniformLightViewBuffer->Map(&lightView);
	lightView->lightViewMat = m_light->lightCam().GetViewMatrix();
	lightView->lightProjMat = m_light->lightCam().GetProjectionMatrix();
	m_uniformLightViewBuffer->Unmap();

	//send light properties data
	m_uniformLightProtBuffer->Map(&lightProt);
	memcpy(lightProt->lightPosition, &m_light->position(), sizeof(HQVector4));
	memcpy(lightProt->lightDirection, &m_light->direction(), sizeof(HQVector4));
	memcpy(lightProt->lightDiffuse, &m_light->diffuseColor, sizeof(HQVector4));
	lightProt->lightFalloff_lightPCosHalfAngle[0] = m_light->falloff;
	lightProt->lightFalloff_lightPCosHalfAngle[1] = pow(cosf(m_light->angle * 0.5f), m_light->falloff);
	m_uniformLightProtBuffer->Unmap();
}

void RenderLoop::RenderImpl(HQTime dt){
	//depth pass rendering
	DepthPassRender(dt);

	//low resolution rendering pass
	LowresPassRender(dt);

	//final rendering pass
	FinalPassRender(dt);
}