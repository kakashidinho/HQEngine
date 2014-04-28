#include "App.h"

#define VERIFY_CODE 0

const float g_camOffsetPerSec = 100.f;//camra offset per second

struct MaterialArray {
	HQFloat4 materialDiffuse[7];
};

struct Subsplat {
	hquint32 level;//level of illumination texture
	hquint32 x;//position in illumination texture
	hquint32 y;//position in illumination texture
};

/*------------App----------------*/
App::App()
: BaseApp(WINDOW_SIZE, WINDOW_SIZE),
	m_vplsDim(16)
{
	//setup resources
	HQEngineApp::GetInstance()->AddFileSearchPath("../../Data");
	HQEngineApp::GetInstance()->AddFileSearchPath("../../Data/multires_splatting");

	//init resources
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile("msplat_resourcesCommon.script");
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile("msplat_resources.script");//API specific

	//create model
	m_model = new HQMeshNode(
		"cornell_box",
		"cornell_box.hqmesh",
		m_pRDevice,
		"depth-pass_vs",
		NULL);

	//init camera
	m_camera = new HQCamera(
		"camera",
		278, 273, 800,
		0, 1, 0,
		0, 0, -1,
		HQToRadian(37),
		1.0f,
		1.0f, 1000.0f,
		this->m_renderAPI_type
		);

	//create light object
	m_light = new DiffuseSpotLight(
		HQColorRGBA(0.8f, 0.65f, 0.23f, 1),
		343.f, 547.8f, -227.f,
		0, -1, 0,
		HQToRadian(70),
		HQToRadian(50),
		2.0f,
		1000.f,
		this->m_renderAPI_type);

	//add all to containers
	m_scene->AddChild(m_model.GetRawPointer());
	m_scene->AddChild(m_camera);
	m_scene->AddChild(&m_light->lightCam());

	//init render device
	m_pRDevice->SetClearColorf(0, 0, 0, 0);
	//m_pRDevice->GetStateManager()->SetFillMode(HQ_FILL_WIREFRAME);

	/*---------create and bind uniform buffers--*/
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(ModelViewInfo), true, &this->m_uniformViewInfoBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(MaterialArray), true, &this->m_uniformMaterialArrayBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hqint32), true, &this->m_uniformMaterialIndexBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(DiffuseLightProperties), true, &this->m_uniformLightProtBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(LightView), true, &this->m_uniformLightViewBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hquint32), true, &this->m_uniformRefineStepBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hquint32)* m_vplsDim * m_vplsDim, true, &m_uniformRSMSamplesBuffer);

	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 0, m_uniformViewInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 0, m_uniformViewInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 1, m_uniformLightViewBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 2, m_uniformMaterialArrayBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 2, m_uniformMaterialArrayBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 3, m_uniformMaterialIndexBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 4, m_uniformLightProtBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 10, m_uniformRefineStepBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 11, m_uniformRSMSamplesBuffer);

	//fill material array buffer
	MaterialArray * materials;
	m_uniformMaterialArrayBuffer->Map(&materials);
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i)
	{
		memcpy(&materials->materialDiffuse[i], &m_model->GetSubMeshInfo(i).colorMaterial.diffuse, sizeof(HQFloat4));
	}
	m_uniformMaterialArrayBuffer->Unmap();

	//init full screen quad buffer and layout
	this->InitFullscreenQuadBuffer();

	/*------init mipmap textures---------*/
	this->InitMipmaps();

	//init reflective shadow map's sample pattern map
	this->InitSamplePattern();

	//init subplat buffers
	this->InitSubplatsBuffers();

	//load effect
	HQEngineApp::GetInstance()->GetEffectManager()->AddEffectsFromFile("msplat_effects.script");

	//retrieve main effect
	m_effect = HQEngineApp::GetInstance()->GetEffectManager()->GetEffect("multires-splatting");
}

App::~App(){
}

void App::Update(HQTime dt){
	//update scene
	m_scene->SetUniformScale(0.01f);
	m_scene->Update(dt);

	//send scene data to shader
	ModelViewInfo * viewInfo;
	LightView * lightView;
	DiffuseLightProperties * lightProt;

	//send view info data
	m_uniformViewInfoBuffer->Map(&viewInfo);
	viewInfo->worldMat = m_model->GetWorldTransform();
	viewInfo->viewMat = m_camera->GetViewMatrix();
	viewInfo->projMat = m_camera->GetProjectionMatrix();
	m_camera->GetWorldPosition(viewInfo->cameraPosition);
	m_uniformViewInfoBuffer->Unmap();

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
	lightProt->lightFalloff_cosHalfAngle_cosHalfTheta[0] = m_light->falloff;
	lightProt->lightFalloff_cosHalfAngle_cosHalfTheta[1] = cosf(m_light->angle * 0.5f);
	lightProt->lightFalloff_cosHalfAngle_cosHalfTheta[2] = cosf(m_light->theta * 0.5f);
	m_uniformLightProtBuffer->Unmap();
}


void App::RenderImpl(HQTime dt){
	HQViewPort viewport = {0, 0, };
	//depth pass rendering
	viewport.width = RSM_DIM;
	viewport.height = RSM_DIM;
	
	m_effect->GetPassByName("depth-pass")->Apply();
	m_pRDevice->Clear(HQ_TRUE, HQ_TRUE, HQ_FALSE, HQ_TRUE);

	this->DrawScene();

	//g-buffer rendering

	m_effect->GetPassByName("gbuffer-pass")->Apply();
	m_pRDevice->SetFullViewPort();
	m_pRDevice->Clear(HQ_TRUE, HQ_TRUE, HQ_FALSE, HQ_TRUE);

	this->DrawScene();

	/*-------start preprocess step----------*/
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(NULL);

	//generate min-max mipmaps
	this->GenMipmaps();

	//refine list of subsplats
	this->RefineSubsplats();

	//multiresolution splatting
	this->MultiresSplat();

	//interpolating between illumination textures
	this->Upsample();

	//final pass, combine direct illumination with indirect illumination
	this->FinalPass();
}

void App::DrawScene()
{
	hqint32* materialIndex;
	//render the scene
	m_model->BeginRender();
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i){
		//send material index to shader
		m_uniformMaterialIndexBuffer->Map(&materialIndex);
		*materialIndex = i;
		m_uniformMaterialIndexBuffer->Unmap();

		m_model->DrawSubMesh(i);
	}
	m_model->EndRender();
}

void App::InitFullscreenQuadBuffer()
{
	//vertex layout
	HQVertexAttribDescArray<1> desc;
	desc.SetPosition(0, 0, 0, HQ_VADT_FLOAT2);
	HQEngineApp::GetInstance()->GetResourceManager()
		->CreateVertexInputLayout(desc, 1, "final-pass_vs", &m_fullscreenQuadVertLayout);

	//vertex buffer
	float verts[] = {
		-1, 1, 1, 1,
		-1, -1, 1, -1
	};

	m_pRDevice->GetVertexStreamManager()->CreateVertexBuffer(verts, sizeof(verts), false, false, &m_fullscreenQuadBuffer);
}

void App::InitSamplePattern()
{
	//store vpl sample pattern size in buffer
	hquint32 * pattern_sizes = NULL;
	m_uniformRSMSamplesBuffer->Map(&pattern_sizes);
	pattern_sizes[0] = pattern_sizes[1] = m_vplsDim;
	m_uniformRSMSamplesBuffer->Unmap();
	//------------
	HQRawPixelBuffer* pixelBuffer = m_pRDevice->GetTextureManager()->CreatePixelBuffer(HQ_RPFMT_R32G32_FLOAT, m_vplsDim, m_vplsDim);

	//uniform grid
	double ds = 1.0 / double(m_vplsDim);

	for (hquint32 i = 0; i < m_vplsDim; ++i)
	{
		for (hquint32 j = 0; j < m_vplsDim; ++j){
			double du = randd(0.0, ds);
			double dv = randd(0.0, ds);

			float u = (float)(ds * i + du);
			float v = (float)(ds * j + dv);

			pixelBuffer->SetPixelf(i, j, u, v, 0.f, 0.f);
		}
	}
	
	//now create texture
	m_pRDevice->GetTextureManager()->AddTexture(pixelBuffer, false, &m_samplePatternTexture);

	//add to resource manager
	HQEngineApp::GetInstance()->GetResourceManager()->AddTextureResource("rsm_sample_map", m_samplePatternTexture);

	pixelBuffer->Release();
}


void App::InitMipmaps()
{
	hquint32 size = MIN_MAX_MIPMAP_FIRST_SIZE;
	for (hqint32 i = NUM_RESOLUTIONS - 2 ; i >= 0; --i)
	{
		
		//min version
		m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R32G32B32A32_FLOAT, size, size, false, &m_mipmapMin[i]);

		//max version
		m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R32G32B32A32_FLOAT, size, size, false, &m_mipmapMax[i]);

		size >>= 1;
	}
}

//dynamically generate mipmap
void App::GenMipmaps()
{
	//thread group size
	const hquint32 threadGroupDim = 16;
	//mipmap size
	hquint32 width = MIN_MAX_MIPMAP_FIRST_SIZE, height = MIN_MAX_MIPMAP_FIRST_SIZE;

	//texture containing dinstance to camera
	HQEngineTextureResource* depthMap = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("depth_materialID_img");
	//texture containing world space normal	
	HQEngineTextureResource* normalMap = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("world_normal_img");
	
	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("mipmap_cs_1st_step")->GetShader();

	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

	/*-----------first step: generate first mipmap level----------------*/
	
	//read from deapth and world space g-buffer
	m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, depthMap->GetTexture());
	m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, normalMap->GetTexture());

	//output of compute shader
	m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(2, m_mipmapMin[NUM_RESOLUTIONS - 2]);
	m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(3, m_mipmapMax[NUM_RESOLUTIONS - 2]);

	m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

	m_pRDevice->TextureUAVBarrier();

	/*-------------generate the rest, starting from second hishest mipmap level----------------*/
	hqint32 level = NUM_RESOLUTIONS - 3;
	width >>= 1;
	height >>= 1;
	
	shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("mipmap_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
	while (level >= 0 && width > 0){
		//output of compute shader
		m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(2, m_mipmapMin[level]);
		m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(3, m_mipmapMax[level]);

		//read from previous mipmap level
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, m_mipmapMin[level + 1]);
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, m_mipmapMax[level + 1]);


		m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

		m_pRDevice->TextureUAVBarrier();

		width >>= 1;
		height >>= 1;
		level--;
	}
}


void App::InitSubplatsBuffers(){
	const hquint32 dispatchGroupThreads = 64;

	//coarsest subplats buffer 
	const hquint32 coarsestSize = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);
	{
		Subsplat* subsplats = new Subsplat[coarsestSize * coarsestSize];
		for (hquint32 i = 0; i < coarsestSize; ++i)
		{
			for (hquint32 j = 0; j < coarsestSize; ++j)
			{
				Subsplat & subsplat = subsplats[j * coarsestSize + i];

				subsplat.level = 0;
				subsplat.x = i;
				subsplat.y = j;
			}
		}

		m_pRDevice->GetShaderManager()->CreateBufferUAV(subsplats, sizeof(Subsplat), coarsestSize * coarsestSize, &m_subplatsRefineStepBuffer[0]);

		delete[] subsplats;
	}

	hquint32 size = coarsestSize;
	//the first element of this array is final subsplats list count, the subsequent ones are counts for each refinement step
	m_initialSubsplatsCounts[0] = 0;
	//the first element of this array is indirect illumination's dispatch's arguments, 
	//the subsequent ones are arguments for refinement steps' dispatchs
	m_initialDispatchArgs[0].Set(1);

	for (hquint32 i = 0; i < NUM_RESOLUTIONS; ++i)
	{
		//initial subsplats count for each step. Initially, the coarsest res has all of its subsplats,
		//while other has zero
		if (i == 0)
		{
			m_initialSubsplatsCounts[i + 1] = coarsestSize * coarsestSize;
			m_initialDispatchArgs[i + 1].Set(max(m_initialSubsplatsCounts[i + 1] / dispatchGroupThreads, 1));
		}
		else if(i < NUM_RESOLUTIONS - 1)
		{
			m_initialSubsplatsCounts[i + 1] = 0;
			m_initialDispatchArgs[i + 1].Set(1);
			
			m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(Subsplat), size * size, &m_subplatsRefineStepBuffer[i]);
		}
		
		//illumination texture
		m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R8G8B8A8_UNORM, size, size, false, &m_subplatsIllumTexture[i]);
		//interpolated illumination texture
		m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R8G8B8A8_UNORM, size, size, false, &m_subplatsInterpolatedTexture[i]);

		size <<= 1;//next finer resolution
	}

	//register final interpolated illumination texture to be used by effect manager
	HQEngineApp::GetInstance()->GetResourceManager()->AddTextureResource("final_interpolated_illum_img", m_subplatsInterpolatedTexture[NUM_RESOLUTIONS - 1]);

	//create buffer containing subsplats' counts.
	//the first element is final subsplats list count, the subsequent ones are counts for each refinement step
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(hquint32), 
		sizeof(m_initialSubsplatsCounts) / sizeof(hquint32), 
		&m_subsplatsCountBuffer);

	//create buffer containing dispatch arguments for indirect illumination and refinement steps
	m_pRDevice->GetShaderManager()->CreateComputeIndirectArgs(NULL,  
		sizeof(m_initialDispatchArgs) / sizeof(DispatchComputeArgs),
		&m_dispatchArgsBuffer);

	//create buffer containing final subsplats list
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(Subsplat), WINDOW_SIZE * WINDOW_SIZE, &m_finalSubsplatsBuffer);

}

void App::RefineSubsplats()
{
	const hquint32 coarsestSize = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);
	const hquint32 numCoarsestSubsplats = coarsestSize * coarsestSize;

	
	m_subsplatsCountBuffer->Update(m_initialSubsplatsCounts);//reset subsplats' counts
	m_dispatchArgsBuffer->Update(m_initialDispatchArgs);//reset dispatch arguments

#if VERIFY_CODE
	{
		//verify buffer content
		hquint32 subsplatCountsBuffercontent[sizeof(m_initialSubsplatsCounts) / sizeof(hquint32)];
		DispatchComputeArgs dispatchArgsBuffercontent[sizeof(m_initialDispatchArgs) / sizeof(DispatchComputeArgs)];

		m_subsplatsCountBuffer->CopyContent(subsplatCountsBuffercontent);
		m_dispatchArgsBuffer->CopyContent(dispatchArgsBuffercontent);
	}
#endif
	
	//activate shader
	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("subsplat_refinement_cs")->GetShader();

	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

	//subsplat count buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(0, m_subsplatsCountBuffer);

	//final subsplats list buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(3, m_finalSubsplatsBuffer);

	//indirect compute dispatch's arguments
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(4, m_dispatchArgsBuffer);

	hquint32 *pCurrentLevel;

	for (hquint32 level = 0; level < NUM_RESOLUTIONS - 1; ++level)
	{
		m_uniformRefineStepBuffer->Map(&pCurrentLevel);
		*pCurrentLevel = level;
		m_uniformRefineStepBuffer->Unmap();

		//mipmap containing min & max depth & normal for finer level
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, m_mipmapMin[level]);
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, m_mipmapMax[level]);
	
		
		if (level < NUM_RESOLUTIONS - 2)
		{
			//finer subsplats buffer
			m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(2, m_subplatsRefineStepBuffer[level + 1]);
		}
		else
		{
			//final step
			shader = HQEngineApp::GetInstance()->GetResourceManager()
				->GetShaderResource("subsplat_final_refinement_cs")->GetShader();
			m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

			//no finer list's buffer is used, instead we will push to final list
			m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(2, NULL);
		}
		
		//current subsplats buffer
		m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(1, m_subplatsRefineStepBuffer[level]);

		//dispatch
		m_pRDevice->DispatchComputeIndirect(m_dispatchArgsBuffer, level + 1);

		m_pRDevice->BufferUAVBarrier();
	}
}

void App::MultiresSplat()
{
	//set output textures
	for (hquint32 level = 0; level < NUM_RESOLUTIONS; ++level)
	{
		m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(level, m_subplatsIllumTexture[level]);
	}

	//number of subsplats in final list
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(NUM_RESOLUTIONS, m_subsplatsCountBuffer, 0, 0);
	//final subsplats list
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(NUM_RESOLUTIONS + 1, m_finalSubsplatsBuffer, 0, 0);

	/*---------------clear illuimination textures first------------------*/
	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("clear_illum_textures_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
	//thread group size
	const hquint32 threadGroupDim = 16;

	m_pRDevice->DispatchCompute(WINDOW_SIZE / threadGroupDim, WINDOW_SIZE / threadGroupDim, 1);
	m_pRDevice->TextureUAVBarrier();

	/*------------now do multiresolution splatting-----------------*/
	m_effect->GetPassByName("multires_splatting")->Apply();

	m_pRDevice->DispatchComputeIndirect(m_dispatchArgsBuffer);
	m_pRDevice->TextureUAVBarrier();

#if 0
	//verify textures
	HQFloat4 texels[16 * 16];
	m_subplatsIllumTexture[0]->CopyTextureContent(texels);

#endif
}

void App::Upsample()
{
	//thread group size
	const hquint32 threadGroupDim = 16;
	//mipmap size
	hquint32 width = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1), height = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);


	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("upsample_cs_1st_step")->GetShader();

	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

	/*-----------first step: generate first interpolated level----------------*/

	//read from first illumination texture
	m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, m_subplatsIllumTexture[0]);

	//output of compute shader
	m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(0, m_subplatsInterpolatedTexture[0]);

	m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

	m_pRDevice->TextureUAVBarrier();

	/*-------------upsample the rest, starting from second lowest level----------------*/
	hqint32 level = 1;
	width <<= 1;
	height <<= 1;

	shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("upsample_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
	while (level < NUM_RESOLUTIONS){
		//output of compute shader
		m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(0, m_subplatsInterpolatedTexture[level]);

		//read from current level's illumination texture computed  in previous pass
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, m_subplatsIllumTexture[level]);
		//read from lower level's interpolated texture
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, m_subplatsInterpolatedTexture[level - 1]);


		m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

		m_pRDevice->TextureUAVBarrier();

		width <<= 1;
		height <<= 1;
		level++;
	}
}

void App::FinalPass()
{
	//combine indirect illumination with direct illumination
	m_effect->GetPassByName("final-pass")->Apply();
	m_pRDevice->GetVertexStreamManager()->SetVertexBuffer(m_fullscreenQuadBuffer, 0, 2 * sizeof(float));
	m_pRDevice->GetVertexStreamManager()->SetVertexInputLayout(m_fullscreenQuadVertLayout);
	m_pRDevice->SetPrimitiveMode(HQ_PRI_TRIANGLE_STRIP);

	m_pRDevice->Clear(HQ_TRUE, HQ_TRUE, HQ_FALSE, HQ_TRUE);
	m_pRDevice->Draw(4, 0);
}

void App::KeyPressed(HQKeyCodeType keyCode)
{
	switch (keyCode)
	{
	case HQKeyCode::A:
		m_cameraTransition.x = g_camOffsetPerSec;//start moving in X direction
		break;
	case HQKeyCode::D:
		m_cameraTransition.x = -g_camOffsetPerSec;//start moving in X direction
		break;
	case HQKeyCode::W:
		m_cameraTransition.z = -g_camOffsetPerSec;//start moving in Z direction
		break;
	case HQKeyCode::S:
		m_cameraTransition.z = g_camOffsetPerSec;//start moving in Z direction
		break;
	}
}


void App::KeyReleased(HQKeyCodeType keyCode)
{
	switch (keyCode)
	{
	case HQKeyCode::Q:
		m_light->theta += 0.1f;
		if (m_light->theta >= m_light->angle)
			m_light->theta = m_light->angle - HQToRadian(2);
		break;
	case HQKeyCode::E:
		m_light->theta -= 0.1f;
		if (m_light->theta <= 0.f)
			m_light->theta = HQToRadian(2);
		break;
	case HQKeyCode::ESCAPE:
		HQEngineApp::GetInstance()->Stop();
		break;
	case HQKeyCode::A:
		if (m_cameraTransition.x > 0)
			m_cameraTransition.x = 0;//stop moving in X direction
		break;
	case HQKeyCode::D:
		if (m_cameraTransition.x < 0)
			m_cameraTransition.x = 0;//stop moving in X direction
		break;
	case HQKeyCode::W:
		if (m_cameraTransition.z < 0)
			m_cameraTransition.z = 0;//stop moving in Z direction
		break;
	case HQKeyCode::S:
		if (m_cameraTransition.z > 0)
			m_cameraTransition.z = 0;//stop moving in Z direction
		break;
	}
}