#include "App.h"
#include "../../BaseSource/lodepng.h"

#include <string>
#include <sstream>

#define VERIFY_CODE 0

const float g_camOffsetPerSec = 100.f;//camra offset per second


struct Subsplat {
	hquint32 level;//level of illumination texture
	hquint32 x;//position in illumination texture
	hquint32 y;//position in illumination texture
};

/*------------App----------------*/
App::App()
: BaseApp("Core-GL4.3", WINDOW_SIZE, WINDOW_SIZE),
	m_vplsDim(16),
	m_giOn(true),
	m_diffuseScene(false),
	m_dynamicLight(true)
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
		"cornell_box_spheres.hqmesh",
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
	m_light = new SpecularSpotLight(
		HQColorRGBA(0.5f, 0.5f, 0.5f, 1),//ambient
		HQColorRGBA(1.0f, 0.85f, 0.43f, 1),//diffuse
		HQColorRGBA(1.0f, 0.85f, 0.43f, 1),//specular
		//HQColorRGBA(1.0f, 1.f, 1.f, 1),//diffuse
		//HQColorRGBA(1.0f, 1.f, 1.f, 1),//specular
		300.f, 547.8f, -180.f,
		0.05f, -1, -0.1f,
		HQToRadian(70),
		HQToRadian(60),
		2.0f,
		800.f,
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
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, 7 * sizeof(SpecularMaterial), true, &this->m_uniformMaterialArrayBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hqint32), true, &this->m_uniformMaterialIndexBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(SpecularLightProperties), true, &this->m_uniformLightProtBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(LightView), true, &this->m_uniformLightViewBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hquint32), true, &m_uniformInterpolatedInfoBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hquint32), true, &this->m_uniformLevelInfoBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(hquint32)* m_vplsDim * m_vplsDim, true, &m_uniformRSMSamplesBuffer);
	m_subsplatsRefineThreshold[0] = 0.06f; m_subsplatsRefineThreshold[1] = 0.2f;//depth and normal thresholds
	m_subsplatsRefineThreshold[2] = 0.002f;//illumination threshold
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(m_subsplatsRefineThreshold, sizeof(m_subsplatsRefineThreshold), true, &m_uniformRefineThresholdBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(Float2), true, &m_uniformVplSampleCoordsBuffer);

	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 0, m_uniformViewInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 0, m_uniformViewInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 0, m_uniformViewInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 1, m_uniformLightViewBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 2, m_uniformMaterialArrayBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 2, m_uniformMaterialArrayBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 3, m_uniformMaterialIndexBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER, 4, m_uniformLightProtBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 9, m_uniformInterpolatedInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 10, m_uniformLevelInfoBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 11, m_uniformRSMSamplesBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 12, m_uniformRefineThresholdBuffer);
	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_COMPUTE_SHADER, 13, m_uniformVplSampleCoordsBuffer);

	//fill material array buffer
	SpecularMaterial * material;
	m_uniformMaterialArrayBuffer->Map(&material);
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i)
	{
		memcpy(&material[i].materialAmbient, &m_model->GetSubMeshInfo(i).colorMaterial.ambient, sizeof(HQFloat4));
		memcpy(&material[i].materialDiffuse, &m_model->GetSubMeshInfo(i).colorMaterial.diffuse, sizeof(HQFloat4));
		memcpy(&material[i].materialSpecular, &m_model->GetSubMeshInfo(i).colorMaterial.specular, sizeof(HQFloat4));
		material[i].materialShininess = m_model->GetSubMeshInfo(i).colorMaterial.power;
	}
	m_uniformMaterialArrayBuffer->Unmap();

	//init full screen quad buffer and layout
	this->InitFullscreenQuadBuffer();

	//init reflective shadow map's sample pattern map
	this->InitSamplePattern();

	//init subplat buffers
	this->InitSubplatsBuffers();

	//init technique for diffuse scene
	this->InitDiffuse();

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

	if (m_dynamicLight)
		m_light->lightCam().RotateY(dt);

	m_scene->Update(dt);

	//send scene data to shader
	ModelViewInfo * viewInfo;
	LightView * lightView;
	SpecularLightProperties * lightProt;

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
	memcpy(lightProt->lightAmbient, &m_light->ambientColor, sizeof(HQVector4));
	memcpy(lightProt->lightDiffuse, &m_light->diffuseColor, sizeof(HQVector4));
	memcpy(lightProt->lightSpecular, &m_light->specularColor, sizeof(HQVector4));
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

	if (m_giOn)
	{
		m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(NULL);

		if (m_diffuseScene)
			this->GatherIndirectDiffuseIllum();
		else
			this->GatherIndirectGlossyIllum();
	}//if (m_giOn)

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
	m_samplePattern = new Float2[m_vplsDim * m_vplsDim];

	//uniform grid
	double ds = 1.0 / double(m_vplsDim);

	for (hquint32 i = 0, sampleIdx = 0; i < m_vplsDim; ++i)
	{
		for (hquint32 j = 0; j < m_vplsDim; ++j, ++sampleIdx){
			double du = randd(0.0, ds);
			double dv = randd(0.0, ds);

			float u = (float)(ds * i + du);
			float v = (float)(ds * j + dv);

			m_samplePattern[sampleIdx].x = u;
			m_samplePattern[sampleIdx].y = v;
		}
	}
	
	//now create texture
	m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R32G32_FLOAT, m_vplsDim, m_vplsDim, false, &m_samplePatternTexture);
	m_samplePatternTexture->SetLevelContent(0, m_samplePattern.GetRawPointer());

	//add to resource manager
	HQEngineApp::GetInstance()->GetResourceManager()->AddTextureResource("rsm_sample_map", m_samplePatternTexture);
}




void App::InitSubplatsBuffers(){
	const hquint32 dispatchGroupThreads = 64;
	const hquint32 coarsestSize = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);

	hquint32 totalSizeExceptLastLevel = 0;
	hquint32 size = coarsestSize;
	hquint32 initialSubsplatsCounts[NUM_RESOLUTIONS];//initial total subsplats count and subsplats count for each refinement step
	DispatchComputeArgs initialDispatchArgs[NUM_RESOLUTIONS];//initial dispatch arguments for indirect illumination step and refinement steps
	//the first element of this array is final subsplats list count, the subsequent ones are counts for each refinement step
	initialSubsplatsCounts[0] = 0;
	//the first element of this array is indirect illumination's dispatch's arguments, 
	//the subsequent ones are arguments for refinement steps' dispatchs
	initialDispatchArgs[0].Set(1);

	for (hquint32 i = 0; i < NUM_RESOLUTIONS - 1; ++i)
	{
		//initial subsplats count for each step. Initially, the coarsest res has all of its subsplats,
		//while other has zero
		if (i == 0)
		{
			initialSubsplatsCounts[i + 1] = coarsestSize * coarsestSize;
			initialDispatchArgs[i + 1].Set(max(initialSubsplatsCounts[i + 1] / dispatchGroupThreads, 1));
		}
		else
		{
			initialSubsplatsCounts[i + 1] = 0;
			initialDispatchArgs[i + 1].Set(1);
		}
		
		size <<= 1;//next finer resolution
		totalSizeExceptLastLevel += size * size;//total number of subsplats except finest resolution
	}

	m_totalSubsplats = totalSizeExceptLastLevel + WINDOW_SIZE * WINDOW_SIZE;
	
	//create temp illumination buffer
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(hquint32), WINDOW_SIZE * WINDOW_SIZE, &m_subsplatsTempIllumBuffer);
	//create illumination buffer
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(hquint32), m_totalSubsplats, &m_subsplatsIllumBuffer);
	
#ifdef DEBUG_ILLUM_BUFFER
	//for debugging
	m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R8G8B8A8_UNORM, WINDOW_SIZE, WINDOW_SIZE, true, &m_subsplatIllumDbgTexture);
#endif

	//create interpolated illumination buffer
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(hquint32), totalSizeExceptLastLevel, &m_subsplatsInterpolatedBuffer);
	
	//create final interpolated texture
	m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R8G8B8A8_UNORM, WINDOW_SIZE, WINDOW_SIZE, false, &m_subsplatsFinalInterpolatedTexture);
	//register final interpolated illumination texture to be used by effect manager
	HQEngineApp::GetInstance()->GetResourceManager()->AddTextureResource("final_interpolated_illum_img", m_subsplatsFinalInterpolatedTexture);

	//create buffer containing subsplats for all refinement steps
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(Subsplat), totalSizeExceptLastLevel, &m_subsplatsRefineStepsBuffer);

	//coarsest subplats 
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

		//write coarsest subsplats to the beginning of buffer
		m_subsplatsRefineStepsBuffer->Update(0, sizeof(Subsplat) * coarsestSize * coarsestSize, subsplats);

		delete[] subsplats;
	}

	/*--------buffers containing initial counts-------------*/
	//create buffer containing subsplats' counts.
	//the first element is final subsplats list count, the subsequent ones are counts for each refinement step
	m_pRDevice->GetShaderManager()->CreateBufferUAV(initialSubsplatsCounts, sizeof(hquint32),
		sizeof(initialSubsplatsCounts) / sizeof(hquint32),
		&m_initialSubsplatsCountsBuffer);
	//create buffer containing initial dispatch arguments for indirect illumination and refinement steps
	m_pRDevice->GetShaderManager()->CreateComputeIndirectArgs(initialDispatchArgs,
		sizeof(initialDispatchArgs) / sizeof(DispatchComputeArgs),
		&m_initialDispatchArgsBuffer);

	/*----buffer containing counts at runtime------------*/
	//create buffer containing subsplats' counts.
	//the first element is final subsplats list count, the subsequent ones are counts for each refinement step
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(hquint32), 
		sizeof(initialSubsplatsCounts) / sizeof(hquint32), 
		&m_subsplatsCountBuffer);

	//create buffer containing dispatch arguments for indirect illumination and refinement steps
	m_pRDevice->GetShaderManager()->CreateComputeIndirectArgs(NULL,  
		sizeof(initialDispatchArgs) / sizeof(DispatchComputeArgs),
		&m_dispatchArgsBuffer);

	/*----------------------------------*/
	//create buffer containing final subsplats list
	m_pRDevice->GetShaderManager()->CreateBufferUAV(NULL, sizeof(Subsplat), WINDOW_SIZE * WINDOW_SIZE, &m_finalSubsplatsBuffer);

}


void App::Upsample()
{
	//thread group size
	const hquint32 threadGroupDim = 16;
	//mipmap size
	hquint32 width = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1), height = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);


	//read from illumination buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(0, m_subsplatsIllumBuffer);

	//output to interpolated buffer and final texture
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(5, m_subsplatsInterpolatedBuffer);
	m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(6, m_subsplatsFinalInterpolatedTexture, 0, true);


	/*-----------first step: generate first interpolated level----------------*/
	hqint32 level = 0;
	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("upsample_cs_1st_step")->GetShader();

	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

	m_uniformLevelInfoBuffer->Update(&level);

	m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

	m_pRDevice->BufferUAVBarrier();

	/*-------------upsample the rest, starting from second lowest level----------------*/
	level = 1;
	width <<= 1;
	height <<= 1;

	shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("upsample_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
	
	while (level < NUM_RESOLUTIONS){
		m_uniformLevelInfoBuffer->Update(&level);


		m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

		m_pRDevice->BufferUAVBarrier();

		width <<= 1;
		height <<= 1;
		level++;
	}

	m_pRDevice->TextureUAVBarrier();
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

void App::KeyPressed(HQKeyCodeType keyCode, bool repeat)
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
	case HQKeyCode::F:
		m_dynamicLight = !m_dynamicLight;
		break;
	case HQKeyCode::G:
	{
		m_giOn = !m_giOn;
		if (!m_giOn)
		{
			//clear indirect illumination texture
			hquint32 *blackColor = new hquint32[WINDOW_SIZE * WINDOW_SIZE];
			memset(blackColor, 0, WINDOW_SIZE * WINDOW_SIZE * sizeof(hquint32));

			m_subsplatsFinalInterpolatedTexture->SetLevelContent(0, blackColor);

			delete[] blackColor;
		}
	}
		break;
	case HQKeyCode::TAB:
		m_diffuseScene = !m_diffuseScene;
		break;
	case HQKeyCode::NUM1:
		m_subsplatsRefineThreshold[0] += 0.01f;
		m_uniformRefineThresholdBuffer->Update(m_subsplatsRefineThreshold);
		break;
	case HQKeyCode::NUM2:
		m_subsplatsRefineThreshold[0] -= 0.01f;
		if (m_subsplatsRefineThreshold[0] <= 0)
			m_subsplatsRefineThreshold[0] = 0;
		m_uniformRefineThresholdBuffer->Update(m_subsplatsRefineThreshold);
		break;

	case HQKeyCode::NUM3:
		m_subsplatsRefineThreshold[1] += 0.1f;
		m_uniformRefineThresholdBuffer->Update(m_subsplatsRefineThreshold);
		break;
	case HQKeyCode::NUM4:
		m_subsplatsRefineThreshold[1] -= 0.1f;
		if (m_subsplatsRefineThreshold[1] <= 0)
			m_subsplatsRefineThreshold[1] = 0;
		m_uniformRefineThresholdBuffer->Update(m_subsplatsRefineThreshold);
		break;

	case HQKeyCode::NUM5:
		m_subsplatsRefineThreshold[2] += 0.1f;
		m_uniformRefineThresholdBuffer->Update(m_subsplatsRefineThreshold);
		break;
	case HQKeyCode::NUM6:
		m_subsplatsRefineThreshold[2] -= 0.1f;
		if (m_subsplatsRefineThreshold[2] <= 0)
			m_subsplatsRefineThreshold[2] = 0;
		m_uniformRefineThresholdBuffer->Update(m_subsplatsRefineThreshold);
		break;
#if defined DEBUG_ILLUM_BUFFER
	case HQKeyCode::C://capture illumination buffer content
	{
		hquint32 coarsestSize = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);
		hquint32 level_size = 0;
		hquint32 numSubsplats = 0;
		hquint32 *buffer = new hquint32[m_totalSubsplats];
		hquint32* level_data;

		//save illumination buffer's content to files
		m_subsplatsIllumBuffer->CopyContent(buffer);

		level_data = buffer;
		for (hquint32 i = 0; i < NUM_RESOLUTIONS; ++i)
		{
			std::stringstream fileName; fileName << ("dbg_illumination_buffer");
			fileName << i << ".bmp";

			level_size = coarsestSize << i;
			numSubsplats = level_size * level_size;

			lodepng::encode(fileName.str(), (const unsigned char*)level_data, level_size, level_size);

			level_data += numSubsplats;
		}

		//save interpolated buffer's content to files
		m_subsplatsInterpolatedBuffer->CopyContent(buffer);

		level_data = buffer;
		for (hquint32 i = 0; i < NUM_RESOLUTIONS - 1; ++i)
		{
			std::stringstream fileName; fileName <<  ("dbg_interpolated_buffer");
			fileName << i << ".bmp";

			level_size = coarsestSize << i;
			numSubsplats = level_size * level_size;

			lodepng::encode(fileName.str(), (const unsigned char*)level_data, level_size, level_size);

			level_data += numSubsplats;
		}

		//final interpolated texture
		m_subsplatsFinalInterpolatedTexture->CopyFirstLevelContent(buffer);
		level_data = buffer;
		{
			std::stringstream fileName; fileName << ("dbg_interpolated_buffer");
			fileName << NUM_RESOLUTIONS - 1 << ".bmp";

			level_size = WINDOW_SIZE;
			numSubsplats = level_size * level_size;

			lodepng::encode(fileName.str(), (const unsigned char*)level_data, level_size, level_size);
		}

		//clean up
		delete[] buffer;
	}
		break;
#endif//#if defined DEBUG_ILLUM_BUFFER
	}
}