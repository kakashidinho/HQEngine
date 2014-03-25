/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "RenderLoop.h"

/*--------SpotLight----------*/
SpotLight::SpotLight(
					 const HQColor& diffuse,
					hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,
					hqfloat32 dirX, hqfloat32 dirY, hqfloat32 dirZ,
					hqfloat32 _angle,
					hqfloat32 _falloff,
					hqfloat32 _maxRange,
					HQRenderAPI renderApi
					 )
:	BaseLight(diffuse, posX, posY, posZ),
	_direction(HQVector4::New(dirX, dirY, dirZ, 0.0f)),
	angle(_angle), falloff(_falloff), maxRange(_maxRange)
{
	HQFloat4 upVec;//up vector for light camera
	if (posX == 0.f && posY == 0.0f)
		upVec.Set(0.0f, -posZ, posY);
	else
		upVec.Set(posY, -posX, 0.0f);

	_lightCam = new HQBasePerspectiveCamera(
			posX, posY, posZ,
			upVec.x, upVec.y, upVec.z,
			dirX, dirY, dirZ,
			angle,
			1.0f,
			1.0f,
			maxRange,
			renderApi
		);
}

SpotLight::~SpotLight()
{
	delete _direction;
	delete _lightCam;
}

/*----------RenderLoop----------*/
//constructor
RenderLoop::RenderLoop(const char* _renderAPI)
{
	strcpy(this->m_renderAPI_name, _renderAPI);
	char apiResNamXML[256];
	if (strcmp(m_renderAPI_name, "GL") == 0)
	{
		this->m_renderAPI_type = HQ_RA_OGL;
		strcpy(apiResNamXML, "rsm_resourcesGL.script");
	}
	else if (strcmp(m_renderAPI_name, "D3D11") == 0)
	{
		this->m_renderAPI_type = HQ_RA_D3D;
		strcpy(apiResNamXML, "rsm_resourcesD3D11.script");
	}
	else
	{
		this->m_renderAPI_type = HQ_RA_D3D;
		strcpy(apiResNamXML, "rsm_resourcesD3D9.script");
	}


	//init resources
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile(apiResNamXML);
	HQEngineApp::GetInstance()->GetResourceManager()->AddResourcesFromFile("rsm_resourcesCommon.script");
	HQEngineApp::GetInstance()->GetEffectManager()->AddEffectsFromFile("rsm_effects.script");

	//retrieve main effect
	rsm_effect = HQEngineApp::GetInstance()->GetEffectManager()->GetEffect("rsm");


	m_pRDevice = HQEngineApp::GetInstance()->GetRenderDevice();


	//create scene container
	m_scene = new HQSceneNode("scene_root");

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
	m_light = new SpotLight(
			HQColorRGBA(1, 1, 1, 1),
			3.43f, 5.478f , -2.27f,
			0, -1, 0,
			HQToRadian(90),
			2.0f,
			1000.f,
			this->m_renderAPI_type);

	//add all to containers
	m_scene->AddChild(m_model);
	m_scene->AddChild(m_camera);

	//decode noise map from RGBA image to float texture
	DecodeNoiseMap();

	//init render device
	m_pRDevice->SetClearColorf(1, 1, 1, 1);
	//m_pRDevice->GetStateManager()->SetFillMode(HQ_FILL_WIREFRAME);

	/*---------create and bind uniform buffers--*/
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(Transform), NULL, true, &this->m_uniformTransformBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(Material), NULL, true, &this->m_uniformMaterialBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(LightProperties), NULL, true, &this->m_uniformLightProtBuffer);
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(sizeof(LightView), NULL, true, &this->m_uniformLightViewBuffer);

	if (m_renderAPI_type == HQ_RA_OGL)
	{
		m_pRDevice->GetShaderManager()->SetUniformBuffer(0, m_uniformTransformBuffer);
		m_pRDevice->GetShaderManager()->SetUniformBuffer(1, m_uniformLightViewBuffer);
		m_pRDevice->GetShaderManager()->SetUniformBuffer(2, m_uniformMaterialBuffer);
		m_pRDevice->GetShaderManager()->SetUniformBuffer(3, m_uniformLightProtBuffer);
	}
	else
	{
		m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 0, m_uniformTransformBuffer);
		m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 1, m_uniformLightViewBuffer);
		m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER | 2, m_uniformMaterialBuffer);
		m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_PIXEL_SHADER | 3, m_uniformLightProtBuffer);
	}
}

//destructor
RenderLoop::~RenderLoop()
{
	delete m_model;
	delete m_camera;
	delete m_light;

	delete m_scene;
}


void RenderLoop::DecodeNoiseMap()//decode noise map from RGBA image to float texture
{
	hquint32 width, height;//encoded noise map's size
	//encoded noise map
	HQEngineTextureResource *encoded_noise_map = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("random_factors_img");
	hquint32 vBuffer, vInputLayout;//vertex buffer and input layout

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
	HQEngineShaderResource* vshader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("noise_decoding_vs");
	HQEngineApp::GetInstance()->GetEffectManager()->CreateVertexInputLayout(vAttrDescs, 2, vshader, &vInputLayout);
	
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
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(HQ_NULL_ID);
	m_pRDevice->DisplayBackBuffer();
}

//rendering loop
void RenderLoop::Render(HQTime dt){

	//update scene
	m_scene->SetUniformScale(0.01f);
	m_scene->Update(dt);

	//send scene data to shader
	Transform * transform;
	LightView * lightView;
	LightProperties * lightProt;
	
	//send transform data
	m_pRDevice->GetShaderManager()->MapUniformBuffer(m_uniformTransformBuffer, (void**)&transform);
	transform->worldMat = m_model->GetWorldTransform();
	transform->viewMat = m_camera->GetViewMatrix();
	transform->projMat = m_camera->GetProjectionMatrix();
	m_pRDevice->GetShaderManager()->UnmapUniformBuffer(m_uniformTransformBuffer);

	//send light camera's matrices data
	m_pRDevice->GetShaderManager()->MapUniformBuffer(m_uniformLightViewBuffer, (void**)&lightView);
	lightView->lightViewMat = m_light->lightCam().GetViewMatrix();
	lightView->lightProjMat = m_light->lightCam().GetProjectionMatrix();
	m_pRDevice->GetShaderManager()->UnmapUniformBuffer(m_uniformLightViewBuffer);

	//send light properties data
	m_pRDevice->GetShaderManager()->MapUniformBuffer(m_uniformLightProtBuffer, (void**)&lightProt);
	memcpy(lightProt->lightPosition, &m_light->position(), sizeof(HQVector4));
	memcpy(lightProt->lightDirection, &m_light->direction(), sizeof(HQVector4));
	memcpy(lightProt->lightDiffuse, &m_light->diffuseColor, sizeof(HQVector4));
	lightProt->lightFalloff_lightPCosHalfAngle[0] = m_light->falloff;
	lightProt->lightFalloff_lightPCosHalfAngle[1] = pow(cosf(m_light->angle * 0.5f), m_light->falloff);
	m_pRDevice->GetShaderManager()->UnmapUniformBuffer(m_uniformLightProtBuffer);
	
	//depth pass rendering
	DepthPassRender(dt);

	//low resolution rendering pass
	LowresPassRender(dt);

	//final rendering pass
	FinalPassRender(dt);
}