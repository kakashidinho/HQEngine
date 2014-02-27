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
RenderLoop::RenderLoop(){
	m_pRDevice = HQEngineApp::GetInstance()->GetRenderDevice();


	//create scene container
	m_scene = new HQSceneNode("scene_root");

	//create model
	m_model = new HQMeshNode(
		"cornell_box", 
		"../Data/cornell_box.hqmesh",
		m_pRDevice,
		HQ_NOT_USE_VSHADER,
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
			HQ_RA_D3D
		);

	//create light object
	m_light = new SpotLight(
			HQColorRGBA(1, 1, 1, 1),
			3.43f, 5.478f , -2.27f,
			0, -1, 0,
			HQToRadian(90),
			2.0f,
			1000.f,
			HQ_RA_D3D);

	//add all to containers
	m_scene->AddChild(m_model);
	m_scene->AddChild(m_camera);

	//init resources for depth pass
	DepthPassInit();

	//init resources and states for low res pass
	LowresPassInit();

	//int final pass
	FinalPassInit();

	//decode noise map from RGBA image to float texture
	DecodeNoiseMap();

	//init render device
	m_pRDevice->SetClearColorf(1, 1, 1, 1);
	m_pRDevice->GetStateManager()->SetFaceCulling(HQ_CULL_CCW);
	HQDepthStencilStateDesc depthEnableState(HQ_DEPTH_FULL);
	hquint32 depthEnableStateID = 0;
	m_pRDevice->GetStateManager()->CreateDepthStencilState(depthEnableState, &depthEnableStateID);
	m_pRDevice->GetStateManager()->ActiveDepthStencilState(depthEnableStateID);
	//m_pRDevice->GetStateManager()->SetFillMode(HQ_FILL_WIREFRAME);

	//point sampling state
	HQSamplerStateDesc stDesc1(HQ_FM_MIN_MAG_MIP_POINT, HQ_TAM_CLAMP, HQ_TAM_CLAMP);
	m_pRDevice->GetStateManager()->CreateSamplerState(stDesc1, &point_sstate);

	//black border color sampling state
	HQSamplerStateDesc stDesc2(HQ_FM_MIN_MAG_MIP_LINEAR, HQ_TAM_BORDER, HQ_TAM_BORDER, 1, HQColorRGBA(0.f, 0.f, 0.f, 1.f));
	m_pRDevice->GetStateManager()->CreateSamplerState(stDesc2, &border_sstate);

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
	hquint32 encoded_noise_map, width, height;//encoded noise map
	hquint32 vid, pid, program;//vertex, pixel shader, program
	hquint32 vBuffer, vInputLayout;//vertex buffer and input layout
	hquint32 pointSamplerState;//point sampling state

	//create decoding shader program
	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_VERTEX_SHADER,
		"../Data/noise_decoding.cg",
		NULL,
		false,
		"VS",
		&vid);

	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_PIXEL_SHADER,
		"../Data/noise_decoding.cg",
		NULL,
		false,
		"PS",
		&pid);

	m_pRDevice->GetShaderManager()->CreateProgram(
		vid, pid, HQ_NULL_GSHADER, NULL, 
		&program);

	//load texture containing encoded random numbers in RGBA format
	m_pRDevice->GetTextureManager()->AddTexture(
		"../Data/random_factors_20x20.tga",
		1.0,
		NULL,
		0,
		false,
		HQ_TEXTURE_2D,
		&encoded_noise_map);
	m_pRDevice->GetTextureManager()->GetTexture2DSize(encoded_noise_map, width, height);

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
	m_pRDevice->GetVertexStreamManager()->CreateVertexInputLayout(vAttrDescs, 2, vid, &vInputLayout);

	//create render target to hold the decoded random numbers 
	hquint32 decodeRT;
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		width, height,
		false,
		HQ_RTFMT_RGBA_FLOAT64,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&decodeRT,
		&m_noise_map);

	//point sampling state
	HQSamplerStateDesc stDesc1(HQ_FM_MIN_MAG_MIP_POINT, HQ_TAM_CLAMP, HQ_TAM_CLAMP);
	m_pRDevice->GetStateManager()->CreateSamplerState(stDesc1, &pointSamplerState);

	//now decode the random numbers by render them to texture. result = (R=decoded number 1, G=sin(2*pi*decoded number 2), B=cos(2*pi*decoded number 2), A = decoded number 1 ^ 2)
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTarget(
		decodeRT, 
		HQ_NULL_ID);

	m_pRDevice->GetShaderManager()->ActiveProgram(program);

	m_pRDevice->GetVertexStreamManager()->SetVertexBuffer(vBuffer, 0, 6 * sizeof(float));
	m_pRDevice->GetVertexStreamManager()->SetVertexInputLayout(vInputLayout);

	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER| 0, pointSamplerState);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER| 0, encoded_noise_map);

	m_pRDevice->SetPrimitiveMode(HQ_PRI_TRIANGLE_STRIP);
	//draw full screen quad
	HQViewPort vp = {0, 0, width, height};
	m_pRDevice->SetViewPort(vp);
	m_pRDevice->BeginRender(HQ_TRUE, HQ_FALSE, HQ_FALSE, HQ_TRUE);
	m_pRDevice->DrawPrimitive(2, 0);
	m_pRDevice->EndRender();
	m_pRDevice->DisplayBackBuffer();//for debugging

	//clean up
	m_pRDevice->GetVertexStreamManager()->RemoveVertexBuffer(vBuffer);
	m_pRDevice->GetTextureManager()->RemoveTexture(encoded_noise_map);
	m_pRDevice->GetShaderManager()->DestroyProgram(program);
	m_pRDevice->GetShaderManager()->DestroyShader(vid);
	m_pRDevice->GetShaderManager()->DestroyShader(pid);

	//switch to default render target
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTarget(
		HQ_NULL_ID,
		HQ_NULL_ID);

	//switch back to default shader program
	m_pRDevice->GetShaderManager()->ActiveProgram(HQ_NOT_USE_SHADER);

	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER| 0, 0);
}

//rendering loop
void RenderLoop::Render(HQTime dt){

	//update scene
	m_scene->SetUniformScale(0.01f);
	m_scene->Update(dt);
	
	//depth pass rendering
	DepthPassRender(dt);

	//low resolution rendering pass
	LowresPassRender(dt);

	//final rendering pass
	FinalPassRender(dt);
}