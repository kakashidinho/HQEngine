#include "Game.h"

//#define GLES
#if !defined HQ_ANDROID_PLATFORM && !defined HQ_IPHONE_PLATFORM
#define GLES_EMULATION
#endif

#define HQ_CUSTOM_MALLOC
#include "../HQEngineCustomHeap.h"

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	define TEXTURE_BUFFER 0

#	if TEXTURE_BUFFER
#		define TEXTURE_BUFFER_DEF_STR "1"
#	else
#		define TEXTURE_BUFFER_DEF_STR "0"
#	endif
#else
#	define TEXTURE_BUFFER_DEF_STR "0"
#endif

#define OGL_RENDERER  1
#define D3D9_RENDERER 0
#define D3D10_RENDERER 2
#define D3D11_RENDERER 3

struct BUFFER
{
	HQBaseMatrix3x4 rotation;
	HQBaseMatrix4 viewProj;
};

struct BUFFER2
{
	HQBaseMatrix3x4 rotation;
	HQBaseMatrix4 viewProj;
	HQBaseMatrix3x4 bones[36];
};

struct VertexPos
{
	hq_float32 x,y,z;
	hq_uint32 color;
};

struct VertexUV
{
	hq_float32 u , v;
};

/*-------data-----------*/

VertexPos vertices[] = {
	-0.5f, -0.5f , 0 , 0,
	-0.5f , 0.5f , 0.0f ,0,
	0.5 , -0.5f , 0.0f ,0,
	0.5 , 0.5f , 0.0f ,0
};
const VertexUV verticeUVs[] = {
	 0.1f, 0.1f,
	 0.1f , 0.9f,
	 0.9f , 0.1f,
	 1.1f , 1.1f
};

const hq_float32 psizes[] = { 5.0f , 10.0f , 15.0f , 20.0f};
const hq_float32 psizes2[] = { 20.0f , 15.0f , 10.0f , 5.0f};
hq_ushort16 indices[] = {
	0 , 1 , 2 ,
	1 , 2 , 3
};

HQVertexAttribDescArray<4> desc;

HQVertexAttribDescArray<3> desc2;

#ifdef LINUX
#error need implement
#else
Game::Game() 
: m_offsetX(0), m_offsetY(0)
#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
 , app_exit( false ), app_pause(false)
#endif

{
#endif

	int *pt=  HQ_NEW int;
	HQ_DELETE pt;

	HQEngineApp::CreateInstance(true); 

	TRACE("here %s %d", __FILE__, __LINE__);
	

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	FILE *apiSetting = fopen("../test/setting/API.txt" , "r");
	char str[20];
	fscanf(apiSetting , "%s" , str);
	if (!strcmp(str , "GL"))
#elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	char str[] = "D3D11";
#else
	char str[] = "GL";
#endif

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		API = D3D11_RENDERER;
#else
		API = OGL_RENDERER;
#ifdef WIN32
	else if (!strcmp(str , "D3D9"))
		API = D3D9_RENDERER;
	else if (!strcmp(str , "D3D11"))
		API = D3D11_RENDERER;

	fclose(apiSetting);
#elif defined __APPLE__
	/*
	NSString * frameworkPath = [[NSBundle mainBundle] privateFrameworksPath];
	int len = [frameworkPath lengthOfBytesUsingEncoding:NSASCIIStringEncoding] + 1 ;
	char *cDir = HQ_NEW char[len];
	[frameworkPath getCString:cDir maxLength:len
					 encoding:NSASCIIStringEncoding];
	chdir(cDir);
	delete[] cDir;
	 */
#endif
#endif//#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#ifdef __APPLE__
	/*resource directory*/
	NSString *resourcePath = [[NSBundle mainBundle] resourcePath];

	int len = [resourcePath lengthOfBytesUsingEncoding:NSASCIIStringEncoding] + 1 ;
	char * cDir = HQ_NEW char[len + strlen("/test")];
	[resourcePath getCString:cDir maxLength:len
					encoding:NSASCIIStringEncoding];

	strcat(cDir, "/test");

	chdir(cDir);

	delete[] cDir;

#endif

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	HQEngineApp::SetCurrentDir("Assets/test");
#endif


#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	logFile = HQCreateDebugLogStream();
#elif defined WIN32
	logFile = HQCreateFileLogStream("../test/log/log.txt");
#elif defined ANDROID
	logFile = HQCreateLogCatStream();
#else
	logFile = HQCreateConsoleLogStream();
#endif

	TRACE("here %s %d", __FILE__, __LINE__);
	
#ifndef DISABLE_AUDIO	
	//audio
	this->audio = HQCreateAudioDevice(HQ_DEFAULT_SPEED_OF_SOUND , logFile, true);
	this->audio->SetListenerVolume(1.0f);
	hquint32 audiobufferID;
#define audioFile "../test/audio/battletoads-double-dragons-2.ogg"
	//this->audio->CreateAudioBufferFromFile(audioFile, &audiobufferID);

	//this->audio->CreateStreamAudioBufferFromFile(audioFile, 5, 65536,  &audiobufferID);
	this->audio->CreateStreamAudioBufferFromFile(audioFile, &audiobufferID);
	
	TRACE("here %s %d", __FILE__, __LINE__);

	HQAudioSourceInfo info = {HQ_AUDIO_PCM, 16, 1};
	//HQAudioSourceInfo info = {HQ_AUDIO_PCM, 16, 2};
	this->audio->CreateSource(info, audiobufferID, &music);

	TRACE("here %s %d", __FILE__, __LINE__);
#endif//#ifndef DISABLE_AUDIO
	//this->audio->GetSourceController(music)->SetVolume(0.5f);

	this->audio->GetSourceController(music)->Enable3DPositioning(HQ_FALSE);

	//application
	HQEngineApp::WindowInitParams params = HQEngineApp::WindowInitParams::Default();
#if defined _DEBUG || defined DEBUG
	params.flushDebugLog = true;
#else
	params.flushDebugLog = false;
#endif
	params.logStream = logFile;
	params.platformSpecific = NULL;
	params.rendererAdditionalSetting = NULL;
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	params.rendererSettingFileDir = "../../Setting.txt";
#else
	params.rendererSettingFileDir = "../test/setting/Setting.txt";
#endif
	params.rendererType = str;
	params.windowTitle = "test";
#ifdef ANDROID
	HQWIPPlatformSpecificType additionalSettings;
	additionalSettings.openGL_ApiLevel = 2;
	params.platformSpecific = &additionalSettings;
#elif defined IOS
	HQWIPPlatformSpecificType additionalSettings;
	additionalSettings.landscapeMode = false;
	params.platformSpecific = &additionalSettings;
#endif
	
	HQEngineApp::GetInstance()->InitWindow(&params);

	TRACE("here %s %d", __FILE__, __LINE__);

	pDevice = HQEngineApp::GetInstance()->GetRenderDevice();	
	pDevice->EndRender();

#if !(defined GLES || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM))
	HQResolution* resolutionList;
	hquint32 numRes;
	pDevice->GetAllDisplayResolution(NULL , numRes);
	resolutionList = HQ_NEW HQResolution[numRes];
	pDevice->GetAllDisplayResolution(resolutionList , numRes);

	for(hquint32 i = 0 ; i < numRes ; ++i)
	{
	    printf("%u x %u\n" , resolutionList[i].width , resolutionList[i].height);
	}
	delete[] resolutionList;
#endif
	/*--------viewproj matrix-----------*/
	HQA16ByteMatrix4Ptr view , proj;
	HQMatrix4rLookAtLH(HQA16ByteVector4Ptr(0,0,-2) , HQA16ByteVector4Ptr(0,0,0) , HQA16ByteVector4Ptr(0,1,0) , view);
	HQMatrix4rPerspectiveProjLH(HQPiFamily::_PIOVER4 , (hq_float32)pDevice->GetWidth() / pDevice->GetHeight() , 1.0f , 1000.0f , proj , (API == OGL_RENDERER)? HQ_RA_OGL : HQ_RA_D3D);

	HQMatrix4Multiply(view, proj, this->viewProj);
	/*-------------states---------------*/
	HQDepthStencilStateDesc dsDesc;
	dsDesc.depthMode = HQ_DEPTH_FULL;
	dsDesc.stencilEnable = true;
	dsDesc.refVal = 0x1;
	dsDesc.stencilMode.compareFunc = HQ_SF_LESS;
	pDevice->GetStateManager()->CreateDepthStencilState(dsDesc , &dsState);
	/*-------------texture--------------*/
	HQColor colorKey = HQColorRGBA(1, 1, 1, 0);

	pDevice->GetTextureManager()->AddTexture("../test/image/Marine.dds" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->texture);
	pDevice->GetTextureManager()->AddSingleColorTexture(HQColoruiRGBA(255 ,0,0,255 , pDevice->GetColoruiLayout()) ,
		&colorTexture);
	pDevice->GetTextureManager()->RemoveTexture(this->texture);
	pDevice->GetTextureManager()->RemoveTexture(this->colorTexture);
	pDevice->GetTextureManager()->AddTexture("../test/image/Marine.dds" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->texture);
	pDevice->GetTextureManager()->RemoveTexture(this->texture);
	pDevice->GetTextureManager()->AddTexture("../test/image/Marine.dds" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->texture);
	pDevice->GetTextureManager()->RemoveTexture(this->texture);
	pDevice->GetTextureManager()->AddSingleColorTexture(HQColoruiRGBA(255 ,0,0,255 , pDevice->GetColoruiLayout()) ,
 		&colorTexture);
	pDevice->GetTextureManager()->AddTexture("../test/image/Marine.dds" , 1.0f , NULL , 1 , true , HQ_TEXTURE_2D , &this->texture);
	//pDevice->GetTextureManager()->AddTexture("../test/image/MarineFlip.pvr" , 1.0f , NULL , 1 , true , HQ_TEXTURE_2D , &this->texture);
	pDevice->GetTextureManager()->AddTexture("../test/image/pen2.png" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->temp[0]);
	pDevice->GetTextureManager()->AddTexture("../test/image/pen2.jpg" , 1.0f , &colorKey , 1 , true , HQ_TEXTURE_2D , &this->temp[1]);
	pDevice->GetTextureManager()->AddTexture("../test/image/metall16bit.bmp" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->temp[2]);

	//pDevice->GetTextureManager()->AddTexture("../test/image/skyboxPVRTC2.pvr" , 1.0f , &colorKey , 1 , true , HQ_TEXTURE_CUBE , &this->texCube);

	this->curTexture = this->texture;

	HQSamplerStateDesc samplerDesc(HQ_FM_MIN_MAG_MIP_POINT, HQ_TAM_BORDER, HQ_TAM_BORDER, 1, HQColorRGBA(1 , 0.5 , 0.2 ,1)) ;

	pDevice->GetStateManager()->CreateSamplerState(samplerDesc , &samplerState[0]);

	samplerDesc.filterMode = HQ_FM_MIN_MAG_LINEAR_MIP_POINT;
	samplerDesc.maxAnisotropy = 16;
	pDevice->GetStateManager()->CreateSamplerState(samplerDesc , &samplerState[1]);

	BUFFER buffer;
	buffer.rotation = this->rotation[0] ;
	buffer.viewProj = *this->viewProj;
#ifndef GLES
	//pDevice->GetTextureManager()->AddTextureBuffer(HQ_TBFMT_R32G32B32A32_FLOAT , sizeof(HQMatrix3x4) + sizeof(HQMatrix4) , &buffer , true, &textureBuffer);
	pDevice->GetTextureManager()->AddTextureBuffer(HQ_TBFMT_R32G32B32A32_FLOAT , sizeof(BUFFER2) , NULL , true, &textureBuffer);
#endif
	if( API == OGL_RENDERER)
	{
		pDevice->GetStateManager()->SetSamplerState(colorTexture , samplerState[0]);
		pDevice->GetStateManager()->SetSamplerState(texture , samplerState[1]);
		pDevice->GetStateManager()->SetSamplerState(temp[0] , samplerState[1]);
		pDevice->GetStateManager()->SetSamplerState(temp[1] , samplerState[1]);
		pDevice->GetStateManager()->SetSamplerState(temp[2] , samplerState[1]);
	}
	else
	{
		pDevice->GetStateManager()->SetSamplerState(HQ_VERTEX_SHADER | 3 , samplerState[0]);
		pDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 15 , samplerState[1]);
		pDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 0 , samplerState[1]);
	}


	/*-------------shader---------------*/
	hq_uint32 vid , pid;
	hquint32 vid2, pid2;


	if (API == OGL_RENDERER)
	{
#ifdef GLES
		HQShaderMacro macro[] = {
			"version" , 
#	ifdef GLES_EMULATION
			"110",
			"highp","", "lowp", "", "mediump", "",
#	else
			"100" ,
#	endif
			"TEXTURE_COLOR" , "0" , "NOT_ORIGIN" , "" , "HQEXT_GLSL_ES", "", NULL , NULL};
#else
		HQShaderMacro macro[] = {"version" , "120" ,"TEXTURE_COLOR" , "0" , "NOT_ORIGIN" , "" , "NUNIFORM_BUFFER" , "" , NULL , NULL};
#endif

			
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_GLSL_DEBUG , "../test/shader/vs.txt",macro,
			"main" , &vid);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_GLSL_DEBUG , "../test/shader/vs-mesh.txt",macro,
			"main" , &vid2);

		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_GLSL_DEBUG , "../test/shader/ps.txt", macro,
			"main" , &pid);

		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_GLSL_DEBUG , "../test/shader/ps-mesh.txt", macro,
			"main" , &pid2);
		/*	
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_CG_DEBUG , "../test/shader/cg2.txt",macro,
			"VS" , &vid);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_CG_DEBUG , "../test/shader/cg2.txt", macro,
			"PS" , &pid);
			*/
	}
	else if (API == D3D9_RENDERER)
	{
		HQShaderMacro macro[] = {"TEXTURE_COLOR" , "0" , NULL , NULL};
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_CG_DEBUG , "../test/shader/cg2.txt", macro,
			"VS" , &vid);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_CG_DEBUG, "../test/shader/cg2-mesh.txt", macro,
			 "VS" , &vid2);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_CG_DEBUG , "../test/shader/cg2.txt", NULL,
			"PS" , &pid);

		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_CG_DEBUG , "../test/shader/cg2-mesh.txt", NULL,
			"PS" , &pid2);
	}

	else if (API == D3D10_RENDERER || API == D3D11_RENDERER)
	{
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		HQShaderMacro macro[] = {"TEXTURE_COLOR" , "0" , "BLENDWEIGHT3", "1", "TEXTURE_BUFFER", TEXTURE_BUFFER_DEF_STR, NULL , NULL};
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_HLSL_10_DEBUG , "../test/shader/hlsl.txt", macro,
			"VS" , &vid);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
			HQ_SCM_HLSL_10_DEBUG , "../test/shader/hlsl-mesh.txt", macro,
			"VS" , &vid2);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_HLSL_10_DEBUG , "../test/shader/hlsl.txt", NULL,
			"PS" , &pid);
		pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
			HQ_SCM_HLSL_10_DEBUG , "../test/shader/hlsl-mesh.txt", NULL,
			"PS" , &pid2);
#else
		pDevice->GetShaderManager()->CreateShaderFromByteCodeFile(HQ_VERTEX_SHADER ,
			"../test/shader/hlsl_vs.cso", &vid);
		pDevice->GetShaderManager()->CreateShaderFromByteCodeFile(HQ_VERTEX_SHADER ,
			"../test/shader/hlsl-mesh_vs.cso", &vid2);
		pDevice->GetShaderManager()->CreateShaderFromByteCodeFile(HQ_PIXEL_SHADER ,
			"../test/shader/hlsl_ps.cso", &pid);
		pDevice->GetShaderManager()->CreateShaderFromByteCodeFile(HQ_PIXEL_SHADER ,
			"../test/shader/hlsl-mesh_ps.cso", &pid2);
#endif
	}

	pDevice->GetShaderManager()->CreateProgram(vid , pid , HQ_NOT_USE_GSHADER, NULL , &program);
	pDevice->GetShaderManager()->CreateProgram(vid2 , pid2 , HQ_NOT_USE_GSHADER, NULL , &programMesh);

#ifndef GLES

	pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(BUFFER) , &buffer , true ,  &this->uniformBuffer[0]);

	if (API == OGL_RENDERER)
	{
		pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(HQColor) , NULL , true , &this->uniformBuffer[1]);
		pDevice->GetShaderManager()->SetUniformBuffer(11 , this->uniformBuffer[0]);
		pDevice->GetShaderManager()->SetUniformBuffer(10 , this->uniformBuffer[1]);
	}
	else if (API == D3D10_RENDERER || API == D3D11_RENDERER)
	{
		pDevice->GetShaderManager()->CreateUniformBuffer(sizeof(BUFFER2) , NULL , true , &this->uniformBuffer[1]);
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 11 , this->uniformBuffer[0]);
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 10 , this->uniformBuffer[1]);
	}
#endif
	/*------------check capabilities----*/
	bool support = pDevice->IsBlendStateExSupported();
	support = pDevice->IsIndexDataTypeSupported(HQ_IDT_UINT);
	/*------------vertex stream---------*/
	
	desc.SetPosition(0, 0, 0,  HQ_VADT_FLOAT3);
	desc.SetColor(1, 0, 3 * sizeof(hqfloat32), HQ_VADT_UBYTE4N);
	desc.SetPointSize(2, 1, 0, HQ_VADT_FLOAT);
	desc.SetTexcoord(3, 2, 0, HQ_VADT_FLOAT2, 6);

	desc2.SetPosition(0, 0, 0,  HQ_VADT_FLOAT3);
	desc2.SetColor(1, 0, 3 * sizeof(hqfloat32), HQ_VADT_UBYTE4N);
	desc2.SetTexcoord(2, 2, 0, HQ_VADT_FLOAT2, 0);

	for (hq_uint32 i = 0 ; i < sizeof(vertices) / sizeof(VertexPos) ; ++i)
		vertices[i].color = HQColoruiRGBA(255 , 255 ,255 ,255 , pDevice->GetColoruiLayout());

	pDevice->GetVertexStreamManager()->CreateVertexBuffer(vertices , 4 * sizeof (VertexPos) , true , true , &vertexbuffer[0]);
	pDevice->GetVertexStreamManager()->CreateVertexBuffer(verticeUVs , 4 * sizeof (VertexUV) , true ,true , &vertexbuffer[1]);
	pDevice->GetVertexStreamManager()->CreateVertexBuffer(psizes , 4 * sizeof (hq_float32) , false ,true , &vertexbuffer[2]);
	pDevice->GetVertexStreamManager()->CreateVertexBuffer(psizes2 , 4 * sizeof (hq_float32) , false ,true , &vertexbuffer[3]);
	pDevice->GetVertexStreamManager()->CreateIndexBuffer(indices , 6 * sizeof (hq_ushort16) , true , HQ_IDT_USHORT , &indexbuffer);


	pDevice->GetVertexStreamManager()->CreateVertexInputLayout(desc , 4 , vid , &vertexLayout[0]);
	pDevice->GetVertexStreamManager()->CreateVertexInputLayout(desc , 4 , vid , &vertexLayout[1]);
	pDevice->GetVertexStreamManager()->CreateVertexInputLayout(desc2 , 3 , vid , &vertexLayout[2]);


	/*---------mesh------------*/
	TRACE("here %s %d", __FILE__, __LINE__);

	mesh = HQ_NEW HQMeshNode("tiny", "../test/meshes/bat.hqmesh", pDevice, vid2, logFile);

	TRACE("here %s %d", __FILE__, __LINE__);

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	meshX = HQ_NEW MeshX("../test/meshes/tiny.x");
#endif

	deviceLost = false;

#ifndef DISABLE_AUDIO
	//play music
	this->audio->GetSourceController(music)->Play(HQ_TRUE);
	//this->audio->GetSourceController(music)->Play(HQ_FALSE);
#endif
}

Game::~Game()
{
	delete mesh;
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	delete meshX;
#endif
#ifndef DISABLE_AUDIO
	this->audio->Release();
#endif
	HQEngineApp::Release();
#ifdef LINUX
#else
	/*
#elif defined IOS
	logFile->Close();
	FILE *clogFile = fopen("../test/log/log.txt" , "r");
	fseek(clogFile, 0, SEEK_END);
	uint size = ftell(clogFile);
	char *logString = HQ_NEW char[size + 1];
	logString[size] = '\0';
	rewind(clogFile);
	fread(logString, size, 1, clogFile);
	printf("%s" , logString);
	delete[] logString;
	fclose(clogFile);
#endif
#ifndef IOS
	 */
	if (logFile)
		logFile->Close();
	
#endif
}

void Game::OnLostDevice()
{
	deviceLost = true;
	OutputDebugStringA("OnLostDevice");
}

void Game::OnResetDevice()
{
	deviceLost = false;
	OutputDebugStringA("OnResetDevice");
	if (!pDevice->IsRunning() && API != D3D9_RENDERER)
		return;

	mesh->OnResetDevice();

	pDevice->GetVertexStreamManager()->UpdateVertexBuffer(vertexbuffer[0] ,0 , sizeof(vertices), vertices );
	pDevice->GetVertexStreamManager()->UpdateVertexBuffer(vertexbuffer[1] ,0 , sizeof(verticeUVs), verticeUVs );
	pDevice->GetVertexStreamManager()->UpdateVertexBuffer(vertexbuffer[2] ,0 , 0, psizes );
	pDevice->GetVertexStreamManager()->UpdateVertexBuffer(vertexbuffer[3] ,0 , 0, psizes2 );
	pDevice->GetVertexStreamManager()->UpdateIndexBuffer(indexbuffer, 0 , 0, indices );


#ifdef ANDROID
	/*-------------reload textures--------------*/
	HQColor colorKey = HQColorRGBA(1, 1, 1, 0);
	pDevice->GetTextureManager()->AddSingleColorTexture(HQColoruiRGBA(255 ,0,0,255 , pDevice->GetColoruiLayout()) ,
 		&colorTexture);
	pDevice->GetTextureManager()->AddTexture("../test/image/Marine.dds" , 1.0f , NULL , 1 , true , HQ_TEXTURE_2D , &this->texture);
	//pDevice->GetTextureManager()->AddTexture("../test/image/MarineFlip.pvr" , 1.0f , NULL , 1 , true , HQ_TEXTURE_2D , &this->texture);
	pDevice->GetTextureManager()->AddTexture("../test/image/pen2.png" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->temp[0]);
	pDevice->GetTextureManager()->AddTexture("../test/image/pen2.jpg" , 1.0f , &colorKey , 1 , true , HQ_TEXTURE_2D , &this->temp[1]);
	pDevice->GetTextureManager()->AddTexture("../test/image/metall16bit.bmp" , 1.0f , NULL , 0 , true , HQ_TEXTURE_2D , &this->temp[2]);

	//pDevice->GetTextureManager()->AddTexture("../test/image/skyboxPVRTC2.pvr" , 1.0f , &colorKey , 1 , true , HQ_TEXTURE_CUBE , &this->texCube);

	this->curTexture = this->texture;

	BUFFER buffer;
	buffer.rotation = this->rotation[0] ;
	buffer.viewProj = *this->viewProj;
#ifndef GLES
	pDevice->GetTextureManager()->AddTextureBuffer(HQ_TBFMT_R32G32B32A32_FLOAT , sizeof(HQMatrix3x4) + sizeof(HQMatrix4) , &buffer , true, &textureBuffer);
#endif
	pDevice->GetStateManager()->SetSamplerState(colorTexture , samplerState[0]);
	pDevice->GetStateManager()->SetSamplerState(texture , samplerState[1]);
	pDevice->GetStateManager()->SetSamplerState(temp[0] , samplerState[1]);
	pDevice->GetStateManager()->SetSamplerState(temp[1] , samplerState[1]);
	pDevice->GetStateManager()->SetSamplerState(temp[2] , samplerState[1]);

	/*-------------reload shaders---------------*/
	hq_uint32 vid , pid, vid2, pid2;

#ifdef GLES
	HQShaderMacro macro[] = {"version" , "100" ,"TEXTURE_COLOR" , "0" , "NOT_ORIGIN" , "" , NULL , NULL};
#else
	HQShaderMacro macro[] = {"version" , "120" ,"TEXTURE_COLOR" , "0" , "NOT_ORIGIN" , "" , "NUNIFORM_BUFFER" , "" , NULL , NULL};
#endif

		
	pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
		HQ_SCM_GLSL_DEBUG , "../test/shader/vs.txt",macro,
		"main" , &vid);
	pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
		HQ_SCM_GLSL_DEBUG , "../test/shader/ps.txt", macro,
		"main" , &pid);

	pDevice->GetShaderManager()->CreateShaderFromFile(HQ_VERTEX_SHADER ,
		HQ_SCM_GLSL_DEBUG , "../test/shader/vs-mesh.txt",macro,
		"main" , &vid2);
	pDevice->GetShaderManager()->CreateShaderFromFile(HQ_PIXEL_SHADER ,
		HQ_SCM_GLSL_DEBUG , "../test/shader/ps-mesh.txt", macro,
		"main" , &pid2);

	pDevice->GetShaderManager()->CreateProgram(vid , pid , HQ_NOT_USE_GSHADER, NULL , &program);
	pDevice->GetShaderManager()->CreateProgram(vid2 , pid2 , HQ_NOT_USE_GSHADER, NULL , &programMesh);
#endif
}

void Game::Render(HQTime dt)
{
#ifdef WIN32
	//Sleep(13);
#endif

#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	mutex.Lock();
	if (app_exit)
	{
		HQEngineApp::GetInstance()->Stop();	
	}
		
	mutex.Unlock();

	bool pause;
	mutex.Lock();
	pause = app_pause;
	mutex.Unlock();
#else
	bool pause = false;
#endif
	
	if (!pDevice->IsRunning() || pause)
		return;
	
	
	if (deviceLost)
		return;
		
	//TRACE("here %s %d", __FILE__, __LINE__);

	this->audio->GetSourceController(music)->UpdateStream();

	static hq_float32 angle = 0.0f;
	angle += HQPiFamily::PI / 180.0f;
	HQ_DECL_STACK_MATRIX3X4( scale) ;
	//HQMatrix3x4Scale(0.8f, 0.8f, 0.8f, &scale);
	HQMatrix3x4cRotateY(-angle , &rotation[0]);
	HQMatrix3x4cRotateX(-angle , &rotation[1]);
	HQMatrix3x4cRotateZ(-angle , &rotation[2]);

	//rotation[0] *= rotation[1];
	//rotation[0] *= rotation[2];

	HQMatrix3x4MultiMultiply(rotation, 2, rotation);
	rotation[0] *= scale;
	
	/*------sound position----------*/
	HQ_DECL_STACK_VECTOR4_CTOR_PARAMS(soundPosition, (0, 1, 1));
	HQVector4TransformCoord(&soundPosition, &rotation[0], &soundPosition);

#ifndef DISABLE_AUDIO
	audio->GetSourceController(music)->SetPosition(soundPosition);
#endif
	
	//TRACE("here %s %d", __FILE__, __LINE__);
	//update mesh
	mesh->SetUniformScale(0.3f);
	mesh->RotateY(HQPiFamily::PI / 180.f);
	mesh->AdvanceAnimationTime(dt);
	mesh->Update(dt);
	const HQMatrix3x4 * boneMatrices = mesh->GetBoneTransformMatrices();

	//TRACE("here %s %d", __FILE__, __LINE__);

#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	meshX->GetAnimationController()->AdvanceTime(dt, NULL);
	meshX->Update();
#endif

	//clear whole render target
	pDevice->SetClearStencilVal(0xff);
	pDevice->SetClearDepthVal(1.0f);
	pDevice->SetClearColorf(0.5f , 1.0f ,0.5f , 1.0f);
	pDevice->BeginRender(HQ_TRUE , HQ_TRUE , HQ_TRUE ,HQ_TRUE);

	hquint32 halfw = pDevice->GetWidth() / 2;
	hquint32 halfh = pDevice->GetHeight() / 2;

	//clear depth
	pDevice->SetClearDepthVal(0.5f);
	HQViewPort viewport2 = {0 + m_offsetX , halfh + m_offsetY , halfw / 2 , halfh / 2};
	pDevice->SetViewPort(viewport2);
	pDevice->Clear(HQ_FALSE, HQ_TRUE, HQ_FALSE, HQ_FALSE);

	//clear stencil
	pDevice->SetClearStencilVal(0x1);
	HQViewPort viewport3 = {halfw / 2  + m_offsetX, halfh * 3 / 2  + m_offsetY, halfw / 2, halfh / 2};
	pDevice->SetViewPort(viewport3);
	pDevice->Clear(HQ_FALSE, HQ_FALSE, HQ_TRUE, HQ_FALSE);

	//clear color
	HQViewPort viewport = {0  + m_offsetX, halfh  + m_offsetY , halfw , halfh};
	pDevice->SetViewPort(viewport);
	pDevice->SetClearColorf(1, 0, 0, 1);
	pDevice->Clear(HQ_TRUE, HQ_FALSE, HQ_FALSE, HQ_FALSE);
	
	//debug textures
	if (API != OGL_RENDERER)
	{
		pDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 2 , this->texCube);
		pDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 2 , this->temp[0]);
		pDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 2 , this->temp[1]);
		pDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 2 , this->temp[2]);
	}

	if( API == OGL_RENDERER)
	{
		pDevice->GetTextureManager()->SetTexture(3 , colorTexture);
		pDevice->GetTextureManager()->SetTexture(5 , curTexture);
#ifndef GLES
		pDevice->GetTextureManager()->SetTexture(11 , textureBuffer);
#endif
	}
	else if( API == D3D9_RENDERER)
	{
		pDevice->GetTextureManager()->SetTexture(HQ_VERTEX_SHADER | 3 , colorTexture);
		//pDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 15 , curTexture);
		pDevice->GetTextureManager()->SetTextureForPixelShader(0 , curTexture);
	}
	else if( API == D3D10_RENDERER || API == D3D11_RENDERER)
	{
		pDevice->GetTextureManager()->SetTextureForPixelShader(100 , curTexture);
		if (pDevice->GetMaxShaderStageTextures(HQ_VERTEX_SHADER) > 0)
			pDevice->GetTextureManager()->SetTexture((HQ_VERTEX_SHADER | 11) , textureBuffer);
	}
	pDevice->SetPrimitiveMode(HQ_PRI_TRIANGLES);
	//pDevice->GetStateManager()->SetFillMode(HQ_FILL_WIREFRAME);

	static int index = 1;
	index = 1 - index;
	pDevice->GetVertexStreamManager()->SetVertexInputLayout(vertexLayout[index]);
	pDevice->GetVertexStreamManager()->SetVertexBuffer(vertexbuffer[0], 0 , sizeof (VertexPos));
	pDevice->GetVertexStreamManager()->SetVertexBuffer(vertexbuffer[1], 2 , sizeof (VertexUV));
	pDevice->GetVertexStreamManager()->SetVertexBuffer(vertexbuffer[2 + index], 1 , sizeof (hq_float32));

	pDevice->GetVertexStreamManager()->SetIndexBuffer(indexbuffer );

	pDevice->GetShaderManager()->ActiveProgram(programMesh);

	if (API == OGL_RENDERER)
	{
		/*
		BUFFER * pTBuffer0 = NULL;

		pDevice->GetTextureManager()->MapTextureBuffer(textureBuffer , (void**)&pTBuffer0);
		if(pTBuffer0)
		{
			memcpy(&pTBuffer0->rotation , &rotation[0] , sizeof(HQMatrix3x4));
			memcpy(&pTBuffer0->viewProj, viewProj , sizeof(HQMatrix4));
		}
		pDevice->GetTextureManager()->UnmapTextureBuffer(textureBuffer );
        */
#ifndef GLES		
		BUFFER * pCBuffer0 = NULL;

		pDevice->GetShaderManager()->MapUniformBuffer(this->uniformBuffer[0] , (void**)&pCBuffer0);
		if(pCBuffer0)
		{
			memcpy(&pCBuffer0->rotation , &rotation[0] , sizeof(HQMatrix3x4));
			memcpy(&pCBuffer0->viewProj, viewProj , sizeof(HQMatrix4));
		}
		pDevice->GetShaderManager()->UnmapUniformBuffer(this->uniformBuffer[0] );
		
		pDevice->GetShaderManager()->SetUniformMatrix("rotation" , &scale , 1);

		pDevice->GetShaderManager()->SetUniformMatrix("boneMatrices" , mesh->GetBoneTransformMatrices() , mesh->GetNumBones());
#else
		pDevice->GetShaderManager()->SetUniform4Float("rotation" , scale , 3);

		pDevice->GetShaderManager()->SetUniform4Float("boneMatrices" , (float*)mesh->GetBoneTransformMatrices() , mesh->GetNumBones() * 3);
#endif
		pDevice->GetShaderManager()->SetUniformMatrix("viewProj" , *viewProj);


	}
	
#ifdef WIN32
	else if (API == D3D9_RENDERER)
	{
		/*
		BUFFER * pCBuffer = NULL;

		pDevice->GetShaderManager()->MapUniformBuffer(this->uniformBuffer , (void**)&pCBuffer);
		if(pCBuffer)
		{
			memcpy(&pCBuffer->rotation , &rotation[0] , sizeof(HQMatrix3x4));
			memcpy(&pCBuffer->viewProj, viewProj , sizeof(HQMatrix4));
		}
		pDevice->GetShaderManager()->UnmapUniformBuffer(this->uniformBuffer );
		pDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 10 , this->uniformBuffer);
		*/
		
		pDevice->GetShaderManager()->SetUniformMatrix("viewProj" , *viewProj);
		pDevice->GetShaderManager()->SetUniformMatrix("rotation" , &scale , 1);
		pDevice->GetShaderManager()->SetUniformMatrix("boneMatrices" , mesh->GetBoneTransformMatrices() , mesh->GetNumBones());
		
		/*
		//pDevice->GetShaderManager()->SetUniform4Float("rotation" , rotation[0] , 3);
		pDevice->GetVertexStreamManager()->SetVertexInputLayout(vertexLayout[2]);
		pDevice->GetShaderManager()->ActiveProgram(HQ_NOT_USE_SHADER);
		HQMatrix4 rot4x4(rotation[0]);
		rot4x4.Transpose();
		//pDevice->GetShaderManager()->SetUniformInt(HQ_LIGHTING_ENABLE , HQ_FALSE);
		//pDevice->GetShaderManager()->SetUniformInt(HQ_TEXTURE_ENABLE , HQ_FALSE);
		pDevice->GetShaderManager()->SetUniformMatrix(HQ_WORLD , rot4x4);
		pDevice->GetShaderManager()->SetUniformMatrix(HQ_VIEW , HQMatrix4::identity);
		pDevice->GetShaderManager()->SetUniformMatrix(HQ_PROJECTION , viewProj);
		*/
	}

	else if (API == D3D10_RENDERER || API == D3D11_RENDERER)
	{
		/*
		BUFFER * pCBuffer0 = NULL;

		pDevice->GetShaderManager()->MapUniformBuffer(this->uniformBuffer[0] , (void**)&pCBuffer0);
		if(pCBuffer0)
		{
			memcpy(&pCBuffer0->rotation , &rotation[0] , sizeof(HQMatrix3x4));
			memcpy(&pCBuffer0->viewProj, viewProj , sizeof(HQMatrix4));
		}
		pDevice->GetShaderManager()->UnmapUniformBuffer(this->uniformBuffer[0] );
		*/
		BUFFER2 * pTBuffer0 = NULL;
#if TEXTURE_BUFFER
#		pragma message ("use TEXTURE_BUFFER")
		pDevice->GetTextureManager()->MapTextureBuffer(textureBuffer , (void**)&pTBuffer0);
#else
		pDevice->GetShaderManager()->MapUniformBuffer(this->uniformBuffer[1] , (void**)&pTBuffer0);
#endif
		if(pTBuffer0)
		{
			const HQMatrix3x4 *boneMatrices = mesh->GetBoneTransformMatrices();
			hquint32 numBones = mesh->GetNumBones();

			memcpy(&pTBuffer0->rotation , &scale , sizeof(HQMatrix3x4));
#ifdef WIN32
			//memcpy(&pTBuffer0->bones , meshX->GetBoneMatrices() , mesh->GetNumBones() * sizeof(HQMatrix3x4));
#endif
			memcpy(&pTBuffer0->bones , boneMatrices , numBones * sizeof(HQMatrix3x4));
			memcpy(&pTBuffer0->viewProj, viewProj , sizeof(HQMatrix4));
		}
#if TEXTURE_BUFFER
		pDevice->GetTextureManager()->UnmapTextureBuffer(textureBuffer );
#else
		pDevice->GetShaderManager()->UnmapUniformBuffer(this->uniformBuffer[1] );
#endif

	}
	
#endif //#ifdef WIN32


	pDevice->GetStateManager()->ActiveDepthStencilState(dsState);

	//pDevice->DrawIndexed(4 , 6 , 0);
	//pDevice->Draw(4 , 0);

	mesh->DrawInOneCall();

	pDevice->EndRender();



	pDevice->GetStateManager()->ActiveDepthStencilState(0);

	//HQTimer::Sleep(0.5f);

}

#if	defined WIN32 || defined APPLE || defined LINUX || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

void Game::KeyPressed(HQKeyCodeType keycode)
{
	static bool playmusic = true; 
	static bool pauseMusic = false;
	switch(keycode)
	{
	case HQKeyCode::M:
		playmusic = !playmusic;
#ifndef DISABLE_AUDIO
		if (playmusic)
			audio->GetSourceController(music)->Play();
		else
			audio->GetSourceController(music)->Stop();
#endif
		break;
	case HQKeyCode::P:
		pauseMusic = !pauseMusic;
#ifndef DISABLE_AUDIO
		if (!pauseMusic)
			audio->GetSourceController(music)->Play();
		else
			audio->GetSourceController(music)->Pause();
#endif
		break;
	case HQKeyCode::L:
#ifndef DISABLE_AUDIO
		audio->GetSourceController(music)->Play(HQ_FALSE);
#endif
		break;
	case HQKeyCode::LSHIFT:
		OutputDebugStringA("lshift\n");
		break;
	case HQKeyCode::RSHIFT:
		OutputDebugStringA("rshift\n");
		break;
	case HQKeyCode::LCONTROL:
		OutputDebugStringA("lctrl\n");
		break;
	case HQKeyCode::RCONTROL:
		OutputDebugStringA("rctrl\n");
		break;
#ifdef WIN32
	case HQKeyCode::LALT:
		OutputDebugStringA("lalt\n");
		break;
	case HQKeyCode::RALT:
		OutputDebugStringA("ralt\n");
		break;
#endif
#if !(defined GLES || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	case HQKeyCode::SPACE:

		pDevice->SetDisplayMode(800, 600 , !pDevice->IsWindowed());
		OnResetDevice();
		break;
	case HQKeyCode::Q:
		pDevice->SetDisplayMode(1280, 800 , !pDevice->IsWindowed());
		OnResetDevice();
#endif
		break;
	case HQKeyCode::TAB:
		HQEngineApp::GetInstance()->EnableMouseCursor(!HQEngineApp::GetInstance()->IsMouseCursorEnabled());
		break;
	case HQKeyCode::DIVIDE:
		OutputDebugStringA("/\n");
		break;
	case HQKeyCode::ESCAPE:
		HQEngineApp::GetInstance()->Stop();
		break;
	}
}

void Game::KeyReleased(HQKeyCodeType keycode)
{
	switch(keycode)
	{
		case HQKeyCode::NUM1:
			curTexture = texture;
			break;
		case HQKeyCode::NUM2:
			curTexture = temp[0];
			break;
		case HQKeyCode::NUM3:
			curTexture = temp[1];
			break;
		case HQKeyCode::NUM4:
			curTexture = temp[2];
			break;
		case HQKeyCode::LSHIFT:
			OutputDebugStringA("lshift released\n");
			break;
		case HQKeyCode::RSHIFT:
			OutputDebugStringA("rshift released\n");
			break;
		case HQKeyCode::LCONTROL:
			OutputDebugStringA("lctrl released\n");
			break;
		case HQKeyCode::RCONTROL:
			OutputDebugStringA("rctrl released\n");
			break;
#ifdef WIN32
		case HQKeyCode::LALT:
			OutputDebugStringA("lalt released\n");
			break;
		case HQKeyCode::RALT:
			OutputDebugStringA("ralt released\n");
			break;
#endif
	}
}

void Game::MousePressed( HQMouseKeyCodeType button, const HQPointi &point) 
{
	switch(button)
	{
	case HQKeyCode::LBUTTON:
		OutputDebugStringA("lmouse\n");
		break;
	case HQKeyCode::RBUTTON:
		OutputDebugStringA("rmouse\n");
		break;
	case HQKeyCode::MBUTTON:
		OutputDebugStringA("mmouse\n");
		break;
	}
}
void Game::MouseReleased( HQMouseKeyCodeType button, const HQPointi &point) 
{
	switch(button)
	{
		case HQKeyCode::LBUTTON:
			OutputDebugStringA("lmouse released\n");
			break;
		case HQKeyCode::RBUTTON:
			OutputDebugStringA("rmouse released\n");
			break;
		case HQKeyCode::MBUTTON:
			OutputDebugStringA("mmouse released\n");
			break;
	}
}
void Game::MouseMove( const HQPointi &point) 
{
	char info[256];
	sprintf(info , "%d , %d\n", point.x, point.y);
	OutputDebugStringA(info);
}
void Game::MouseWheel( hq_float32 delta, const HQPointi &point)
{
	char info[10];
	int len;
#ifdef WIN32
	len = _snprintf(info , 10 , "%.5f\n", delta);
#else
	len = snprintf(info , 10 , "%.5f\n", delta);
#endif
	if (len >= 10 || len == -1)
	{
		info[8] = '\n';
		info[9] = '\0';
	}
	OutputDebugStringA(info);
}
#endif//defined WIN32 || defined APPLE || defined LINUX || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

void Game::TouchBegan(const HQTouchEvent &event) 
{
	static int index = 0;
	index = (index + 1) % 4;
	switch (index)
	{
		case 0:
			curTexture = texture;
			break;
		case 1:
			curTexture = temp[0];
			break;
		case 2:
			curTexture = temp[1];
			break;
		case 3:
			curTexture = temp[2];
			break;
	}
	
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchBegan : numTouches:%u first touch: id=%d, position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y);
	OutputDebugStringA(info);
}

void Game::TouchMoved(const HQTouchEvent &event) 
{
	
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	const HQPointf &prev_point = event.GetPrevPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchMoved : numTouches:%u first touch: id=%d, position={%.4f , %.4f}, prev_position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y, prev_point.x, prev_point.y);
	OutputDebugStringA(info);
}
void Game::TouchEnded(const HQTouchEvent &event) 
{
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	const HQPointf &prev_point = event.GetPrevPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchEnded : numTouches:%u first touch: id=%d, position={%.4f , %.4f}, prev_position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y, prev_point.x, prev_point.y);
	OutputDebugStringA(info);
}
void Game::TouchCancelled(const HQTouchEvent &event)
{
	char info[256];
	const HQPointf &point = event.GetPosition(0);
	hqint32 touchID = event.GetTouchID(0);
	sprintf(info , "TouchCancelled : numTouches:%u first touch: id=%d, position={%.4f , %.4f}\n", event.GetNumTouches(), touchID, point.x, point.y);
	OutputDebugStringA(info);
}

void Game::OnDestroy()
{
	HQMutex::ScopeLock sl(mutex);
	app_exit = true;
	
	OutputDebugStringA("app will be destroyed!\n");
}

void Game::OnPause() 
{
	OutputDebugStringA("app is paused!\n");

	audio->GetSourceController(music)->Pause();
}

void Game::OnResume()
{
	OutputDebugStringA("app is resumed!\n");

	audio->GetSourceController(music)->Play();
}

void Game::ChangedToPortrait() 
{
	m_offsetX = m_offsetY = 0;
	OutputDebugStringA("portrait mode\n");
}
void Game::ChangedToPortraitUpsideDown() 
{
	m_offsetX = pDevice->GetWidth()/2;
	m_offsetY = -(int)pDevice->GetHeight()/2;
	OutputDebugStringA("portrait upside down mode\n");
}
///right side is at top
void Game::ChangedToLandscapeLeft() 
{
	m_offsetX = 0;
	m_offsetY = -(int)pDevice->GetHeight()/2;
	OutputDebugStringA("landscape left mode\n");
}
///left side is at top
void Game::ChangedToLandscapeRight()
{
	m_offsetX = pDevice->GetWidth()/2;
	m_offsetY = 0;
	OutputDebugStringA("landscape right mode\n");
}

#if defined ANDROID || defined HQ_WIN_PHONE_PLATFORM
bool Game::BackButtonPressed()
{
	OutputDebugStringA("back key is pressed!\n");
	app_exit = true;

	return false;
}
#endif

#endif //#if	defined WIN32 || defined APPLE || defined LINUX
