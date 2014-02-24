#include "Game.h"
#include "../HQUtil.h"

#ifdef WIN32
#define USE_D3D9 0
#endif

struct Vertex{
	float x, y, z;
	float u, v;
};

HQLogStream *logStream = NULL;

/*---------------------------*/
Game::Game(JNIEnv *env, jobject canvas_ref)
: renderer(), texture(HQ_NOT_AVAIL_ID)
{
	logStream = HQCreateFileLogStream("log.txt");

	JAWT awt;
    JAWT_DrawingSurface *ds = NULL;
    JAWT_DrawingSurfaceInfo *dsi = NULL;
	JAWT_Win32DrawingSurfaceInfo *dsi_win = NULL;
	jboolean result;
    jint lock;

    // Get the AWT
    awt.version = JAWT_VERSION_1_4;
	if (env != NULL)
	{
		result = JAWT_GetAWT(env, &awt);
		// Get the drawing surface
		if (canvas_ref != NULL)
			ds = awt.GetDrawingSurface(env, canvas_ref);
	    
		// Lock the drawing surface
		if (ds != NULL)
		{
			lock = ds->Lock(ds);
			//assert((lock & JAWT_LOCK_ERROR) == 0);

			// Get the drawing surface info
			dsi = ds->GetDrawingSurfaceInfo(ds);
		}
	}
#ifdef WIN32
	// Get the platform-specific drawing info
	if (dsi != NULL)
		dsi_win = (JAWT_Win32DrawingSurfaceInfo*)dsi->platformInfo;

#if USE_D3D9
	renderer.CreateD3DDevice9();
#else
	renderer.CreateGLDevice();
#endif
	if (dsi_win != NULL)
	{
		renderer.GetDevice()->Init(dsi_win->hwnd, "rendererInit.txt", logStream, "GLSL-only");
	}
#endif
	if (ds != NULL)
	{
		// Free the drawing surface info
		if (dsi != NULL)
        ds->FreeDrawingSurfaceInfo(dsi);
		ds->Unlock(ds);
        // Free the drawing surface
        awt.FreeDrawingSurface(ds);
	}

	Init();

}

#ifdef WIN32
Game::Game(HWND hwnd)
: renderer(), texture(HQ_NOT_AVAIL_ID)
{
	logStream = HQCreateFileLogStream("log.txt");

#if USE_D3D9
	renderer.CreateD3DDevice9();
#else
	renderer.CreateGLDevice();
#endif
	renderer.GetDevice()->Init(hwnd, "rendererInit.txt", logStream, "GLSL-only");

	Init();
}
#endif

Game::~Game()
{
	renderer.Release();

	logStream->Close();
}

void Game::Init()
{
	renderer.GetDevice()->SetClearColorf(1.0f, 1.0f, 1.0f, 1.0f);
	renderer.GetDevice()->Clear(HQ_TRUE, HQ_FALSE, HQ_FALSE);
	renderer.GetDevice()->DisplayBackBuffer();

	/*--------------------------*/
	worldOrigin.x = 0;
	worldOrigin.y = 0;

	/*-------create 2d matrices----------*/
	CreateViewProjMatrix();

	/*------create shader----------*/
	HQShaderMacro macro[] = {
#if USE_D3D9
		"USE_D3D9" , "1" ,
#else
		"version" , "110" ,
#endif
		NULL , NULL};
	hquint32 vid, pid;
	renderer.GetDevice()->GetShaderManager()->CreateShaderFromFile(
		HQ_VERTEX_SHADER,
#if USE_D3D9
		HQ_SCM_CG_DEBUG,
#else
		HQ_SCM_GLSL,
#endif
		"shader/vs.txt",
		macro,
		"main",
		&vid);

	renderer.GetDevice()->GetShaderManager()->CreateShaderFromFile(
		HQ_PIXEL_SHADER,
#if USE_D3D9
		HQ_SCM_CG_DEBUG,
		"shader/ps.hlsl",
#else
		HQ_SCM_GLSL,
		"shader/ps.glsl",
#endif
		macro,
		"main",
		&pid);

	renderer.GetDevice()->GetShaderManager()->CreateProgram(vid, pid, HQ_NOT_USE_GSHADER, NULL, &this->program);


	/*------create vertex buffer & input layout----------*/
	HQVertexAttribDescArray<2> vdesc;
	vdesc.SetPosition(0, 0, 0,  HQ_VADT_FLOAT3);
	vdesc.SetTexcoord(1, 0, 3 * sizeof(float), HQ_VADT_FLOAT2, 0);

	renderer.GetDevice()->GetVertexStreamManager()->CreateVertexInputLayout(vdesc, 2, vid, &this->vinputlayout);

	renderer.GetDevice()->GetVertexStreamManager()->CreateVertexBuffer(NULL , 4 * sizeof (Vertex) , true , false , &this->quadVbuffer);
}

void Game::Paint()
{
	//renderer.GetDevice()->SetClearColori(rand() % 256, rand() % 256, rand() % 256, 255);
	renderer.GetDevice()->GetShaderManager()->ActiveProgram(this->program);

	renderer.GetDevice()->GetShaderManager()->SetUniformMatrix("MVP", worldMatrix * viewprojMatrix);

	renderer.GetDevice()->BeginRender(HQ_TRUE, HQ_FALSE, HQ_FALSE);
	//hightlight texture
	if (this->texture != HQ_NOT_AVAIL_ID)
	{
		hquint width, height;

		GetRenderer().GetDevice()->GetTextureManager()->GetTexture2DSize(this->texture, width, height);

		HQRectf rect = {0, 0, width, height};
		DrawRect(rect, this->texture);
	}
	renderer.GetDevice()->EndRender();
	renderer.GetDevice()->DisplayBackBuffer();
}

bool Game::LoadTexture(const char *file)
{
	hquint32 oldTexture =  this->texture;
	HQReturnVal re = renderer.GetDevice()->GetTextureManager()->AddTexture(file, 1.0f, NULL, 0, false, 
		HQ_TEXTURE_2D, &this->texture);
	
	bool ok = re == HQ_OK;

	if (oldTexture != HQ_NOT_AVAIL_ID)
	{
		if (ok)
		{
			if (oldTexture != this->texture)
				renderer.GetDevice()->GetTextureManager()->RemoveTexture(oldTexture);
		}
		else
			this->texture = oldTexture;
	}

	return ok;
}

void Game::OnWindowResized(hquint32 width, hquint32 height)
{
	GetRenderer().GetDevice()->OnWindowSizeChanged(width, height);
	HQViewPort vp = {0,0, width, height};
	GetRenderer().GetDevice()->SetViewPort(vp);

	CreateViewProjMatrix();
}

void Game::CreateViewProjMatrix()
{
	HQMatrix4 proj2D, view2D;
	HQMatrix4rOrthoProjLH(renderer.GetDevice()->GetWidth(), renderer.GetDevice()->GetHeight(), 0.0, 1.0f, &proj2D, 
#if USE_D3D9
		HQ_RA_D3D
#else
		HQ_RA_OGL
#endif
		);

	HQMatrix4rView(&HQVector4(1, 0, 0), 
					&HQVector4(0, -1, 0),
					&HQVector4(0, 0, 1),
					&HQVector4(renderer.GetDevice()->GetWidth()/ 2.0f, renderer.GetDevice()->GetHeight() / 2.0f, 0.0f),
					&view2D);

	this->viewprojMatrix = view2D * proj2D;
}


void Game::DrawRect(HQRectf rect, hquint32 texture)
{

	//set texture
	GetRenderer().GetDevice()->GetTextureManager()->SetTextureForPixelShader(0, texture);

	//fill vertex buffer
	Vertex *vertices;
	
	GetRenderer().GetDevice()->GetVertexStreamManager()->MapVertexBuffer(quadVbuffer, HQ_MAP_DISCARD, (void**)&vertices);

	vertices[0].x = rect.x;
	vertices[0].y = rect.y;
	vertices[0].z = 1.0f;
	vertices[0].u = 0.0f;
	vertices[0].v = 0.0f;

	vertices[1].x = rect.x + rect.w;
	vertices[1].y = rect.y;
	vertices[1].z = 1.0f;
	vertices[1].u = 1.0f;
	vertices[1].v = 0.0f;

	vertices[2].x = rect.x;
	vertices[2].y = rect.y + rect.h;
	vertices[2].z = 1.0f;
	vertices[2].u = 0.0f;
	vertices[2].v = 1.0f;

	vertices[3].x = rect.x + rect.w;
	vertices[3].y = rect.y + rect.h;
	vertices[3].z = 1.0f;
	vertices[3].u = 1.0f;
	vertices[3].v = 1.0f;

	GetRenderer().GetDevice()->GetVertexStreamManager()->UnmapVertexBuffer(quadVbuffer);

	GetRenderer().GetDevice()->GetVertexStreamManager()->SetVertexBuffer(quadVbuffer, 0, sizeof(Vertex));
	
	GetRenderer().GetDevice()->GetVertexStreamManager()->SetVertexInputLayout(vinputlayout);

	GetRenderer().GetDevice()->SetPrimitiveMode(HQ_PRI_TRIANGLE_STRIP);

	GetRenderer().GetDevice()->Draw(4, 0);

}