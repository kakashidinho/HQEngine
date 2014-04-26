#include "../HQEngineApp.h"

#include <iostream>
#include <fstream>
#include <string>

struct UniformBlock{
	float translation[2];
};

class Game : public HQEngineRenderDelegate {
public:
	Game();

	void Render(HQTime dt);
private:
	HQEngineApp *m_app;
	HQRenderDevice *m_pRDevice;
	HQVertexBufferUAV * m_vertexBuffer;
	HQDrawIndirectArgsBuffer *m_indirectDrawBuffer;
	HQBufferUAV* m_counterBuffer;
	HQVertexLayout *m_vertexLayout;
	HQUniformBuffer* m_uBuffer;
};

Game::Game()
{
	m_app = HQEngineApp::GetInstance();
	m_pRDevice = m_app->GetRenderDevice();

	m_pRDevice->SetClearColorf(1, 1, 1, 1);
	m_pRDevice->SetFullViewPort();

	//load resources
	m_app->GetResourceManager()->AddResourcesFromFile("resources.script");//API specific resources script
	m_app->GetResourceManager()->AddResourcesFromFile("resourcesCommon.script");


	//create counter buffer
	hquint32 zero = 0;
	m_app->GetResourceManager()->AddShaderBufferResource("counter_buffer", HQ_ESBT_SHADER_USE_ONLY, 1, sizeof(hquint32), &zero);
	m_counterBuffer = m_app->GetResourceManager()->GetShaderBufferResource("counter_buffer")->GetBuffer();

	//create UAV indirect draw buffer buffer to be written by compute shader
	m_indirectDrawBuffer = m_app->GetResourceManager()->GetShaderBufferResource("indirect_draw_buffer")->GetBuffer();

	//create UAV vertex buffer to be written by compute shader.
	//vertex format is {float2 position; float2 texcoords}
	m_vertexBuffer = m_app->GetResourceManager()->GetShaderBufferResource("vertex_buffer")->GetBuffer();
	
	//create vertex layout
	HQVertexAttribDescArray<2> vAttribsDesc;

	vAttribsDesc.SetPosition(0, 0, 0, HQ_VADT_FLOAT2);
	vAttribsDesc.SetTexcoord(1, 0, 2 * sizeof(float), HQ_VADT_FLOAT2, 0);

	m_app->GetResourceManager()->CreateVertexInputLayout(vAttribsDesc, vAttribsDesc.GetNumAttribs(), "draw-shader-vs", &m_vertexLayout);

	//create uniform buffer
	m_pRDevice->GetShaderManager()->CreateUniformBuffer(NULL, sizeof(UniformBlock), true, &m_uBuffer);

	m_pRDevice->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER, 0, m_uBuffer);

	//load effect 
	m_app->GetEffectManager()->AddEffectsFromFile("effects.script");
}

void Game::Render(HQTime dt){
	HQTexture* colorTexture = m_app->GetResourceManager()->GetTextureResource("color_texture")->GetTexture();
	/*--------------------use compute shader to generate texture, vertex buffer and draw arguments----------------------------------*/
	m_app->GetEffectManager()->GetEffect("draw")->GetPassByName("compute")->Apply();
	
	m_pRDevice->DispatchCompute(1, 1, 1);//run compute shader

	//place a barrier for UAV buffers and texture access
	m_pRDevice->BufferUAVBarrier();
	m_pRDevice->TextureUAVBarrier();

	/*-------------now draw-------------------*/
	m_app->GetEffectManager()->GetEffect("draw")->GetPassByName("draw")->Apply();
	m_pRDevice->GetVertexStreamManager()->SetVertexBuffer(m_vertexBuffer, 0, 4 * sizeof(float));
	m_pRDevice->GetVertexStreamManager()->SetVertexInputLayout(m_vertexLayout);


	m_pRDevice->SetPrimitiveMode(HQ_PRI_TRIANGLE_STRIP);

	m_pRDevice->BeginRender(HQ_TRUE, HQ_FALSE, HQ_FALSE);

	UniformBlock * transformBuf;
	/*----------draw quad-----------*/
	m_uBuffer->Map(&transformBuf);
	transformBuf->translation[0] = 0.0f;
	transformBuf->translation[1] = 0.0f;
	m_uBuffer->Unmap();

	m_pRDevice->DrawInstancedIndirect(m_indirectDrawBuffer);

	/*-----------draw triangle--------------*/
	m_uBuffer->Map(&transformBuf);
	transformBuf->translation[0] = 0.5f;//translate triangle to the right
	transformBuf->translation[1] = 0.0f;
	m_uBuffer->Unmap();

	m_pRDevice->DrawInstancedIndirect(m_indirectDrawBuffer, 1);

	m_pRDevice->EndRender();
}

/*----------------main-----------------------*/
int HQEngineMain(int argc, char** argv){
	//config renderer type
	std::string rendererAPI = "D3D11";
	std::ifstream stream("API.txt");
	if (stream.good())
	{
		stream >> rendererAPI;
	}

	stream.close();


	//create app
	HQEngineApp::CreateInstanceAndWindow(rendererAPI.c_str());
	HQEngineApp::GetInstance()->GetRenderDevice()->SetDisplayMode(800, 600, true);

	//create game instance
	Game game;

	HQEngineApp::GetInstance()->SetRenderDelegate(game);

	HQEngineApp::GetInstance()->Run();

	HQEngineApp::GetInstance()->Release();

	return 0;
}