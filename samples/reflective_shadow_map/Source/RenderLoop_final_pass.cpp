#include "RenderLoop.h"

void RenderLoop::FinalPassInit()
{
	//shader program
	hquint32 vid, pid;
	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_VERTEX_SHADER,
		"../Data/final-gathering.cg",
		NULL,
		false,
		"VS",
		&vid);

	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_PIXEL_SHADER,
		"../Data/final-gathering.cg",
		NULL,
		false,
		"PS",
		&pid);

	m_pRDevice->GetShaderManager()->CreateProgram(
		vid, pid, HQ_NULL_GSHADER, NULL, 
		&final_pass_program);
}

void RenderLoop::FinalPassRender(HQTime dt){
	
	const HQViewPort viewport = {0, 0, 600, 600};
	m_pRDevice->SetViewPort(viewport);

	//active shader program
	m_pRDevice->GetShaderManager()->ActiveProgram(this->final_pass_program);

	//set light parameters
	m_pRDevice->GetShaderManager()->SetUniform3Float("lightPosition", m_light->position(), 1);
	m_pRDevice->GetShaderManager()->SetUniformMatrix("lightViewMat", m_light->lightCam().GetViewMatrix());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("lightProjMat", m_light->lightCam().GetProjectionMatrix());
	//set light camera's matrices
	m_pRDevice->GetShaderManager()->SetUniformMatrix("worldMat", m_model->GetWorldTransform());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("viewMat", m_camera->GetViewMatrix(), 1);
	m_pRDevice->GetShaderManager()->SetUniformMatrix("projMat", m_camera->GetProjectionMatrix(), 1);
	
	//set textures
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 0, point_sstate);//point sampling state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 1, border_sstate);//black border state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 2, border_sstate);//black border state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 3, border_sstate);//black border state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 4, point_sstate);//point state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 5, point_sstate);//point state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 6, point_sstate);//point state
	m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 7, point_sstate);//point state
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 0, depth_pass_rtTextures[0]);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 1, depth_pass_rtTextures[1]);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 2, depth_pass_rtTextures[2]);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 3, depth_pass_rtTextures[3]);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 4, m_noise_map);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 5, lowres_pass_rtTextures[0]);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 6, lowres_pass_rtTextures[1]);
	m_pRDevice->GetTextureManager()->SetTexture(HQ_PIXEL_SHADER | 7, lowres_pass_rtTextures[2]);

	//start rendering
	m_pRDevice->BeginRender(HQ_TRUE, HQ_TRUE, HQ_FALSE);
	//render the scene
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i){
		m_model->BeginRender();
		m_pRDevice->GetShaderManager()->SetUniform4Float("materialDiffuse", m_model->GetSubMeshInfo(i).colorMaterial.diffuse, 1);
		m_model->DrawSubMesh(i);
		m_model->EndRender();
	}
	m_pRDevice->EndRender();

	m_pRDevice->GetShaderManager()->ActiveProgram(HQ_NOT_USE_SHADER);
}