#include "RenderLoop.h"

void RenderLoop::LowresPassInit()
{
	HQRenderTargetDesc lowres_pass_rTargets[3];

	//illumination buffer
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT,
		false,
		HQ_RTFMT_RGBA_32,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&lowres_pass_rTargets[0].renderTargetID,
		&lowres_pass_rtTextures[0]);

	//world space position buffer
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT,
		false,
		HQ_RTFMT_RGBA_FLOAT128,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&lowres_pass_rTargets[1].renderTargetID,
		&lowres_pass_rtTextures[1]);

	//world space normal buffer
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT,
		false,
		HQ_RTFMT_RGBA_FLOAT64,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&lowres_pass_rTargets[2].renderTargetID,
		&lowres_pass_rtTextures[2]);

	// depth stencil buffer
	m_pRDevice->GetRenderTargetManager()->CreateDepthStencilBuffer(
			LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT,
			HQ_DSFMT_DEPTH_24_STENCIL_8,
			HQ_MST_NONE,
			&lowres_depth_buffer
		);

	//create render target group
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetGroup(
			lowres_pass_rTargets,
			lowres_depth_buffer,
			3,
			&lowres_pass_rtGroupID
		);

	//shader program
	hquint32 vid, pid;
	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_VERTEX_SHADER,
		API_BASED_SHADER_MODE(this->m_renderAPI_type),
		API_BASED_VSHADER_FILE(this->m_renderAPI_type, "lowres-pass"),
		NULL,
		"VS",
		&vid);

	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_PIXEL_SHADER,
		API_BASED_SHADER_MODE(this->m_renderAPI_type),
		API_BASED_FSHADER_FILE(this->m_renderAPI_type, "lowres-pass"),
		NULL,
		"PS",
		&pid);

	m_pRDevice->GetShaderManager()->CreateProgram(
		vid, pid, HQ_NULL_GSHADER, NULL, 
		&lowres_pass_program);
}

void RenderLoop::LowresPassRender(HQTime dt){
	//switch to offscreen render targets
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(lowres_pass_rtGroupID);

	//set viewport
	const HQViewPort viewport = {0, 0, LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT};
	m_pRDevice->SetViewPort(viewport);

	//active shader program
	m_pRDevice->GetShaderManager()->ActiveProgram(this->lowres_pass_program);

	//set light parameters
	m_pRDevice->GetShaderManager()->SetUniformMatrix("lightViewMat", m_light->lightCam().GetViewMatrix());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("lightProjMat", m_light->lightCam().GetProjectionMatrix());
	//set light camera's matrices
	this->SetUniformMatrix3x4("worldMat", m_model->GetWorldTransform());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("viewMat", m_camera->GetViewMatrix(), 1);
	m_pRDevice->GetShaderManager()->SetUniformMatrix("projMat", m_camera->GetProjectionMatrix(), 1);
	
	if (strcmp(this->m_renderAPI_name, "GL") != 0)
	{
		//set sampling states
		m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 0, point_sstate);//point sampling state
		m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 1, border_sstate);//black border state
		m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 2, border_sstate);//black border state
		m_pRDevice->GetStateManager()->SetSamplerState(HQ_PIXEL_SHADER | 3, border_sstate);//black border state
	}
	else{
		//set textures' sampling states
		m_pRDevice->GetStateManager()->SetSamplerState(m_noise_map, point_sstate);//point sampling state
		m_pRDevice->GetStateManager()->SetSamplerState(depth_pass_rtTextures[1], border_sstate);//black border state
		m_pRDevice->GetStateManager()->SetSamplerState(depth_pass_rtTextures[2], border_sstate);//black border state
		m_pRDevice->GetStateManager()->SetSamplerState(depth_pass_rtTextures[3], border_sstate);//black border state
	}
	//set textures
	m_pRDevice->GetTextureManager()->SetTextureForPixelShader(0, m_noise_map);
	m_pRDevice->GetTextureManager()->SetTextureForPixelShader(1, depth_pass_rtTextures[1]);
	m_pRDevice->GetTextureManager()->SetTextureForPixelShader(2, depth_pass_rtTextures[2]);
	m_pRDevice->GetTextureManager()->SetTextureForPixelShader(3, depth_pass_rtTextures[3]);

	//start rendering
	m_pRDevice->BeginRender(HQ_TRUE, HQ_TRUE, HQ_FALSE);
	//render the scene
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i){
		m_model->BeginRender();
		m_model->DrawSubMesh(i);
		m_model->EndRender();
	}
	m_pRDevice->EndRender();

	//switch to default render target
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(HQ_NULL_ID);

	m_pRDevice->GetShaderManager()->ActiveProgram(HQ_NOT_USE_SHADER);
}