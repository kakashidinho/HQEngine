/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "RenderLoop.h"

//initilize resources for depth pass rendering
void RenderLoop::DepthPassInit(){
	HQRenderTargetDesc depth_pass_renderTargets[4];

	//depth render target
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		DEPTH_PASS_RT_WIDTH, DEPTH_PASS_RT_HEIGHT,
		false,
		HQ_RTFMT_R_FLOAT32,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&depth_pass_renderTargets[0].renderTargetID,
		&depth_pass_rtTextures[0]);

	//world space position buffer render target
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		DEPTH_PASS_RT_WIDTH, DEPTH_PASS_RT_HEIGHT,
		false,
		HQ_RTFMT_RGBA_FLOAT64,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&depth_pass_renderTargets[1].renderTargetID,
		&depth_pass_rtTextures[1]);

	//world space normal buffer render target
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		DEPTH_PASS_RT_WIDTH, DEPTH_PASS_RT_HEIGHT,
		false,
		HQ_RTFMT_RGBA_FLOAT64,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&depth_pass_renderTargets[2].renderTargetID,
		&depth_pass_rtTextures[2]);

	//flux buffer
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetTexture(
		DEPTH_PASS_RT_WIDTH, DEPTH_PASS_RT_HEIGHT,
		false,
		HQ_RTFMT_RGBA_32,
		HQ_MST_NONE,
		HQ_TEXTURE_2D,
		&depth_pass_renderTargets[3].renderTargetID,
		&depth_pass_rtTextures[3]);

	// depth stencil buffer
	m_pRDevice->GetRenderTargetManager()->CreateDepthStencilBuffer(
			DEPTH_PASS_RT_WIDTH, DEPTH_PASS_RT_HEIGHT,
			HQ_DSFMT_DEPTH_24_STENCIL_8,
			HQ_MST_NONE,
			&depth_pass_depth_buffer
		);

	//create render target group
	m_pRDevice->GetRenderTargetManager()->CreateRenderTargetGroup(
			depth_pass_renderTargets,
			depth_pass_depth_buffer,
			4,
			&depth_pass_rtGroupID
		);

	//create shader program
	hquint32 vid, pid;
	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_VERTEX_SHADER,
		API_BASED_SHADER_MODE(this->m_renderAPI_type),
		API_BASED_VSHADER_FILE(this->m_renderAPI_type, "depth-pass"),
		NULL,
		"VS",
		&vid);

	m_pRDevice->GetShaderManager()->CreateShaderFromFile(
		HQ_PIXEL_SHADER,
		API_BASED_SHADER_MODE(this->m_renderAPI_type),
		API_BASED_FSHADER_FILE(this->m_renderAPI_type, "depth-pass"),
		NULL,
		"PS",
		&pid);

	m_pRDevice->GetShaderManager()->CreateProgram(
		vid, pid, HQ_NULL_GSHADER, NULL, 
		&depth_pass_program);
}

void RenderLoop::DepthPassRender(HQTime dt){
	//switch to offscreen render targets
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(depth_pass_rtGroupID);

	//set viewport
	const HQViewPort viewport = {0, 0, DEPTH_PASS_RT_WIDTH, DEPTH_PASS_RT_HEIGHT};
	m_pRDevice->SetViewPort(viewport);

	//active shader program
	m_pRDevice->GetShaderManager()->ActiveProgram(this->depth_pass_program);

	//set light parameters
	m_pRDevice->GetShaderManager()->SetUniform3Float("lightPosition", m_light->position(), 1);
	m_pRDevice->GetShaderManager()->SetUniform3Float("lightDirection", m_light->direction(), 1);
	m_pRDevice->GetShaderManager()->SetUniformFloat("lightFalloff", m_light->falloff);
	m_pRDevice->GetShaderManager()->SetUniformFloat("lightPCosHalfAngle", pow(cosf(m_light->angle * 0.5f), m_light->falloff));
	m_pRDevice->GetShaderManager()->SetUniform4Float("lightDiffuse", m_light->diffuseColor, 1);

	//set light camera's matrices
	this->SetUniformMatrix3x4("worldMat", m_model->GetWorldTransform());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("viewMat", m_light->lightCam().GetViewMatrix(), 1);
	m_pRDevice->GetShaderManager()->SetUniformMatrix("projMat", m_light->lightCam().GetProjectionMatrix(), 1);

	m_pRDevice->BeginRender(HQ_TRUE, HQ_TRUE, HQ_FALSE);
	//render the scene
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i){
		m_model->BeginRender();
		m_pRDevice->GetShaderManager()->SetUniform4Float("materialDiffuse", m_model->GetSubMeshInfo(i).colorMaterial.diffuse, 1);
		m_model->DrawSubMesh(i);
		m_model->EndRender();
	}
	m_pRDevice->EndRender();

	//switch to default render target
	m_pRDevice->GetRenderTargetManager()->ActiveRenderTargets(HQ_NULL_ID);

	m_pRDevice->GetShaderManager()->ActiveProgram(HQ_NOT_USE_SHADER);

}