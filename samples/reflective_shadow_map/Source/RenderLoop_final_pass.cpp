/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "RenderLoop.h"

void RenderLoop::FinalPassRender(HQTime dt){
	
	//activate the pass
	rsm_effect->GetPassByName("final-gathering")->Apply();

	const HQViewPort viewport = {0, 0, 600, 600};
	m_pRDevice->SetViewPort(viewport);

	//set light parameters
	m_pRDevice->GetShaderManager()->SetUniform3Float("lightPosition", m_light->position(), 1);
	m_pRDevice->GetShaderManager()->SetUniformMatrix("lightViewMat", m_light->lightCam().GetViewMatrix());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("lightProjMat", m_light->lightCam().GetProjectionMatrix());
	//set light camera's matrices
	this->SetUniformMatrix3x4("worldMat", m_model->GetWorldTransform());
	m_pRDevice->GetShaderManager()->SetUniformMatrix("viewMat", m_camera->GetViewMatrix(), 1);
	m_pRDevice->GetShaderManager()->SetUniformMatrix("projMat", m_camera->GetProjectionMatrix(), 1);
	
 
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

}