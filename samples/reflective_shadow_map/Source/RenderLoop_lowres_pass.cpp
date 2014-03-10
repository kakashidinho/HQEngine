#include "RenderLoop.h"


void RenderLoop::LowresPassRender(HQTime dt){
	//activate the pass
	rsm_effect->GetPassByName("lowres-pass")->Apply();

	//set viewport
	const HQViewPort viewport = {0, 0, LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT};
	m_pRDevice->SetViewPort(viewport);

	//set light parameters
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
		m_model->DrawSubMesh(i);
		m_model->EndRender();
	}
	m_pRDevice->EndRender();
}