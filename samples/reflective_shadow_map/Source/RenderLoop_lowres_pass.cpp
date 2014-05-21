#include "RenderLoop.h"


void RenderLoop::LowresPassRender(HQTime dt){
	//activate the pass
	rsm_effect->GetPassByName("lowres-pass")->Apply();

	//set viewport
	const HQViewPort viewport = {0, 0, LOWRES_RT_WIDTH, LOWRES_RT_HEIGHT};
	m_pRDevice->SetViewport(viewport);

	//start rendering
	m_pRDevice->Clear(HQ_TRUE, HQ_TRUE, HQ_FALSE);
	//render the scene
	m_model->BeginRender();
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i){
		m_model->DrawSubMesh(i);
	}
	m_model->EndRender();
}