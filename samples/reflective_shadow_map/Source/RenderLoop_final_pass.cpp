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

	DiffuseMaterial * material;

	//start rendering
	m_pRDevice->Clear(HQ_TRUE, HQ_TRUE, HQ_FALSE);
	//render the scene
	m_model->BeginRender();
	for (hquint32 i = 0; i < m_model->GetNumSubMeshes(); ++i){

		//send material info to shader
		m_uniformMaterialBuffer->Map(&material);
		memcpy(material, &m_model->GetSubMeshInfo(i).colorMaterial.diffuse, sizeof(HQVector4));
		m_uniformMaterialBuffer->Unmap();
		
		m_model->DrawSubMesh(i);
	}
	m_model->EndRender();

}