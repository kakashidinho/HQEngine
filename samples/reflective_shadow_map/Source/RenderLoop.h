/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef RENDER_LOOP_H
#define RENDER_LOOP_H

#include "HQMeshNode.h"

#include "../../BaseSource/BaseApp.h"
#include "../../BaseSource/Light.h"

#if defined HQ_USE_CUDA
#include "../../../utilities/CUDA/src/HQCudaBinding.h"
#endif//#if defined HQ_USE_CUDA

#define DEPTH_PASS_RT_WIDTH 512 //offscreen render target size
#define DEPTH_PASS_RT_HEIGHT 512 //offscreen render target size
#define LOWRES_RT_WIDTH 128 //offscreen render target size
#define LOWRES_RT_HEIGHT 128 //offscreen render target size



/*-------structures that match those in shader--------*/
struct Transform {
	HQBaseMatrix3x4 worldMat;
	HQBaseMatrix4 viewMat;
	HQBaseMatrix4 projMat;
};

struct Material {
	float materialDiffuse[4];
};

struct LightProperties {
	
	float lightPosition[4];
	float lightDirection[4];
	float lightDiffuse[4];
	float lightFalloff_lightPCosHalfAngle[2];
};

struct LightView {
	HQBaseMatrix4 lightViewMat;//light camera's view matrix
	HQBaseMatrix4 lightProjMat;//light camera's projection matrix
};

/*-------rendering loop-----------*/
class RenderLoop : public BaseApp
{
public:
	RenderLoop(const char* rendererAPI,
				HQLogStream *logStream,
				const char * additionalAPISettings);
	~RenderLoop();

	
	//implement BaseApp
	virtual void Update(HQTime dt);
	virtual void RenderImpl(HQTime dt);

private:
#ifdef HQ_USE_CUDA
	void CudaGenerateNoiseMap();//generate noise map using CUDA
#else
	void DecodeNoiseMap();//decode noise map from RGBA image to float texture
	void ComputeShaderDecodeNoiseMap();//decode noise map from RGBA image to float texture using compute shader
#endif

	void DepthPassRender(HQTime dt);//render depth pass
	void LowresPassRender(HQTime dt);
	void FinalPassRender(HQTime dt);

	HQMeshNode* m_model;

	DiffuseSpotLight * m_light;

	HQEngineRenderEffect* rsm_effect;

	HQUniformBuffer* m_uniformTransformBuffer;
	HQUniformBuffer* m_uniformLightProtBuffer;
	HQUniformBuffer* m_uniformMaterialBuffer;
	HQUniformBuffer* m_uniformLightViewBuffer;

};

#endif