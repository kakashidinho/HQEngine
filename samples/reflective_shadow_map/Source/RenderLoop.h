/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef RENDER_LOOP_H
#define RENDER_LOOP_H

#include "HQEngineApp.h"
#include "HQMeshNode.h"
#include "HQCamera.h"

#include "../../../ThirdParty-mod/MyGUI/include/MyGUI.h"
#include "../../../ThirdParty-mod/MyGUI/include/MyGUI_HQEnginePlatform.h"

#if defined HQ_USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#if defined _MSC_VER
#include <cuda_d3d11_interop.h>
#include <cuda_d3d9_interop.h>
#endif
#include <cuda_gl_interop.h>
#endif//#if defined HQ_USE_CUDA

#define DEPTH_PASS_RT_WIDTH 512 //offscreen render target size
#define DEPTH_PASS_RT_HEIGHT 512 //offscreen render target size
#define LOWRES_RT_WIDTH 128 //offscreen render target size
#define LOWRES_RT_HEIGHT 128 //offscreen render target size


/*------basic light structure---------*/
struct BaseLight{
	BaseLight(const HQColor& diffuse,
			hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ)
		:	diffuseColor(diffuse),
			_position(HQVector4::New(posX, posY, posZ, 1.0f)) 
	{}
	virtual ~BaseLight() {delete _position;}

	HQVector4& position() {return *_position;}

	HQColor diffuseColor;
private:
	HQVector4* _position;//vector4 object needs to be aligned in memory, that's why we need to dynamically allocate it
};

/*------------spot light--------*/
struct SpotLight: public BaseLight{
	SpotLight(
			const HQColor& diffuse,
			hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,
			hqfloat32 dirX, hqfloat32 dirY, hqfloat32 dirZ,
			hqfloat32 angle,//cone angle in radian
			hqfloat32 falloff,
			hqfloat32 maxRange,
			HQRenderAPI renderApi
		);
	~SpotLight();

	HQVector4& direction() {return *_direction;}
	HQBaseCamera& lightCam() {return *_lightCam;}

	hqfloat32 angle;//cone's angle
	hqfloat32 falloff;//falloff factor
	hqfloat32 maxRange;
private:
	HQVector4* _direction;//vector4 object needs to be aligned in memory, that's why we need to dynamically allocate it
	HQBaseCamera* _lightCam;//light camera
};


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
class RenderLoop: public HQEngineRenderDelegate{
public:
	RenderLoop(const char* renderAPI);
	~RenderLoop();

	void Render(HQTime dt);

private:
#ifdef HQ_USE_CUDA
	void CudaGenerateNoiseMap();//generate noise map using CUDA
#else
	void DecodeNoiseMap();//decode noise map from RGBA image to float texture
#endif

	void DepthPassRender(HQTime dt);//render depth pass
	void LowresPassRender(HQTime dt);
	void FinalPassRender(HQTime dt);

	HQMeshNode* m_model;
	HQRenderDevice *m_pRDevice;
	HQCamera * m_camera;
	HQSceneNode* m_scene;//the whole scene

	SpotLight * m_light;

	HQEngineRenderEffect* rsm_effect;

	char m_renderAPI_name[6];//"D3D9" or "GL"
	HQRenderAPI m_renderAPI_type;

	HQUniformBuffer* m_uniformTransformBuffer;
	HQUniformBuffer* m_uniformLightProtBuffer;
	HQUniformBuffer* m_uniformMaterialBuffer;
	HQUniformBuffer* m_uniformLightViewBuffer;

	//GUI 
	MyGUI::HQEnginePlatform* m_guiPlatform;
	MyGUI::Gui *m_myGUI;
	MyGUI::TextBox* m_fpsTextBox;
};

#endif