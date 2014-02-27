#ifndef RENDER_LOOP_H
#define RENDER_LOOP_H

#include "HQEngineApp.h"
#include "HQMeshNode.h"
#include "HQCamera.h"

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


/*-------rendering loop-----------*/
class RenderLoop: public HQEngineRenderDelegate{
public:
	RenderLoop();
	~RenderLoop();

	void Render(HQTime dt);

private:
	void DepthPassInit();//init render targets
	void LowresPassInit();
	void FinalPassInit();
	void DecodeNoiseMap();//decode noise map from RGBA image to float texture

	void DepthPassRender(HQTime dt);//render depth pass
	void LowresPassRender(HQTime dt);
	void FinalPassRender(HQTime dt);

	HQMeshNode* m_model;
	HQRenderDevice *m_pRDevice;
	HQCamera * m_camera;
	HQSceneNode* m_scene;//the whole scene

	SpotLight * m_light;

	//for depth pass rendering
	HQRenderTargetDesc depth_pass_renderTargets[4];
	hquint32 depth_pass_rtTextures[4];//textures associating with render targets
	hquint32 depth_pass_depth_buffer;//depth buffer for depth pass
	hquint32 depth_pass_program;//shader program

	//for low res rendering
	HQRenderTargetDesc lowres_pass_rTargets[3];//render targets
	hquint32 lowres_pass_rtTextures[3];//textures associating with this render targets
	hquint32 lowres_depth_buffer;//depth buffer for lowres pass
	hquint32 lowres_pass_program;//shader program

	//for final gathering
	hquint32 final_pass_program;

	//for low res rendering and final gathering
	hquint32 m_noise_map;//noise map texture
	hquint32 point_sstate;//point sampling state
	hquint32 border_sstate;//black border sampling state
};

#endif