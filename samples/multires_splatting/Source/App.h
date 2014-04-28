#ifndef APP_H
#define APP_H

#include "../../BaseSource/BaseApp.h"
#include "../../BaseSource/Light.h"

#define RSM_DIM 512 //reflective shadow map size
#define WINDOW_SIZE 512
#define MIN_MAX_MIPMAP_FIRST_SIZE (WINDOW_SIZE / 2)
#define NUM_RESOLUTIONS 6


/*---------App-------------*/
class App : public BaseApp{
public:
	App();
	~App();

	virtual void Update(HQTime dt);
	virtual void RenderImpl(HQTime dt);
private:
	void KeyPressed(HQKeyCodeType keyCode);//implement HQEngineKeyListener
	void KeyReleased(HQKeyCodeType keyCode);//override BaseApp
	void InitSamplePattern();
	void InitSubplatsBuffers();
	void InitMipmaps();
	void InitFullscreenQuadBuffer();
	void GenMipmaps();
	void RefineSubsplats();
	void MultiresSplat();
	void Upsample();
	void DrawScene();
	void FinalPass();

	hquint32 m_vplsDim;//number of VPLs per RSM's dimension

	HQSharedPtr<HQMeshNode> m_model;
	HQSharedPtr<DiffuseSpotLight> m_light;

	HQEngineRenderEffect* m_effect;

	HQTexture * m_mipmapMin[NUM_RESOLUTIONS - 1];
	HQTexture * m_mipmapMax[NUM_RESOLUTIONS - 1];

	HQTexture * m_samplePatternTexture;

	HQTexture* m_subplatsIllumTexture[NUM_RESOLUTIONS];
	HQTexture* m_subplatsInterpolatedTexture[NUM_RESOLUTIONS];
	HQBufferUAV* m_subplatsRefineStepBuffer[NUM_RESOLUTIONS - 1];//subplats' buffers for each refinement step
	HQBufferUAV* m_finalSubsplatsBuffer;//buffer contains final list of subsplats
	HQBufferUAV* m_subsplatsCountBuffer;//buffer containing subsplats count. see {m_initialSubsplatsCounts}
	HQBufferUAV* m_dispatchArgsBuffer;//buffer containing dispatch's arguments. see {m_initialDispatchArgs}
	hquint32 m_initialSubsplatsCounts[NUM_RESOLUTIONS];//initial total subsplats count and subsplats count for each refinement step
	DispatchComputeArgs m_initialDispatchArgs[NUM_RESOLUTIONS];//initial dispatch arguments for indirect illumination step and refinement steps

	HQUniformBuffer* m_uniformViewInfoBuffer;
	HQUniformBuffer* m_uniformLightProtBuffer;
	HQUniformBuffer* m_uniformMaterialArrayBuffer;
	HQUniformBuffer* m_uniformMaterialIndexBuffer;
	HQUniformBuffer* m_uniformLightViewBuffer;
	HQUniformBuffer* m_uniformRefineStepBuffer;
	HQUniformBuffer* m_uniformRSMSamplesBuffer;

	HQVertexBuffer * m_fullscreenQuadBuffer;
	HQVertexLayout * m_fullscreenQuadVertLayout;
};

#endif