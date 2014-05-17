#ifndef APP_H
#define APP_H

#include "../../BaseSource/BaseApp.h"
#include "../../BaseSource/Light.h"

#define RSM_DIM 512 //reflective shadow map size
#define WINDOW_SIZE 512
#define MIN_MAX_MIPMAP_FIRST_SIZE (WINDOW_SIZE / 2)
#define NUM_RESOLUTIONS 5

#if defined _DEBUG
#	define DEBUG_ILLUM_BUFFER
#endif


/*---------App-------------*/
class App : public BaseApp{
public:
	App();
	~App();

	virtual void Update(HQTime dt);
	virtual void RenderImpl(HQTime dt);
private:
	void KeyPressed(HQKeyCodeType keyCode, bool repeat);//implement HQEngineKeyListener
	void KeyReleased(HQKeyCodeType keyCode);//override BaseApp
	void InitSamplePattern();
	void InitSubplatsBuffers();
	void InitFullscreenQuadBuffer();

	void Upsample();
	void DrawScene();
	void FinalPass();

	/*------for diffuse scene-------------*/
	void InitDiffuse();
	void GatherIndirectDiffuseIllum();//gather indirect illumination in diffuse scene
	void InitMinmaxMipmaps();
	void GenMinmaxMipmaps();
	void RefineSubsplatsDiffuse();
	void MultiresSplatDiffuse();

	/*------for caustics scene-------------*/
	void CausticsGathering();

#ifdef DEBUG_ILLUM_BUFFER
	void DbgIlluminationBuffer();
#endif
	/*-----------------------------------*/

	bool m_giOn;//global illumination on?
	bool m_diffuseScene;//is this diffuse scene
	bool m_dynamicLight;
	hquint32 m_vplsDim;//number of VPLs per RSM's dimension

	HQSharedPtr<HQMeshNode> m_model;
	HQSharedPtr<SpecularSpotLight> m_light;

	HQEngineRenderEffect* m_effect;

	HQTexture * m_mipmapMin[NUM_RESOLUTIONS - 1];
	HQTexture * m_mipmapMax[NUM_RESOLUTIONS - 1];

	HQTexture * m_samplePatternTexture;
	HQSharedArrayPtr< Float2>  m_samplePattern;
#ifdef DEBUG_ILLUM_BUFFER
	HQTexture * m_subsplatIllumDbgTexture;//for debuggin
#endif
	HQBufferUAV* m_subsplatsTempIllumBuffer;//subplat temp illumination buffer
	HQBufferUAV* m_subsplatsIllumBuffer;//multires subplat illumination buffer
	HQBufferUAV* m_subsplatsInterpolatedBuffer;//interpolated illumination buffer for every resolution except final one
	HQTexture* m_subsplatsFinalInterpolatedTexture;//final interpolated illumination texture
	HQBufferUAV* m_subsplatsRefineStepsBuffer;//subplats' buffers for every refinement step
	HQBufferUAV* m_finalSubsplatsBuffer;//buffer contains final list of subsplats
	HQBufferUAV* m_subsplatsCountBuffer;//buffer containing subsplats count. see {m_initialSubsplatsCounts}
	HQBufferUAV* m_dispatchArgsBuffer;//buffer containing dispatch's arguments. see {m_initialDispatchArgs}
	HQBufferUAV* m_initialSubsplatsCountsBuffer;//initial total subsplats count and subsplats count for each refinement step
	HQBufferUAV* m_initialDispatchArgsBuffer;//initial dispatch arguments for indirect illumination step and refinement steps
	hqfloat32 m_subsplatsRefineThreshold[3];//depth, normal, illumination thresholds
	hquint32 m_totalSubsplats;//total number of subsplats in all resolutions

	HQUniformBuffer* m_uniformViewInfoBuffer;
	HQUniformBuffer* m_uniformLightProtBuffer;
	HQUniformBuffer* m_uniformMaterialArrayBuffer;
	HQUniformBuffer* m_uniformMaterialIndexBuffer;
	HQUniformBuffer* m_uniformLightViewBuffer;
	HQUniformBuffer* m_uniformInterpolatedInfoBuffer;
	HQUniformBuffer* m_uniformLevelInfoBuffer;
	HQUniformBuffer* m_uniformRSMSamplesBuffer;
	HQUniformBuffer* m_uniformRefineThresholdBuffer;
	HQUniformBuffer* m_uniformVplSampleCoordsBuffer;

	HQVertexBuffer * m_fullscreenQuadBuffer;
	HQVertexLayout * m_fullscreenQuadVertLayout;
};

#endif