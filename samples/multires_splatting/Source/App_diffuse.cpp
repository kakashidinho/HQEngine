#include "App.h"

void App::InitDiffuse()
{
	/*------init mipmap textures---------*/
	this->InitMinmaxMipmaps();

}


void App::InitMinmaxMipmaps()
{
	hquint32 size = MIN_MAX_MIPMAP_FIRST_SIZE;
	for (hqint32 i = NUM_RESOLUTIONS - 2; i >= 0; --i)
	{

		//min version
		m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R32G32B32A32_FLOAT, size, size, false, &m_mipmapMin[i]);

		//max version
		m_pRDevice->GetTextureManager()->AddTextureUAV(HQ_UAVTFMT_R32G32B32A32_FLOAT, size, size, false, &m_mipmapMax[i]);

		size >>= 1;
	}
}

//dynamically generate minmax mipmap
void App::GenMinmaxMipmaps()
{
	//thread group size
	const hquint32 threadGroupDim = 16;
	//mipmap size
	hquint32 width = MIN_MAX_MIPMAP_FIRST_SIZE, height = MIN_MAX_MIPMAP_FIRST_SIZE;

	//texture containing dinstance to camera
	HQEngineTextureResource* depthMap = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("depth_materialID_img");
	//texture containing world space normal	
	HQEngineTextureResource* normalMap = HQEngineApp::GetInstance()->GetResourceManager()->GetTextureResource("world_normal_img");

	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("mipmap_cs_1st_step")->GetShader();

	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

	/*-----------first step: generate first mipmap level----------------*/

	//read from deapth and world space g-buffer
	m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, depthMap->GetTexture());
	m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, normalMap->GetTexture());

	//output of compute shader
	m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(2, m_mipmapMin[NUM_RESOLUTIONS - 2]);
	m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(3, m_mipmapMax[NUM_RESOLUTIONS - 2]);

	m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

	m_pRDevice->TextureUAVBarrier();

	/*-------------generate the rest, starting from second hishest mipmap level----------------*/
	hqint32 level = NUM_RESOLUTIONS - 3;
	width >>= 1;
	height >>= 1;

	shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("mipmap_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
	while (level >= 0 && width > 0){
		//output of compute shader
		m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(2, m_mipmapMin[level]);
		m_pRDevice->GetTextureManager()->SetTextureUAVForComputeShader(3, m_mipmapMax[level]);

		//read from previous mipmap level
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, m_mipmapMin[level + 1]);
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, m_mipmapMax[level + 1]);


		m_pRDevice->DispatchCompute(max(width / threadGroupDim, 1), max(height / threadGroupDim, 1), 1);//run compute shader

		m_pRDevice->TextureUAVBarrier();

		width >>= 1;
		height >>= 1;
		level--;
	}
}


void App::RefineSubsplatsDiffuse()
{
	const hquint32 coarsestSize = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);
	const hquint32 numCoarsestSubsplats = coarsestSize * coarsestSize;


	m_subsplatsCountBuffer->TransferData(m_initialSubsplatsCountsBuffer);//reset subsplats' counts
	m_dispatchArgsBuffer->TransferData(m_initialDispatchArgsBuffer);//reset dispatch arguments

#if VERIFY_CODE
	{
		//verify buffer content
		hquint32 subsplatCountsBuffercontent[sizeof(m_initialSubsplatsCounts) / sizeof(hquint32)];
		DispatchComputeArgs dispatchArgsBuffercontent[sizeof(m_initialDispatchArgs) / sizeof(DispatchComputeArgs)];

		m_subsplatsCountBuffer->CopyContent(subsplatCountsBuffercontent);
		m_dispatchArgsBuffer->CopyContent(dispatchArgsBuffercontent);
	}
#endif

	//activate shader
	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("subsplat_refinement_cs")->GetShader();

	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);

	//subsplat count buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(0, m_subsplatsCountBuffer);

	//subsplats buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(1, m_subsplatsRefineStepsBuffer);

	//final subsplats list buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(2, m_finalSubsplatsBuffer);

	//indirect compute dispatch's arguments
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(3, m_dispatchArgsBuffer);

	hquint32 *pCurrentLevel;

	for (hquint32 level = 0; level < NUM_RESOLUTIONS - 1; ++level)
	{
		m_uniformLevelInfoBuffer->Map(&pCurrentLevel);
		*pCurrentLevel = level;
		m_uniformLevelInfoBuffer->Unmap();

		//mipmap containing min & max depth & normal for finer level
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 0, m_mipmapMin[level]);
		m_pRDevice->GetTextureManager()->SetTexture(HQ_COMPUTE_SHADER, 1, m_mipmapMax[level]);


		if (level == NUM_RESOLUTIONS - 2)
		{
			//final step
			shader = HQEngineApp::GetInstance()->GetResourceManager()
				->GetShaderResource("subsplat_final_refinement_cs")->GetShader();
			m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
		}

		//dispatch
		m_pRDevice->DispatchComputeIndirect(m_dispatchArgsBuffer, level + 1);

		m_pRDevice->BufferUAVBarrier();
	}
}

/*-------------multires splatting---------------------*/
void App::MultiresSplatDiffuse()
{
	//set output buffer
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(0, m_subsplatsIllumBuffer);

	//number of subsplats in final list
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(1, m_subsplatsCountBuffer, 0, 0);
	//final subsplats list
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(2, m_finalSubsplatsBuffer, 0, 0);

	/*---------------clear illumination buffer first------------------*/
	HQShaderObject* shader = HQEngineApp::GetInstance()->GetResourceManager()->GetShaderResource("clear_illum_buffer_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(shader);
	//thread group size
	const hquint32 threadGroupSize = 64;

	m_pRDevice->DispatchCompute(m_totalSubsplats / threadGroupSize, 1, 1);
	m_pRDevice->BufferUAVBarrier();

	/*------------now do multiresolution splatting-----------------*/
	m_effect->GetPassByName("multires_splatting_diffuse")->Apply();

	m_pRDevice->DispatchComputeIndirect(m_dispatchArgsBuffer);
	m_pRDevice->BufferUAVBarrier();
}

void App::GatherIndirectDiffuseIllum()
{
	//generate min-max mipmaps
	this->GenMinmaxMipmaps();

	//refine list of subsplats
	this->RefineSubsplatsDiffuse();

	//multiresolution splatting
	this->MultiresSplatDiffuse();

	//interpolating between illumination resolutions
	hquint32 shouldAdditivelyBlend = 0;
	m_uniformInterpolatedInfoBuffer->Update(&shouldAdditivelyBlend);
	this->Upsample();
}
