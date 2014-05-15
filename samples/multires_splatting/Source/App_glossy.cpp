#include "App.h"

#if 0 && _MSC_VER >= 1800 && defined _DEBUG && defined DEBUG_ILLUM_BUFFER
#	define DONT_SAVE_VSGLOG_TO_TEMP
#	define VSG_DEFAULT_RUN_FILENAME L"graphics-capture.vsglog"
//#	define VSG_NODEFAULT_INSTANCE
#	include <vsgcapture.h> 
#	define DX_FRAME_CAPTURE
#endif

void App::GatherIndirectGlossyIllum()
{
	HQEngineTextureResource * depthMatMap = m_resManager->GetTextureResource("depth_materialID_img");
	HQEngineTextureResource * worldPosMap = m_resManager->GetTextureResource("world_pos_img");
	HQEngineTextureResource * worldNormalMap = m_resManager->GetTextureResource("world_normal_img");
	
	//generate g-buffer mipmaps
	m_pRDevice->GetRenderTargetManager()->GenerateMipmaps(depthMatMap->GetRenderTargetView());
	m_pRDevice->GetRenderTargetManager()->GenerateMipmaps(worldPosMap->GetRenderTargetView());
	m_pRDevice->GetRenderTargetManager()->GenerateMipmaps(worldNormalMap->GetRenderTargetView());

	//set output buffers
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(0, m_subsplatsIllumBuffer);
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(1, m_subsplatsTempIllumBuffer);
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(2, m_subsplatsCountBuffer);
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(3, m_subsplatsRefineStepsBuffer);
	m_pRDevice->GetShaderManager()->SetBufferUAVForComputeShader(4, m_dispatchArgsBuffer);

	/*---------------clear illumination buffer------------------*/
	HQShaderObject* clear_shader = m_resManager->GetShaderResource("clear_illum_buffer_cs")->GetShader();
	m_pRDevice->GetShaderManager()->ActiveComputeShader(clear_shader);
	//thread group size
	const hquint32 threadGroupSize = 64;

	//clear buffer
	m_pRDevice->DispatchCompute(m_totalSubsplats / threadGroupSize, 1, 1);
	m_pRDevice->BufferUAVBarrier();

	/*-----------splatting each VPL-------------*/
	HQEngineRenderPass* splatting_pass = m_effect->GetPassByName("multires_refine_vs_splatting_glossy");
	splatting_pass->Apply();

	static bool l_firstFrame = true;

	/*-------frame debugging-----------*/
#ifdef DX_FRAME_CAPTURE
	VsgDbg * vsgDbg = NULL;
	if (l_firstFrame)
	{
		vsgDbg = new VsgDbg(true);
		vsgDbg->BeginCapture();
	}
#endif

	const hquint32 totalVPLs = m_vplsDim * m_vplsDim;
	for (hquint32 vpl = 0; vpl < totalVPLs; ++vpl)
	{
		m_uniformVplSampleCoordsBuffer->Update(&m_samplePattern[vpl]);
		m_subsplatsCountBuffer->TransferData(m_initialSubsplatsCountsBuffer);//reset subsplats' counts
		m_dispatchArgsBuffer->TransferData(m_initialDispatchArgsBuffer);//reset dispatch arguments


		/*---------splatting------------------*/
		this->MultiresSplattingAndRefineGlossy();

#if defined DEBUG_ILLUM_BUFFER && defined DX_FRAME_CAPTURE
		//capture content of illumination buffer in 1st frame only
		if (l_firstFrame)
		{
			this->DbgIlluminationBuffer();
		}
#endif
	}

	/*--------upsample-------------*/
	hquint32 shouldAdditivelyBlend = 0;
	m_uniformInterpolatedInfoBuffer->Update(&shouldAdditivelyBlend);

	//interpolating between illumination resolutions
	this->Upsample();

	/*-------frame debugging-----------*/
#ifdef DX_FRAME_CAPTURE
	if (l_firstFrame)
	{
		vsgDbg->EndCapture();
		delete vsgDbg;
	}
#endif

	l_firstFrame = false;
}


void App::MultiresSplattingAndRefineGlossy()
{
	for (hquint32 level = 0; level < NUM_RESOLUTIONS - 1; ++level)
	{
		m_uniformLevelInfoBuffer->Update(&level);

		//dispatch
		m_pRDevice->DispatchComputeIndirect(m_dispatchArgsBuffer, level + 1);

		m_pRDevice->BufferUAVBarrier();
	}
}

#ifdef DEBUG_ILLUM_BUFFER
void App::DbgIlluminationBuffer()
{
	hquint32 size = WINDOW_SIZE >> (NUM_RESOLUTIONS - 1);
	//capture content of illumination buffer
	hquint32 *bufferData = new hquint32[m_totalSubsplats];
	m_subsplatsIllumBuffer->CopyContent(bufferData);//copy data to system memory

	//write to texture so that the buffer can be visualized in debugger
	hquint32 * bufferLevelData = bufferData;
	for (hquint32 i = 0; i < NUM_RESOLUTIONS; ++i)
	{
		m_subsplatIllumDbgTexture->SetLevelContent(NUM_RESOLUTIONS - 1 - i, bufferLevelData);

		//next level
		bufferLevelData += size * size;

		size <<= 1;
	}
	delete[] bufferData;
}
#endif