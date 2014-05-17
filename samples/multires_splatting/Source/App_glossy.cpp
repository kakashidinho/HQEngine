#include "App.h"

#if 0 && _MSC_VER >= 1800 && defined _DEBUG && defined DEBUG_ILLUM_BUFFER
#	define DONT_SAVE_VSGLOG_TO_TEMP
#	define VSG_DEFAULT_RUN_FILENAME L"graphics-capture.vsglog"
//#	define VSG_NODEFAULT_INSTANCE
#	include <vsgcapture.h> 
#	define DX_FRAME_CAPTURE
#endif

void App::CausticsGathering()
{
	//generate mipmap
	HQEngineTextureResource * fluxMap = m_resManager->GetTextureResource("rsm_flux_img");
	m_pRDevice->GetRenderTargetManager()->GenerateMipmaps(fluxMap->GetRenderTargetView());

	const hquint32 threadGroup_dim = 16;
	HQBufferUAV * causticsSamplesCountBuf = m_resManager->GetShaderBufferResource("caustics_samples_count")->GetBuffer();

	//filter out specular objects
	m_effect->GetPassByName("caustics_filter")->Apply();

	//reset caustics samples count
	hquint32 count = 0;
	causticsSamplesCountBuf->Update(&count);

	//run compute shader
	m_pRDevice->DispatchCompute((WINDOW_SIZE >> 4) / threadGroup_dim, (WINDOW_SIZE>>4) / threadGroup_dim, 1);

	//test
	//causticsSamplesCountBuf->CopyContent(&count);

	//gather illumination from specular objects
	m_effect->GetPassByName("caustics_pass")->Apply();

	//run compute shader
	m_pRDevice->DispatchCompute(WINDOW_SIZE / threadGroup_dim, WINDOW_SIZE / threadGroup_dim, 1);
}

#ifdef DEBUG_ILLUM_BUFFER
void App::DbgIlluminationBuffer()
{
	
}
#endif