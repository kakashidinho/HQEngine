#include "GpuProfiler.h"
#include "../../HQEngine/Source/HQEngineApp.h"

#if _MSC_VER >= 1700
#	define D3D11_SUPPORTED
#	include "GpuProfilerD3D11.h"
#endif

#include <string.h>

GpuProfiler * CreateGpuProfiler()
{
	const char * renderAPI = HQEngineApp::GetInstance()->GetRendererType();
#ifdef D3D11_SUPPORTED
	if (!strcmp(renderAPI, "D3D11"))
		return new GpuProfilerD3D11();
#else
#error "need implemenetation"
#endif

	return NULL;
}