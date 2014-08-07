
#include "../../HQEngine/Source/HQEngineApp.h"
#include "GpuProfilerD3D11.h"

#include <stdio.h>

/*-------------------SubframeQuery---------------------*/
SubframeQuery::SubframeQuery()
:startTimeQuery(0), endTimeQuery(0)
{
}
SubframeQuery::SubframeQuery(const SubframeQuery& src)
{
	startTimeQuery = src.startTimeQuery;
	endTimeQuery = src.startTimeQuery;

	startTimeQuery->AddRef();
	endTimeQuery->AddRef();
}
SubframeQuery::~SubframeQuery()
{
	Release();
}

SubframeQuery& SubframeQuery::operator = (const SubframeQuery& src)
{
	Release();

	startTimeQuery = src.startTimeQuery;
	endTimeQuery = src.startTimeQuery;

	startTimeQuery->AddRef();
	endTimeQuery->AddRef();

	return *this;
}

void SubframeQuery::Release()
{
	SafeRelease(startTimeQuery);
	SafeRelease(endTimeQuery);
}

void SubframeQuery::Init()
{
	HQRenderDevice * renderDevice = HQEngineApp::GetInstance()->GetRenderDevice();
	ID3D11Device * d3d11device = (ID3D11Device*)renderDevice->GetRawHandle();

	D3D11_QUERY_DESC desc;
	desc.MiscFlags = 0;
	desc.Query = D3D11_QUERY_TIMESTAMP;
	d3d11device->CreateQuery(&desc, &startTimeQuery);
	d3d11device->CreateQuery(&desc, &endTimeQuery);
}

void SubframeQuery::Start(ID3D11DeviceContext * d3dcontext)
{
	if (startTimeQuery == NULL)
		Init();

	d3dcontext->End(startTimeQuery);
}

void SubframeQuery::End(ID3D11DeviceContext * d3dcontext)
{
	if (endTimeQuery == NULL)
		Init();
	d3dcontext->End(endTimeQuery);
}

/*-------------------GpuProfilerD3D11------------------*/
GpuProfilerD3D11::GpuProfilerD3D11()
: m_cachedQueryDataValid(false)
{
	HQRenderDevice * renderDevice = HQEngineApp::GetInstance()->GetRenderDevice();
	ID3D11Device * d3d11device = (ID3D11Device*)renderDevice->GetRawHandle();
	d3d11device->GetImmediateContext(&m_d3d11context);

	D3D11_QUERY_DESC desc;
	desc.MiscFlags = 0;
	desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
	d3d11device->CreateQuery(&desc, &m_frameDisjointQuery);
}

GpuProfilerD3D11::~GpuProfilerD3D11()
{
	m_d3d11context->Release();
	m_frameDisjointQuery->Release();
}

void GpuProfilerD3D11::BeginFrame()
{
	m_cachedQueryDataValid = false;
	m_d3d11context->Begin(m_frameDisjointQuery);
	m_frameTimeQuery.Start(m_d3d11context);
}
void GpuProfilerD3D11::EndFrame()
{
	m_frameTimeQuery.End(m_d3d11context);
	m_d3d11context->End(m_frameDisjointQuery);
}

void GpuProfilerD3D11::BeginSubFrame(const char *name)
{
	m_subframes[name].Start(m_d3d11context);
}
void GpuProfilerD3D11::EndSubFrame(const char *name)
{
	m_subframes[name].End(m_d3d11context);
}

HQTime GpuProfilerD3D11::GetElapsedTime(const char *subframeName)
{
	return GetElapsedTime(m_subframes[subframeName]);
}

HQTime GpuProfilerD3D11::GetElapsedTime(SubframeQuery &subframe)
{
	if (subframe.startTimeQuery == 0)
		return 0.f;
	if (m_cachedQueryDataValid == false)
	{
		while (m_d3d11context->GetData(m_frameDisjointQuery, NULL, 0, 0) == S_FALSE)
		{
			Sleep(1);// Wait a bit, but give other threads a chance to run
		}

		m_cachedQueryDataValid = true;

		m_d3d11context->GetData(m_frameDisjointQuery, &m_cachedFrameQueryData, sizeof(m_cachedFrameQueryData), 0);
		if (m_cachedFrameQueryData.Disjoint)
		{
			return 0.f;
		}
	}//if (m_cachedQueryDataValid == false)

	UINT64 tsBeginTime, tsEndTime;
	m_d3d11context->GetData(subframe.startTimeQuery, &tsBeginTime, sizeof(UINT64), 0);
	m_d3d11context->GetData(subframe.endTimeQuery, &tsEndTime, sizeof(UINT64), 0);

	//Convert to real time
	HQTime seconds = HQTime(tsEndTime - tsBeginTime) /
		HQTime(m_cachedFrameQueryData.Frequency);

	return seconds;
}

void GpuProfilerD3D11::DeleteSubframeProfile(const char *subframeName)
{
	m_subframes.erase(subframeName);
}

void GpuProfilerD3D11::Report(const char* reportFile)
{
	FILE* f = fopen(reportFile, "w");

	if (f)
	{
		HQTime frameTime = GetElapsedTime(m_frameTimeQuery);
		fprintf(f, "total time = %f ms\nfps = %f\n", frameTime * 1000.f, 1.0f / frameTime);

		HQTime collectedSubframesTime = 0.f;

		for (std::unordered_map<std::string, SubframeQuery>::iterator ite = m_subframes.begin();
			ite != m_subframes.end();
			++ite)
		{
			HQTime seconds = GetElapsedTime(ite->second);
			collectedSubframesTime += seconds;
			float ms = seconds * 1000.f;
			fprintf(f, "%s = %f ms\n", ite->first.c_str(), ms);
		}

		fprintf(f, "collected subframes time = %f ms\n", collectedSubframesTime * 1000.f);
		fprintf(f, "uncollected subframes time = %f ms\n", (frameTime - collectedSubframesTime) * 1000.f);
	}
	fclose(f);
}

void GpuProfilerD3D11::IterateCollectedSubframes(SubframeIteratorListener* listener)
{
	for (std::unordered_map<std::string, SubframeQuery>::iterator ite = m_subframes.begin();
		ite != m_subframes.end();
		++ite)
	{
		float seconds = GetElapsedTime(ite->second);
		listener->CollectSubframeReport(ite->first.c_str(), seconds);
	}
}