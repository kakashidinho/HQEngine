#include "GpuProfiler.h"

#include <d3d11.h>
#include <unordered_map>

#ifndef SafeRelease
#define SafeRelease(p) {if(p) {p->Release(); p = 0;}}
#endif

struct SubframeQuery {
	SubframeQuery();
	SubframeQuery(const SubframeQuery& src);
	~SubframeQuery();

	SubframeQuery& operator = (const SubframeQuery& src);

	void Start(ID3D11DeviceContext * d3dcontext);
	void End(ID3D11DeviceContext * d3dcontext);

	ID3D11Query* startTimeQuery;
	ID3D11Query* endTimeQuery;

private:
	void Release();
	void Init();
};

class GpuProfilerD3D11 : public GpuProfiler{
public:
	GpuProfilerD3D11();
	~GpuProfilerD3D11();

	virtual void BeginFrame();
	virtual void EndFrame();

	virtual void BeginSubFrame(const char *name);
	virtual void EndSubFrame(const char *name);

	virtual HQTime GetFrameElapsedTime() { return GetElapsedTime(m_frameTimeQuery); }
	virtual HQTime GetElapsedTime(const char *subframeName);

	virtual void DeleteSubframeProfile(const char *subframeName);

	virtual void Report(const char* reportFile);

	virtual void IterateCollectedSubframes(SubframeIteratorListener* listener);
private:
	HQTime GetElapsedTime(SubframeQuery &subframe);

	std::unordered_map<std::string, SubframeQuery> m_subframes;

	ID3D11DeviceContext* m_d3d11context;
	ID3D11Query* m_frameDisjointQuery;

	SubframeQuery m_frameTimeQuery;

	D3D10_QUERY_DATA_TIMESTAMP_DISJOINT m_cachedFrameQueryData;
	bool m_cachedQueryDataValid;
};