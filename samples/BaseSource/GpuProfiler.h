#ifndef GPU_PROFILER_H
#define GPU_PROFILER_H

#include "../../HQEngine/Source/HQTimer.h"

#include <string>

class GpuProfiler {
public:
	class SubframeIteratorListener
	{
	public:
		virtual void CollectSubframeReport(const char* name, HQTime elapsedTime) = 0;
	};

	virtual ~GpuProfiler() {}

	virtual void BeginFrame() = 0;
	virtual void EndFrame() = 0;

	virtual void BeginSubFrame(const char *name) = 0;
	virtual void EndSubFrame(const char *name) = 0;

	virtual HQTime GetFrameElapsedTime() = 0;
	virtual HQTime GetElapsedTime(const char *subframeName) = 0;

	virtual void DeleteSubframeProfile(const char *subframeName) = 0;

	virtual void Report(const char* reportFile) = 0;

	///iterate through collected subframes
	virtual void IterateCollectedSubframes(SubframeIteratorListener* listener) = 0;
};

GpuProfiler * CreateGpuProfiler();

class GpuScopeProfiler{
public:
	GpuScopeProfiler(GpuProfiler *profiler, const char* scopeName, bool enable)
	{
		m_profiler = profiler;
		m_scopeName = scopeName;
		m_enable = enable;
		if (m_enable)
			m_profiler->BeginSubFrame(scopeName);
	}

	~GpuScopeProfiler()
	{
		if (m_enable)
			m_profiler->EndSubFrame(m_scopeName.c_str());
	}
private:
	GpuProfiler *m_profiler;
	std::string m_scopeName;
	bool m_enable;
};

#endif