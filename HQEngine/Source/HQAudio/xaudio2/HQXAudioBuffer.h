#pragma once
#include "HQXAudio.h"
#include "../../HQThread.h"
#include "../../HQConditionVariable.h"
#include "../../HQLinkedList.h"


/*-------------HQXAudioBuffer-------------------------*/
class HQXAudioBuffer : public HQBaseAudioBuffer
{
public:
	HQXAudioBuffer(const char *fileName, HQBaseAudioDevice::AudioDataInfo &info)
		: HQBaseAudioBuffer(NORMAL_BUFFER), m_info(info)
	{
		//copy file name
		size_t len = strlen(fileName);
		m_audioFileName = HQ_NEW char [len + 1];
		strcpy(m_audioFileName, fileName);

		//init buffer
		m_xBuffer.pAudioData = (BYTE*)info.data;
		m_xBuffer.AudioBytes = info.size;
		m_xBuffer.Flags = 0;
		m_xBuffer.pContext = NULL;
		m_xBuffer.PlayBegin = 0;
		m_xBuffer.PlayLength = 0;
		m_xBuffer.LoopBegin = m_xBuffer.LoopLength = 0;
		m_xBuffer.LoopCount = XAUDIO2_LOOP_INFINITE;

		info.data = NULL;//take ownership of audio data

		m_playTime = (hqfloat32) ((double)info.samples / info.sampleRate);//playback duration

	}

	~HQXAudioBuffer()
	{
		if(m_info.data)
			free( m_info.data );
	}

	const XAUDIO2_BUFFER &GetBuffer() {return m_xBuffer;}
	const HQBaseAudioDevice::AudioDataInfo& GetDataInfo() {return m_info;}

private:
	XAUDIO2_BUFFER m_xBuffer;
	HQBaseAudioDevice::AudioDataInfo m_info;
};


/*-------------HQXAudioStreamBuffer-------------------------*/

class HQXAudioStreamBuffer : public HQBaseAudioBuffer, public HQThread
{
public:
	struct XAUDIO2_BUFFER_EX: public XAUDIO2_BUFFER
	{
		BYTE * pActualData;
		HQXAudioStreamBuffer * pParent;
	};

	HQXAudioStreamBuffer(
		const char *fileName, 
		HQBaseAudioDevice::AudioInfo &info,
		HQStreamBufferSupplier *supplier,//this object will take ownership of supplier
		size_t numSubBuffers,
		size_t subBufferSize);

	~HQXAudioStreamBuffer();

	size_t GetNumSubBuffers() const {return m_numSubBuffers;}
	size_t GetSubBufferSize() const {return m_subBufferSize;}
	const XAUDIO2_BUFFER &GetSubBuffer(size_t index) {return m_xSubBuffers[index];}
	const HQBaseAudioDevice::AudioInfo& GetInfo() {return m_info;}

	IXAudio2SourceVoice * GetAttachedSource() {return m_xAttachedSource;}
	void AttachSource(IXAudio2SourceVoice * source);

	void RestartStream(bool repushUsedList);

	void UpdateStream();

	static void OnSubBufferUsed(void *pBufferContext);
private:
	typedef HQLinkedList<XAUDIO2_BUFFER_EX*, HQPoolMemoryManager> SubBufferListType;

	virtual void Run();//worker thread implementation
	void FillSubBuffers();
	void FillSubBuffer(XAUDIO2_BUFFER_EX* buffer);
	void OnSubBufferUsed(XAUDIO2_BUFFER_EX* buffer);
	void RestartStreamNonLock(bool repushUsedList = true);

	XAUDIO2_BUFFER_EX *m_xSubBuffers;
	IXAudio2SourceVoice * m_xAttachedSource;//attached source
	size_t m_numSubBuffers;
	size_t m_subBufferSize;
	size_t m_streamPosition;
	HQStreamBufferSupplier *m_supplier;
	HQBaseAudioDevice::AudioInfo m_info;

	HQSimpleConditionVar m_lock;


	HQSharedPtr<HQPoolMemoryManager> m_subBufferListMemPool;
	SubBufferListType m_unusedBufferList;
	SubBufferListType m_usedBufferList;

	size_t m_state;
};