#ifndef HQ_AUDIO_BUFFER_AL_H
#define HQ_AUDIO_BUFFER_AL_H
#include "HQAudioAL.h"
#include "../../HQThread.h"
#include "../../HQConditionVariable.h"
#include "../../HQLinkedList.h"
#include "../../HQAtomic.h"

/*-------------HQAudioBufferAL-------------------------*/
class HQAudioBufferAL : public HQBaseAudioBuffer
{
public:
	HQAudioBufferAL(const char *fileName, ALenum alFormat, const HQBaseAudioDevice:: AudioDataInfo &info)
	{
		m_bufferName = 0;
		alGenBuffers(1, &m_bufferName);
		if (AL_OUT_OF_MEMORY == alGetError())
			throw std::bad_alloc();
		//copy file name
		size_t len = strlen(fileName);
		m_audioFileName = new char [len + 1];
		strcpy(m_audioFileName, fileName);

		//specify buffer format
		this->SetFormatType(alFormat);
		
		//copy buffer data
		alBufferData(m_bufferName , alFormat , info.data,(ALsizei) info.size, info.sampleRate);
		if (AL_OUT_OF_MEMORY == alGetError())
				throw std::bad_alloc();

		//playback duration
		m_playTime = (hqfloat32) ((double)info.samples / info.sampleRate);
	}

	~HQAudioBufferAL()
	{
		if (m_bufferName != 0)
			alDeleteBuffers(1, &m_bufferName);
	}

	ALuint GetBufferName() {return m_bufferName;}
	
	HQAudioFormatType GetFormatType() {return m_formatType;}
	
private:
	void SetFormatType(ALenum alFormat)
	{
		switch(alFormat)
		{
		case AL_FORMAT_MONO8:case AL_FORMAT_MONO16:
		case AL_FORMAT_STEREO8:case AL_FORMAT_STEREO16:
			m_formatType = HQ_AUDIO_PCM;
			break;
		default:
			m_formatType = HQ_AUDIO_FMT_FORCE_DWORD;
		}
	}
	HQAudioFormatType m_formatType;
	ALuint m_bufferName;
};

/*-------------HQAudioStreamBufferAL-------------------------*/

class HQAudioStreamBufferAL : public HQBaseAudioBuffer, public HQThread
{
public:

	HQAudioStreamBufferAL(
		const char *fileName, 
		ALenum alFormat, 
		HQBaseAudioDevice::AudioInfo &info,
		HQStreamBufferSupplier *supplier,//this object will take ownership of supplier
		size_t numSubBuffers,
		size_t subBufferSize);

	~HQAudioStreamBufferAL();

	size_t GetNumSubBuffers() const {return m_numSubBuffers;}
	size_t GetSubBufferSize() const {return m_subBufferSize;}
	ALuint GetSubBuffer(size_t index) {return m_subBuffers[index];}
	const HQBaseAudioDevice::AudioInfo& GetInfo() {return m_info;}

	HQAudioFormatType GetFormatType() {return m_formatType;}

	ALuint GetAttachedSource() {return m_attachedSource;}
	void AttachSource(ALuint  source);

	void Stop();

	void RestartStream(bool repushUsedList);

	void PrepareToPlay();

	void EnableLoop(bool loop);

	void UpdateStream();
private:
	struct SubBufferData
	{
		hqubyte8 * data;//client data buffer
		size_t	   dataSize;//valid data size
	};

	typedef HQLinkedList<SubBufferData*, HQPoolMemoryManager> SubBufferClientDataListType;
	typedef HQLinkedList<ALuint, HQPoolMemoryManager> SubBufferALListType;

	virtual void Run();//worker thread implementation
	void FillSubBuffers();
	void FillSubBuffer(SubBufferData * bufferClientData);
	void RestartStreamNonLock(bool repushUsedList = true);
	void SetHQFormatType(ALenum alFormat)
	{
		switch(alFormat)
		{
		case AL_FORMAT_MONO8:case AL_FORMAT_MONO16:
		case AL_FORMAT_STEREO8:case AL_FORMAT_STEREO16:
			m_formatType = HQ_AUDIO_PCM;
			break;
		default:
			m_formatType = HQ_AUDIO_FMT_FORCE_DWORD;
		}
	}

	ALuint *m_subBuffers;
	ALuint  m_attachedSource;//attached source
	size_t m_numSubBuffers;
	size_t m_subBufferSize;
	size_t m_streamPosition;
	HQStreamBufferSupplier *m_supplier;
	HQBaseAudioDevice::AudioInfo m_info;

	ALuint *   m_removedSubBuffersCache;
	SubBufferData * m_subBufferClientData;//client data for each sub buffer

	HQAudioFormatType m_formatType;
	ALenum			  m_formatAL;

	HQSimpleConditionVar m_lock;


	HQSharedPtr<HQPoolMemoryManager> m_subBufferListMemPool;

	SubBufferClientDataListType m_unusedBufferClienDataList;
	SubBufferClientDataListType m_usedBufferClienDataList;

	SubBufferALListType m_removedSubBuffers;//list of removed buffers from source

	HQAtomic<hquint32> m_state;
};

#endif