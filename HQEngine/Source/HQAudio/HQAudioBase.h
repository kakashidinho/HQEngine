/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_AUDIO_BASE_H
#define HQ_AUDIO_BASE_H

#include "../HQPlatformDef.h"
#include "../HQAudio.h"
#include "../HQItemManager.h"
#include "../HQLoggableObject.h"
#include "../HQDataStream.h"
#include <string.h>
#include <vorbis/vorbisfile.h>

#ifdef APPLE
#	define RAW_VORBIS
#endif

#define HQ_AUDIO_FILE_STREAM HQDataReaderStream


/*---------HQBaseAudioSourceController---------------*/
class HQBaseAudioSourceController : public HQAudioSourceController
{
public:
	virtual ~HQBaseAudioSourceController() {}
};

/*----------HQBaseAudioBuffer-------------------*/
class HQBaseAudioBuffer
{
public:
	enum BufferType {
		NORMAL_BUFFER,
		STREAM_BUFFER,
	};

	HQBaseAudioBuffer(BufferType type = NORMAL_BUFFER) :m_audioFileName(NULL), m_type(type) {}
	virtual ~HQBaseAudioBuffer() {
		if (m_audioFileName != NULL)
			delete[] m_audioFileName;
	}

	char * m_audioFileName;

	hqfloat32 m_playTime;//playback durarion;

	BufferType m_type;
};

/*--HQStreamBufferSupplier-----------*/

class HQStreamBufferSupplier
{
public:
	virtual ~HQStreamBufferSupplier() {}

	virtual size_t GetNextData(void * buffer, size_t bufferSize) = 0;
	virtual HQReturnVal Rewind() = 0;
};

/*-----------HQBaseAudioDevice----------------*/
class HQBaseAudioDevice : public HQAudioDevice,  public HQLoggableObject {
public:
	HQBaseAudioDevice(HQLogStream *logStream, const char *logPrefix, bool flushLog);
	virtual HQReturnVal Release() ;
	
	virtual hqfloat32 GetAudioBufferPlaybackTime(hquint32 bufferID)
	{
		HQSharedPtr<HQBaseAudioBuffer> buffer = m_bufferManager.GetItemPointer(bufferID);
		if (buffer == NULL)
			return 0.f;

		return buffer->m_playTime;
	}
	virtual HQReturnVal DeleteAudioBuffer(hquint32 bufferID) 
	{
		return (HQReturnVal) m_bufferManager.Remove(bufferID);
	}
	virtual void DeleteAllAudioBuffers() 
	{
		m_bufferManager.RemoveAll();
	}


	virtual HQAudioSourceController *GetSourceController(hquint32 sourceID)
	{
		return m_sourceManager.GetItemRawPointer(sourceID);
	}
	
	virtual HQReturnVal DeleteSource(hquint32 sourceID) 
	{
		return (HQReturnVal) m_sourceManager.Remove(sourceID);
	}
	virtual void DeleteAllSources()
	{
		m_sourceManager.RemoveAll();
	}

	///sub class can overwrite this method
	virtual bool IsMultiChannelsSupported(hquint32 channels) {
		if (channels == 1 || channels == 2)
			return true;
		return false;
	}

	virtual bool IsAudioFormatTypeSupported(HQAudioFormatType formatType)
	{
		switch(formatType)
		{
		case HQ_AUDIO_PCM:
			return true;
		}
		return false;
	}
	
	struct AudioInfo
	{
		size_t size;
		hqint32 channels;
		hqint32 bits;//bits per channel
		hquint32 samples;//total samples
		size_t sampleRate;
	};

	struct AudioDataInfo: public AudioInfo
	{
		hqbyte * data;
	};
	
protected:
	~HQBaseAudioDevice() {
		Log("Released!");
	}
	
	
	static bool IsWaveFile(HQ_AUDIO_FILE_STREAM *f);
	static bool IsVorbisFile(HQ_AUDIO_FILE_STREAM *f, OggVorbis_File &vf);
	static void ReleaseAudioDataInfo(AudioDataInfo &info);


	bool IsAudioFileLoaded(const char *fileName, hquint32 *existID);
	///{fileName} is used for logging
	HQReturnVal GetWaveInfo(const char *fileName, HQ_AUDIO_FILE_STREAM *f, AudioInfo &info);
	
	///{fileName} is used for logging
	HQReturnVal GetVorbisInfo(const char *fileName, OggVorbis_File &vf, AudioInfo &info);

	///Get the whole wave  file data. {fileName} is used for logging
	HQReturnVal GetWaveData(const char *fileName, HQ_AUDIO_FILE_STREAM *f, AudioDataInfo &info);
	///Get the whole vorbis file data. {fileName} is used for logging
	HQReturnVal DecodeVorbis(const char *fileName, OggVorbis_File &vf, AudioDataInfo &info);

#ifdef RAW_VORBIS
	///Get the whole Compressed vorbis file data without decode it. {fileName} is used for logging
	HQReturnVal GetVorbisData(const char *fileName, OggVorbis_File &vf, AudioDataInfo &info);
#endif

	///{fileName} is used for logging. {supplierOut} will take over the ownership of {f}
	HQReturnVal CreateWaveStreamBufferSupplier(const char *fileName, HQ_AUDIO_FILE_STREAM *f, AudioInfo &info, HQStreamBufferSupplier*& supplierOut);

	///{fileName} is used for logging. {supplierOut} will take over the ownership of {vf}
	HQReturnVal CreateVorbisStreamBufferSupplier(const char *fileName, OggVorbis_File &vf, AudioInfo &info, HQStreamBufferSupplier*& supplierOut);



	HQItemManager<HQBaseAudioSourceController> m_sourceManager;
	HQItemManager<HQBaseAudioBuffer> m_bufferManager;
};

extern HQBaseAudioDevice * ge_pAudioDevice;

#endif
