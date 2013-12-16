// HQAudio.cpp : Defines the exported functions for the DLL application.
//
#include "../HQAudioPCH.h"
#include "HQAudioSourceAL.h"

#ifdef WIN32
#	ifdef IMPLICIT_LINK
#		pragma comment(lib, "OpenAL32.lib")
#	endif
#endif


#ifndef HQ_NO_OPEN_AL


/*---------------HQAudioDeviceAL - right handed coordinates using device-------------------------*/

HQAudioDeviceAL::HQAudioDeviceAL(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog)
	: HQBaseAudioDevice(logStream, "OpenAL Audio Device :", flushLog) , 
	m_device(NULL),
	m_context(NULL)
{
#ifndef IMPLICIT_LINK
	if(!InitALFunctionPointers())
	{
		Log("Error : Couldn't find OpenAL library!");
		throw std::bad_alloc();
	}
#endif

	m_device = alcOpenDevice(NULL);
	if (m_device == NULL)
	{
		Log("Error : alcOpenDevice(NULL) failed!");
		throw std::bad_alloc();
	}
	m_context = alcCreateContext(m_device, NULL);
	if (m_context == NULL)
	{
		Log("Error : alcCreateContext(m_device, NULL) failed!");
		alcCloseDevice(m_device);
		throw std::bad_alloc();
	}
	if (alcMakeContextCurrent(m_context) == AL_FALSE)
	{
		Log("Error : alcMakeContextCurrent(m_context) failed!");
		alcDestroyContext(m_context);
		alcCloseDevice(m_device);
		throw std::bad_alloc();
	}

	//set speed of sound
	if (speedOfSound < 0.00001f)
		speedOfSound = 0.00001f;
	alSpeedOfSound(speedOfSound);

	alGetError();//reset error state
	
	const ALchar *renderer = alcGetString(m_device, ALC_DEVICE_SPECIFIER); 
	Log("Init done! Device Description: %s" , renderer);
}
HQAudioDeviceAL::~HQAudioDeviceAL() 
{
	DeleteAllAudioBuffers();
	DeleteAllSources();

	alcMakeContextCurrent(NULL);
	if (m_context != NULL)
		alcDestroyContext(m_context);
	if (m_device != NULL)
		alcCloseDevice(m_device);

#ifndef IMPLICIT_LINK
	ReleaseALFunctionPointers();
#endif
}

HQReturnVal HQAudioDeviceAL::CreateAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID) 
{
	if (this->IsAudioFileLoaded(fileName, pBufferID))
	{
		Log("Buffer from file \"%s\" is already created!", fileName);
		return HQ_OK;
	}

	FILE *f = fopen(fileName, "rb");
	if (f == NULL)
	{
		Log("File \"%s\" doesn't exist!", fileName);
		return HQ_FAILED;
	}
	
	AudioDataInfo info;
	HQReturnVal re;

#ifdef RAW_VORBIS
	bool isRawVorbis = false;
#endif
	
	if (HQBaseAudioDevice::IsWaveFile(f))//wave file
	{
		re = this->GetWaveData(fileName, f, info);
	}
	else
	{
		OggVorbis_File vf;
		if (HQBaseAudioDevice::IsVorbisFile(f, vf))//vorbis
		{
#ifdef RAW_VORBIS
			if (alIsExtensionPresent("AL_EXT_vorbis"))
			{
				re = this->GetVorbisData(fileName, vf, info);
				isRawVorbis = true;
			}
			else//raw vorbis not supported, we need to decode it
#endif
				re = this->DecodeVorbis(fileName, vf, info);
		}
		else
		{
			Log("Error : File \"%s\" is not supported!", fileName);
			re = HQ_FAILED_FORMAT_NOT_SUPPORT;
		}
	}
	if (re != HQ_OK)//failed
		return re;
	
	ALenum format = 0;
#ifdef RAW_VORBIS
	if (isRawVorbis)
		format = alGetEnumValue("AL_FORMAT_VORBIS_EXT");
	else {
#endif
		switch(info.channels)
		{
			case 1:
				if (info.bits == 8)
					format = AL_FORMAT_MONO8;
				else
					format = AL_FORMAT_MONO16;
				break;
			case 2:
				if (info.bits == 8)
					format = AL_FORMAT_STEREO8;
				else
					format = AL_FORMAT_STEREO16;
				break;
		}
#ifdef RAW_VORBIS
	}
#endif

	
	if (format != 0)
	{
		//send data to openAL buffer
		HQAudioBufferAL *pBufferAL = NULL;	
		try{
			pBufferAL = new HQAudioBufferAL(fileName, format, info);
			if (m_bufferManager.AddItem(pBufferAL, pBufferID) == false)
				throw std::bad_alloc();
		}
		catch(std::bad_alloc e)
		{
			re = HQ_FAILED_MEM_ALLOC;
			if (pBufferAL != NULL)
				delete pBufferAL;
		}

	}
	HQBaseAudioDevice::ReleaseAudioDataInfo(info);
	
	

	fclose(f);

	return re;
}


HQReturnVal HQAudioDeviceAL::CreateStreamAudioBufferFromFile(const char *fileName, hquint32 numSubBuffers, hquint32 subBufferSize, hq_uint32 *pBufferID)
{
	FILE *f = fopen(fileName, "rb");

	if (f == NULL)
	{
		Log("File \"%s\" doesn't exist!", fileName);
		return HQ_FAILED;
	}
	
	AudioInfo info;
	HQReturnVal re;
	
	HQStreamBufferSupplier * streamBufferSupplier = NULL;

	if (HQBaseAudioDevice::IsWaveFile(f))//wave file
	{
		re = this->CreateWaveStreamBufferSupplier(fileName, f, info, streamBufferSupplier);
	}
	else
	{
		OggVorbis_File vf;
		if (HQBaseAudioDevice::IsVorbisFile(f, vf))//vorbis
		{
			re = this->CreateVorbisStreamBufferSupplier(fileName, vf, info, streamBufferSupplier);
		}
		else
		{
			Log("Error : File \"%s\" is not supported!", fileName);
			re = HQ_FAILED_FORMAT_NOT_SUPPORT;
		}
	}
	if (re != HQ_OK)//failed
		return re;

	ALenum format = 0;
	switch(info.channels)
	{
		case 1:
			if (info.bits == 8)
				format = AL_FORMAT_MONO8;
			else
				format = AL_FORMAT_MONO16;
			break;
		case 2:
			if (info.bits == 8)
				format = AL_FORMAT_STEREO8;
			else
				format = AL_FORMAT_STEREO16;
			break;
	}

	
	HQAudioStreamBufferAL *pBuffer = NULL;	
	try{
		pBuffer = HQ_NEW HQAudioStreamBufferAL(fileName, format, info, streamBufferSupplier, numSubBuffers, subBufferSize);
		if (m_bufferManager.AddItem(pBuffer, pBufferID) == false)
			throw std::bad_alloc();
	}
	catch(std::bad_alloc e)
	{
		re = HQ_FAILED_MEM_ALLOC;
		if (pBuffer != NULL)
			delete pBuffer;
	}


	return re;
}


HQAudioSourceControllerAL *HQAudioDeviceAL::CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer)
{
	return new HQAudioSourceControllerAL(info, pBuffer);//this source uses right handed coordinates
}

///{bufferID} can be NOT_AVAIL_ID, in that case no audio buffer is attached to source
HQReturnVal HQAudioDeviceAL::CreateSource(const HQAudioSourceInfo &info, hq_uint32 bufferID, hq_uint32 *pSourceID)
{
	HQSharedPtr<HQBaseAudioBuffer> buffer = m_bufferManager.GetItemPointer(bufferID);
	HQAudioSourceControllerAL *source = NULL;
	try
	{
		source = this->CreateNewSourceObject(info, buffer);
		if (!m_sourceManager.AddItem(source, pSourceID))
			throw std::bad_alloc();
	}
	catch (std::bad_alloc e)
	{
		if (source != NULL)
			delete source;
		return HQ_FAILED_MEM_ALLOC;
	}

	if (buffer != source->GetAttachedBuffer())
	{
		Log("Warning : Buffer couldn't attached to source!");
		return HQ_WARNING_AUDIO_SOURCE_BUFFER_MISMATCH;
	}

	return HQ_OK;
	
}
///create source without audio buffer
HQReturnVal HQAudioDeviceAL::CreateSource(const HQAudioSourceInfo &info, hq_uint32 *pSourceID) 
{
	return CreateSource(info, NOT_AVAIL_ID, pSourceID);
}
HQReturnVal HQAudioDeviceAL::AttachAudioBuffer(hq_uint32 bufferID, hq_uint32 sourceID) 
{
	HQSharedPtr<HQBaseAudioBuffer> buffer = m_bufferManager.GetItemPointer(bufferID);
	HQSharedPtr<HQBaseAudioSourceController> source = m_sourceManager.GetItemPointer(sourceID);
	if (source == NULL)
		return HQ_FAILED_INVALID_ID;
	HQReturnVal re = ((HQAudioSourceControllerAL*) source.GetRawPointer())->AttachBuffer(buffer);
#if defined _DEBUG || defined DEBUG
	if (re != HQ_OK)
		this->Log("Error : Buffer couldn't attached to source!");
#endif
	return re;
}

///default position is {0 , 0 , 0}
HQReturnVal HQAudioDeviceAL::SetListenerPosition(const hqfloat32 position[3])
{
	alListenerfv(AL_POSITION, position);
	return HQ_OK;
}
///default velocity is {0 , 0 , 0}
HQReturnVal HQAudioDeviceAL::SetListenerVelocity(const hqfloat32 velocity[3])
{
	alListenerfv(AL_VELOCITY, velocity);
	return HQ_OK;
}
///default direction is {0 , 0 , -1, 0} &  up is (0 , 1 , 0 , 0)
HQReturnVal HQAudioDeviceAL::SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) 
{
	hqfloat32 orientationArray[] = 
	{
		at[0], at[1], at[2],
		up[0], up[1], up[2]
	};
	alListenerfv(AL_ORIENTATION, orientationArray);
	return HQ_OK;
}
HQReturnVal HQAudioDeviceAL::SetListenerVolume(hqfloat32 volume)
{
#if defined _DEBUG || defined DEBUH
	if (volume < 0.0f || volume > 1.f)
		return HQ_FAILED_INVALID_PARAMETER;
#endif
	alListenerf(AL_GAIN, volume);
	return HQ_OK;
}



/*----------HQAudioDeviceLHAL - left handed coordinates using device----------*/

HQAudioSourceControllerAL *HQAudioDeviceLHAL::CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer)
{
	return new HQAudioSourceControllerLHAL(info, pBuffer);//this source uses left handed coordinates
}

///default position is {0 , 0 , 0}
HQReturnVal HQAudioDeviceLHAL::SetListenerPosition(const hqfloat32 position[3])
{
	alListener3f(AL_POSITION, position[0], position[1], -position[2]);
	return HQ_OK;
}
///default velocity is {0 , 0 , 0}
HQReturnVal HQAudioDeviceLHAL::SetListenerVelocity(const hqfloat32 velocity[3])
{
	alListener3f(AL_VELOCITY, velocity[0], velocity[1], -velocity[2]);
	return HQ_OK;
}
///default direction is {0 , 0 , -1, 0} &  up is (0 , 1 , 0 , 0)
HQReturnVal HQAudioDeviceLHAL::SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) 
{
	hqfloat32 orientationArray[] = 
	{
		at[0], at[1], -at[2],
		up[0], up[1], -up[2]
	};
	alListenerfv(AL_ORIENTATION, orientationArray);
	return HQ_OK;
}



/*-----------------------------------------*/
HQBaseAudioDevice * HQCreateAudioDeviceAL(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog, bool leftHandedCoordinate)
{
	if (ge_pAudioDevice == NULL)
	{
		try{
			if (leftHandedCoordinate)
				ge_pAudioDevice = new HQAudioDeviceLHAL(speedOfSound, logStream, flushLog);
			else
				ge_pAudioDevice = new HQAudioDeviceAL(speedOfSound, logStream, flushLog);
		}
		catch (std::bad_alloc e)
		{
			return NULL;
		}
	}

	return ge_pAudioDevice;
}

#endif//#ifndef HQ_NO_OPEN_AL
