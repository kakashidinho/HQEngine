/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "HQXAudioSource.h"

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#include "../../HQEngine/winstore/HQWinStoreUtil.h"

#include <agile.h>

using namespace Windows::UI::Core;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel;
using namespace Windows::Foundation;

using namespace HQWinStoreFileSystem;

extern Platform::Agile<CoreWindow> hq_engine_coreWindow_internal;

#endif

/*---------------HQXAudioDevice - left handed coordinates using device-------------------------*/

HQXAudioDevice::HQXAudioDevice(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog)
	: HQBaseAudioDevice(logStream, "XAudio2 Device :", flushLog) , 
	m_xAudioObj(NULL),
	m_masterVoice(NULL)
{
#ifndef IMPLICIT_LINK
	if(!InitXAudioFunctions())
	{
		Log("Error : Couldn't find XAudio2 library!");
		throw std::bad_alloc();
	}
#endif
	CoInitializeEx(NULL, COINIT_MULTITHREADED);

	HRESULT hr;
	//create xaudio2 object
	UINT32 flags = 0;
#if !HQ_XAUDIO_2_8 && (defined _DEBUG || defined DEBUG)
	flags = XAUDIO2_DEBUG_ENGINE;
#endif
	if ( FAILED(hr = XAudio2Create( &m_xAudioObj, flags, XAUDIO2_DEFAULT_PROCESSOR ) ) )
	{
		//try with zero flags
		if ( flags != 0 && FAILED(hr = XAudio2Create( &m_xAudioObj, 0, XAUDIO2_DEFAULT_PROCESSOR ) ) )
		{
			Log("Error : XAudio2Create() failed! error code : %d", hr);
			throw std::bad_alloc();
		}
	}
	//create master voice
	if ( FAILED(hr = m_xAudioObj->CreateMasteringVoice( &m_masterVoice ) ) )
	{
		Log("Error : CreateMasteringVoice() failed! error code : %d", hr);
		m_xAudioObj->Release();
		throw std::bad_alloc();
	}
	
	m_masterVoice->GetVoiceDetails(&m_masterVoiceDetails);

#if !HQ_XAUDIO_2_8
	//device details
	m_xAudioObj->GetDeviceDetails(0 , &m_deviceDetails);
#endif

	//init x3daudio
	if (speedOfSound < 0.00001f)
		speedOfSound = 0.00001f;

#if !HQ_XAUDIO_2_8
	UINT channelMask = m_deviceDetails.OutputFormat.dwChannelMask;
#else
	DWORD channelMask;
	m_masterVoice->GetChannelMask(&channelMask);
#endif
	X3DAudioInitialize(channelMask, speedOfSound, m_audioHandle);
	
	//init listener
	m_listener.OrientFront.x = 0.f;
	m_listener.OrientFront.y = 0.f;
	m_listener.OrientFront.z = -1.f;

	m_listener.OrientTop.x = 0.f;
	m_listener.OrientTop.y = 1.f;
	m_listener.OrientTop.z = 0.f;

	m_listener.Position.x = m_listener.Position.y = m_listener.Position.z = 0.f;
	m_listener.Velocity.x = m_listener.Velocity.y = m_listener.Velocity.z = 0.f;

	m_listener.pCone = NULL;

	/*--------device name-------------*/
	char *rendererDesc = NULL;

#if !HQ_XAUDIO_2_8
	size_t len = wcstombs(NULL , m_deviceDetails.DisplayName , 0);
	if (len != -1)
	{
		try
		{
			rendererDesc = HQ_NEW char[len + 1];
			wcstombs(rendererDesc , m_deviceDetails.DisplayName , len + 1);
		}
		catch(std::bad_alloc e)
		{
			
		}
	}
#endif

	if (rendererDesc == NULL)
		Log("Init done!");
	else
	{
		Log("Init done! Device Description: %s" , rendererDesc);
		delete[] rendererDesc;
	}

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	//pause and resume audio engine when needed
	HQWinStoreUtil::RunOnUIThread(hq_engine_coreWindow_internal->Dispatcher, [this] ()
	{
		m_appSuspendToken = CoreApplication::Suspending +=
				ref new EventHandler<SuspendingEventArgs^>([this](Platform::Object^ sender, Windows::ApplicationModel::SuspendingEventArgs^ args)
					{
						Pause();
					}
				);

		m_appResumeToken = CoreApplication::Resuming +=
				ref new EventHandler<Platform::Object^>([this](Platform::Object^ sender, Platform::Object^ args)
					{
						Resume();
					}
				);
	});
#endif

}
HQXAudioDevice::~HQXAudioDevice() 
{

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	//unregister callbacks
	HQWinStoreUtil::RunOnUIThreadAndWait(hq_engine_coreWindow_internal->Dispatcher, [this] ()
	{
		CoreApplication::Suspending -= m_appSuspendToken;

		CoreApplication::Resuming -= m_appResumeToken;
	});
#endif

	DeleteAllAudioBuffers();
	DeleteAllSources();

	ULONG ref;
	if (m_masterVoice)
		m_masterVoice->DestroyVoice();
	
	if (m_xAudioObj)
		ref = m_xAudioObj->Release();

#ifndef IMPLICIT_LINK
	ReleaseXAudioFunctions();
#endif

	CoUninitialize();
}

void HQXAudioDevice::Pause()
{
	if (this->m_xAudioObj)
		m_xAudioObj->StopEngine();
}

void HQXAudioDevice::Resume()
{
	if (this->m_xAudioObj)
		m_xAudioObj->StartEngine();
}

UINT HQXAudioDevice::GetMasterVoiceInputChannels() const
{
#if !HQ_XAUDIO_2_8
	return m_deviceDetails.OutputFormat.Format.nChannels;
#else
	return m_masterVoiceDetails.InputChannels;
#endif
}

//use to re calculate source's audio setting after listener state is changed
void HQXAudioDevice::SetSourceAudioSettings()
{
	HQItemManager<HQBaseAudioSourceController>::Iterator ite;
	m_sourceManager.GetIterator(ite);

	while(!ite.IsAtEnd())
	{
		HQXAudioSourceController* pXSource = (HQXAudioSourceController*)ite.GetItemPointerNonCheck().GetRawPointer();
		pXSource->SetAudioSetting();
		++ite;
	}
}

HQReturnVal HQXAudioDevice::CreateAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID) 
{
	if (this->IsAudioFileLoaded(fileName, pBufferID))
	{
		Log("Buffer from file \"%s\" is already created!", fileName);
		return HQ_OK;
	}

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	auto f = HQWinStoreFileSystem::OpenFileForRead(fileName);
#else
	FILE *f = fopen(fileName, "rb");
#endif
	if (f == NULL)
	{
		Log("File \"%s\" doesn't exist!", fileName);
		return HQ_FAILED;
	}
	
	AudioDataInfo info;
	HQReturnVal re;
	
	if (HQBaseAudioDevice::IsWaveFile(f))//wave file
	{
		re = this->GetWaveData(fileName, f, info);
	}
	else
	{
		OggVorbis_File vf;
		if (HQBaseAudioDevice::IsVorbisFile(f, vf))//vorbis
		{
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

	
	HQXAudioBuffer *pBuffer = NULL;	
	try{
		pBuffer = HQ_NEW HQXAudioBuffer(fileName, info);
		if (m_bufferManager.AddItem(pBuffer, pBufferID) == false)
			throw std::bad_alloc();
	}
	catch(std::bad_alloc e)
	{
		re = HQ_FAILED_MEM_ALLOC;
		if (pBuffer != NULL)
			delete pBuffer;
	}

	HQBaseAudioDevice::ReleaseAudioDataInfo(info);
	
	

	fclose(f);

	return re;
}


HQReturnVal HQXAudioDevice::CreateStreamAudioBufferFromFile(const char *fileName, hquint32 numSubBuffers, hquint32 subBufferSize, hq_uint32 *pBufferID)
{
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	auto f = HQWinStoreFileSystem::OpenFileForRead(fileName);
#else
	FILE *f = fopen(fileName, "rb");
#endif
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

	
	HQXAudioStreamBuffer *pBuffer = NULL;	
	try{
		pBuffer = HQ_NEW HQXAudioStreamBuffer(fileName, info, streamBufferSupplier, numSubBuffers, subBufferSize);
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

HQXAudioSourceController *HQXAudioDevice::CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer)
{
	return new HQXAudioSourceController(m_xAudioObj, info, pBuffer);//this source uses left handed coordinates
}

///{bufferID} can be HQ_NOT_AVAIL_ID, in that case no audio buffer is attached to source
HQReturnVal HQXAudioDevice::CreateSource(const HQAudioSourceInfo &info, hq_uint32 bufferID, hq_uint32 *pSourceID)
{
	HQSharedPtr<HQBaseAudioBuffer> buffer = m_bufferManager.GetItemPointer(bufferID);
	HQXAudioSourceController *source = NULL;
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
HQReturnVal HQXAudioDevice::CreateSource(const HQAudioSourceInfo &info, hq_uint32 *pSourceID) 
{
	return CreateSource(info, HQ_NOT_AVAIL_ID, pSourceID);
}
HQReturnVal HQXAudioDevice::AttachAudioBuffer(hq_uint32 bufferID, hq_uint32 sourceID) 
{
	HQSharedPtr<HQBaseAudioBuffer> buffer = m_bufferManager.GetItemPointer(bufferID);
	HQSharedPtr<HQBaseAudioSourceController> source = m_sourceManager.GetItemPointer(sourceID);
	if (source == NULL)
		return HQ_FAILED_INVALID_ID;
	HQReturnVal re = ((HQXAudioSourceController*) source.GetRawPointer())->AttachBuffer(buffer);
#if defined _DEBUG || defined DEBUG
	if (re != HQ_OK)
		this->Log("Error : Buffer couldn't attached to source!");
#endif
	return re;
}

///default position is {0 , 0 , 0}
HQReturnVal HQXAudioDevice::SetListenerPosition(const hqfloat32 position[3])
{
	memcpy(&m_listener.Position, position, 3 *sizeof(hqfloat32));
	this->SetSourceAudioSettings();
	return HQ_OK;
}
///default velocity is {0 , 0 , 0}
HQReturnVal HQXAudioDevice::SetListenerVelocity(const hqfloat32 velocity[3])
{
	memcpy(&m_listener.Velocity, velocity, 3 *sizeof(hqfloat32));
	this->SetSourceAudioSettings();
	return HQ_OK;
}
///default orientation(direction & up) is {0 , 0 , -1 , 0 , 1 , 0}
HQReturnVal HQXAudioDevice::SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) 
{
	memcpy(&m_listener.OrientFront, at, 3 *sizeof(hqfloat32));
	memcpy(&m_listener.OrientTop, up, 3 *sizeof(hqfloat32));

	this->SetSourceAudioSettings();
	return HQ_OK;
}
HQReturnVal HQXAudioDevice::SetListenerVolume(hqfloat32 volume)
{
#if defined _DEBUG || defined DEBUH
	if (volume < 0.0f || volume > 1.f)
		return HQ_FAILED_INVALID_PARAMETER;
#endif
	
	m_masterVoice->SetVolume(volume);

	return HQ_OK;
}



/*----------HQXAudioDeviceRH - right handed coordinates using device----------*/

HQXAudioSourceController *HQXAudioDeviceRH::CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer)
{
	return new HQXAudioSourceControllerRH(m_xAudioObj, info, pBuffer);//this source uses right handed coordinates
}

///default position is {0 , 0 , 0}
HQReturnVal HQXAudioDeviceRH::SetListenerPosition(const hqfloat32 position[3])
{
	m_listener.Position.x = position[0];
	m_listener.Position.y = position[1];
	m_listener.Position.z = -position[2];
	this->SetSourceAudioSettings();
	return HQ_OK;
}
///default velocity is {0 , 0 , 0}
HQReturnVal HQXAudioDeviceRH::SetListenerVelocity(const hqfloat32 velocity[3])
{
	m_listener.Velocity.x = velocity[0];
	m_listener.Velocity.y = velocity[1];
	m_listener.Velocity.z = -velocity[2];
	this->SetSourceAudioSettings();
	return HQ_OK;
}
///default orientation(direction & up) is {0 , 0 , -1 , 0 , 1 , 0}
HQReturnVal HQXAudioDeviceRH::SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) 
{
	m_listener.OrientFront.x = at[0];
	m_listener.OrientFront.y = at[1];
	m_listener.OrientFront.z = -at[2];

	m_listener.OrientTop.x = up[0];
	m_listener.OrientTop.y = up[1];
	m_listener.OrientTop.z = -up[2];
	this->SetSourceAudioSettings();
	return HQ_OK;
}



/*-----------------------------------------*/
HQBaseAudioDevice * HQCreateXAudioDevice(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog, bool leftHandedCoordinate)
{
	if (ge_pAudioDevice == NULL)
	{
		try{
			if (leftHandedCoordinate)
				ge_pAudioDevice = HQ_NEW HQXAudioDevice(speedOfSound, logStream, flushLog);
			else
				ge_pAudioDevice = HQ_NEW HQXAudioDeviceRH(speedOfSound, logStream, flushLog);
		}
		catch (std::bad_alloc e)
		{
			return NULL;
		}
	}

	return ge_pAudioDevice;
}
