/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_X_AUDIO_H
#define HQ_X_AUDIO_H

#include "../HQAudioBase.h"

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	define IMPLICIT_LINK
#	define HQ_XAUDIO_2_8 1//xaudio 2.8
#else
#	define HQ_XAUDIO_2_8 0
#endif

#ifdef IMPLICIT_LINK
#	include <XAudio2.h>
#	include <X3DAudio.h>
#	if !HQ_XAUDIO_2_8
#		pragma comment (lib, "X3DAudio.lib")
#	else
#		pragma comment (lib, "XAudio2.lib")
#	endif
#else
#	include "xFunctionPointers.h"
#endif

class HQXAudioSourceController;

//this class uses left handed coordinates system
class HQXAudioDevice : public HQBaseAudioDevice
{
public:
	HQXAudioDevice(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog);
	~HQXAudioDevice() ;

	virtual HQReturnVal CreateAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID) ;
	virtual HQReturnVal CreateStreamAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID)
	{
		return CreateStreamAudioBufferFromFile(fileName, 3, 65536, pBufferID);
	}
	virtual HQReturnVal CreateStreamAudioBufferFromFile(const char *fileName, hquint32 numSubBuffers, hquint32 subBufferSize, hq_uint32 *pBufferID);
	///{bufferID} can be NOT_AVAIL_ID, in that case no audio buffer is attached to source
	virtual HQReturnVal CreateSource(const HQAudioSourceInfo &info, hq_uint32 bufferID, hq_uint32 *pSourceID);
	///create source without audio buffer
	virtual HQReturnVal CreateSource(const HQAudioSourceInfo &info, hq_uint32 *pSourceID) ;
	virtual HQReturnVal AttachAudioBuffer(hq_uint32 bufferID, hq_uint32 sourceID) ;
	
	///default position is {0 , 0 , 0}
	virtual HQReturnVal SetListenerPosition(const hqfloat32 position[3]) ;
	///default velocity is {0 , 0 , 0}
	virtual HQReturnVal SetListenerVelocity(const hqfloat32 velocity[3]) ;
	///default direction is {0 , 0 , -1} &  up is (0 , 1 , 0 )
	virtual HQReturnVal SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) ;

	virtual HQReturnVal SetListenerVolume(hqfloat32 volume);

	const X3DAUDIO_LISTENER &GetListener() {return m_listener;}
	const X3DAUDIO_HANDLE& Get3DAudioHandle() {return m_audioHandle;}
#if !HQ_XAUDIO_2_8
	const XAUDIO2_DEVICE_DETAILS &GetDeviceDetails() {return m_deviceDetails;}
#endif
	IXAudio2Voice * GetMasterVoice() {return m_masterVoice;}
	UINT GetMasterVoiceInputChannels() const;
protected:
	virtual HQXAudioSourceController *CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer);//create left handed source

	void SetSourceAudioSettings();//use to re calculate source's audio setting after listener state is changed
	void Pause();
	void Resume();

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	Windows::Foundation::EventRegistrationToken m_appSuspendToken;
	Windows::Foundation::EventRegistrationToken m_appResumeToken;
#endif

	IXAudio2 * m_xAudioObj;
	IXAudio2MasteringVoice * m_masterVoice;
	X3DAUDIO_LISTENER m_listener;
	X3DAUDIO_HANDLE m_audioHandle;
#if !HQ_XAUDIO_2_8
	XAUDIO2_DEVICE_DETAILS m_deviceDetails;
#endif
	XAUDIO2_VOICE_DETAILS m_masterVoiceDetails;
};

//this class uses right handed coordinates system
class HQXAudioDeviceRH : public HQXAudioDevice
{
public:
	HQXAudioDeviceRH(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog):
	  HQXAudioDevice(speedOfSound, logStream, flushLog)
	{
		hqfloat32 orientation[] = {0,0,-1, 0 , 1, 0};
		this->SetListenerOrientation(orientation, &orientation[3]);
	}
	~HQXAudioDeviceRH() {}

	///default position is {0 , 0 , 0}
	virtual HQReturnVal SetListenerPosition(const hqfloat32 position[3]) ;
	///default velocity is {0 , 0 , 0}
	virtual HQReturnVal SetListenerVelocity(const hqfloat32 velocity[3]) ;
	///default direction is {0 , 0 , -1} &  up is (0 , 1 , 0 )
	virtual HQReturnVal SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) ;

protected:
	virtual HQXAudioSourceController *CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer);//create right handed source
};

#endif
