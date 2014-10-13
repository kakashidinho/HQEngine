/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_AUDIO_H
#define HQ_AUDIO_H
#include "HQReturnVal.h"
#include "HQPrimitiveDataType.h"
#include "HQLogStream.h"

#if defined HQ_STATIC_ENGINE
#	ifndef HQ_STATIC_AUDIO_ENGINE
#		define HQ_STATIC_AUDIO_ENGINE
#	endif
#endif

#ifdef HQ_STATIC_AUDIO_ENGINE
#	define HQAUDIO_API
#else
#	if defined WIN32 || defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
#		ifdef HQAUDIO_EXPORTS
#			define HQAUDIO_API __declspec(dllexport)
#		else
#			define HQAUDIO_API __declspec(dllimport)
#		endif
#	else
#		define HQAUDIO_API __attribute__ ((visibility("default")))
#	endif
#endif

#ifndef HQ_NOT_AVAIL_ID
#	define HQ_NOT_AVAIL_ID 0xcdcdcdcd
#endif

const hqfloat32 HQ_DEFAULT_SPEED_OF_SOUND = 343.3f;

enum HQAudioFormatType
{
	HQ_AUDIO_PCM,//linear pcm
	HQ_AUDIO_FMT_FORCE_DWORD = 0xffffffff
};

/*---------HQAudioSourceInfo------------*/
struct HQAudioSourceInfo
{
	HQAudioFormatType formatType;//format type
	hquint32 bits;//bits per channel
	hquint32 numChannels;//number of channels
};
inline HQAudioSourceInfo HQCreateAudioSrcInfo(HQAudioFormatType formatType,
											  hquint32 bits,
											  hquint32 numChannels)
{
	HQAudioSourceInfo info;
	info.formatType = formatType;
	info.bits = bits;
	info.numChannels = numChannels;

	return info;
}

inline HQAudioSourceInfo HQDefaultAudioSourceInfo()
{
	return HQCreateAudioSrcInfo(HQ_AUDIO_PCM, 16, 1);
}
/*--------HQAudioSourceController---------------*/
class HQAudioSourceController {
public:
	enum State{
		PLAYING,
		STOPPED,
		PAUSED,
		UNDEFINED = 0xffffffff
	};

	HQAudioSourceController() {}
	
	///3D positioning is enabled by default
	virtual HQReturnVal Enable3DPositioning(HQBool enable) = 0;

	/*----------set position methods return HQ_FAILED when 3d positioning mode is disabled, only in debug mode---*/
	///default position is {0 , 0 , 0}
	virtual HQReturnVal SetPosition(const hqfloat32 position[3]) = 0;
	///default velocity is {0 , 0 , 0}
	virtual HQReturnVal SetVelocity(const hqfloat32 velocity[3]) = 0;
	///default direction is {0 , 0 , 0} => omnidirectional
	virtual HQReturnVal SetDirection(const hqfloat32 direction[3]) = 0;
	///for directional source, {innerAngle} is in degree, default is 360.f
	virtual HQReturnVal SetInnerAngle(hqfloat32 innerAngle) = 0;
	///for directional source, {outerAngle} is in degree, default is 360.f
	virtual HQReturnVal SetOuterAngle(hqfloat32 outerAngle) = 0;

	virtual HQReturnVal SetMaxDistance(hqfloat32 maxDistance) = 0;

	///{volume}must be in [0..1]. default is 1.0
	virtual HQReturnVal SetVolume(hqfloat32 volume) = 0;

	virtual HQReturnVal Play(HQBool continuous = HQ_TRUE) = 0;///will resume playing if the source is paused before
	virtual HQReturnVal Pause() = 0;
	virtual HQReturnVal Stop() = 0;
	virtual State GetCurrentState() = 0;

	virtual HQReturnVal UpdateStream() = 0;///call this if a stream buffer is attached to this source
protected:
	virtual ~HQAudioSourceController() {}
};

/*------HQAudioDevice----------------*/
class HQAudioDevice {
public:
	HQAudioDevice() {}
	virtual HQReturnVal Release() = 0;
	virtual HQReturnVal CreateAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID) = 0;
	virtual HQReturnVal CreateStreamAudioBufferFromFile(const char *fileName, hquint32 numSubBuffers, hquint32 subBufferSize, hq_uint32 *pBufferID) = 0;
	virtual HQReturnVal CreateStreamAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID) = 0;///Create Stream Buffer with default num sub buffers and default sub buffer size (Implementation dependent)
	///get playback duration in seconds of audio buffer
	virtual hqfloat32 GetAudioBufferPlaybackTime(hquint32 bufferID) = 0;
	virtual HQReturnVal RemoveAudioBuffer(hquint32 bufferID) = 0;
	virtual void RemoveAllAudioBuffers() = 0;
	///
	///{info} specifies format, channels and samples rate ... of audio buffers that can be attached to source. 
	///{bufferID} can be HQ_NOT_AVAIL_ID, in that case no audio buffer is attached to source. 
	///Note: stream buffer cannot be attached to multiple sources at the same time
	///
	virtual HQReturnVal CreateSource(const HQAudioSourceInfo &info, hq_uint32 bufferID, hq_uint32 *pSourceID) = 0;
	///
	///create source without audio buffer
	///{info} specifies format, channels and samples rate ... of audio buffers that can be attached to source. 
	///
	virtual HQReturnVal CreateSource(const HQAudioSourceInfo &info, hq_uint32 *pSourceID) = 0;
	virtual HQReturnVal AttachAudioBuffer(hq_uint32 bufferID, hq_uint32 sourceID) = 0;///Note: stream buffer cannot be attached to multiple sources at the same time

	virtual HQAudioSourceController *GetSourceController(hquint32 sourceID) = 0;
	virtual HQReturnVal RemoveSource(hquint32 sourceID) = 0;
	virtual void RemoveAllSources() = 0;

	///default position is {0 , 0 , 0}
	virtual HQReturnVal SetListenerPosition(const hqfloat32 position[3]) = 0;
	///default velocity is {0 , 0 , 0}
	virtual HQReturnVal SetListenerVelocity(const hqfloat32 velocity[3]) = 0;
	///default direction is {0 , 0 , -1} &  up is (0 , 1 , 0 )
	virtual HQReturnVal SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) = 0;

	///{volume}must be in [0..1]. default is 1.0
	virtual HQReturnVal SetListenerVolume(hqfloat32 volume) = 0;

	virtual void Pause() = 0;
	virtual void Resume() = 0;

	/*----------capabilities-----------*/
	virtual bool IsMultiChannelsSupported(hquint32 numChannels) = 0;
	virtual bool IsAudioFormatTypeSupported(HQAudioFormatType formatType) = 0;
protected:
	virtual ~HQAudioDevice() {}
};

extern "C"
{
	///return already created device if there's a device created before this call
	HQAUDIO_API HQAudioDevice * HQCreateAudioDevice(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog, bool leftHandedCoordinate = true);
}
#endif
