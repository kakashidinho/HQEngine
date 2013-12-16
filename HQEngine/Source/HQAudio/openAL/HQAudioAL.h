#ifndef HQ_AUDIO_AL_H
#define HQ_AUDIO_AL_H

#include "../HQAudioBase.h"

#ifndef WIN32
#	ifndef IMPLICIT_LINK
#		define IMPLICIT_LINK
#	endif
#endif

#ifdef IMPLICIT_LINK
#	ifdef __APPLE__
#		include <OpenAL/al.h>
#		include <OpenAL/alc.h>
#	else
#		include <al.h>
#		include <alc.h>
#	endif
#else
#	include "alFunctionPointers.h"
#endif

class HQAudioSourceControllerAL;

//this class uses right handed coordinates system
class HQAudioDeviceAL : public HQBaseAudioDevice
{
public:
	HQAudioDeviceAL(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog);
	~HQAudioDeviceAL() ;

	virtual HQReturnVal CreateAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID) ;
	virtual HQReturnVal CreateStreamAudioBufferFromFile(const char *fileName, hq_uint32 *pBufferID)
	{
		return CreateStreamAudioBufferFromFile(fileName, 5, 65536, pBufferID);
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
protected:
	virtual HQAudioSourceControllerAL *CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer);//create right handed source
private:
	ALCdevice * m_device;
	ALCcontext * m_context;
};

//this class uses left handed coordinates system
class HQAudioDeviceLHAL : public HQAudioDeviceAL
{
public:
	HQAudioDeviceLHAL(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog):
	  HQAudioDeviceAL(speedOfSound, logStream, flushLog)
	{
		hqfloat32 orientation[] = {0,0,-1, 0 , 1, 0};
		this->SetListenerOrientation(orientation, &orientation[3]);
	}
	~HQAudioDeviceLHAL() {}

	///default position is {0 , 0 , 0}
	virtual HQReturnVal SetListenerPosition(const hqfloat32 position[3]) ;
	///default velocity is {0 , 0 , 0}
	virtual HQReturnVal SetListenerVelocity(const hqfloat32 velocity[3]) ;
	///default direction is {0 , 0 , -1} &  up is (0 , 1 , 0 )
	virtual HQReturnVal SetListenerOrientation(const hqfloat32 at[3], const hqfloat32 up[3]) ;

protected:
	virtual HQAudioSourceControllerAL *CreateNewSourceObject(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer);//create left handed source
};

#endif