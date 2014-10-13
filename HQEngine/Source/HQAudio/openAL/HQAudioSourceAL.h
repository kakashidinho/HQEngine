/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_AUDIO_SOURCE_AL_H
#define HQ_AUDIO_SOURCE_AL_H

#include "HQAudioBufferAL.h"



/*-----------HQAudioSourceControllerAL - right handed coordinates using source--------------------*/
class HQAudioSourceControllerAL: public  HQBaseAudioSourceController{
public:

	HQAudioSourceControllerAL(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer) 
		:m_positioning(HQ_TRUE), m_info(info), m_continuous(HQ_FALSE)
	{
		alGenSources(1, &m_sourceName);
		if (AL_OUT_OF_MEMORY == alGetError())
			throw std::bad_alloc();

		AttachBuffer(pBuffer);
	}
	~HQAudioSourceControllerAL() 
	{
		alDeleteSources(1, &m_sourceName);
	}

	HQReturnVal AttachBuffer(HQSharedPtr<HQBaseAudioBuffer> &pBuffer)
	{
		if (m_pBuffer != pBuffer)
		{
			alSourcei(m_sourceName, AL_BUFFER, AL_NONE);//detach old buffer
			
			if (m_pBuffer != NULL && m_pBuffer->m_type == HQBaseAudioBuffer::STREAM_BUFFER)
			{
				//detach source from old stream buffer
				HQAudioStreamBufferAL *oldBuffer = (HQAudioStreamBufferAL*)m_pBuffer.GetRawPointer();
				oldBuffer->AttachSource(0);
			}

			HQBaseAudioBuffer* pRawBuffer = pBuffer.GetRawPointer();
			if (pRawBuffer != NULL)
			{
				switch (pRawBuffer->m_type)
				{
				case HQBaseAudioBuffer::NORMAL_BUFFER:
					{
						HQAudioBufferAL *bufferAL = (HQAudioBufferAL*)pRawBuffer;
						//get info of buffer
						if (bufferAL->GetFormatType() != m_info.formatType)//data format type
						{
							ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
							return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
						}
						ALint info;
						alGetBufferi(bufferAL->GetBufferName(), AL_BITS, &info);//bits per channel
						if (info != m_info.bits)
						{
							ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
							return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
						}

						alGetBufferi(bufferAL->GetBufferName(), AL_CHANNELS, &info);//num channels
						if (info != m_info.numChannels)
						{
							ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
							return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
						}


						alSourcei(m_sourceName, AL_BUFFER, bufferAL->GetBufferName());
					}
					break;
				case HQBaseAudioBuffer::STREAM_BUFFER:
					{
						HQAudioStreamBufferAL *bufferAL = (HQAudioStreamBufferAL*)pRawBuffer;

						if (bufferAL->GetAttachedSource() != 0)
						{
							ge_pAudioDevice->Log ("Error : Streaming buffer cannot attach to multiple sources!");
							return HQ_FAILED;
						}

						//get info of buffer
						if (bufferAL->GetFormatType() != m_info.formatType)//data format type
						{
							ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
							return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
						}
						if (bufferAL->GetInfo().bits != m_info.bits || bufferAL->GetInfo().channels != m_info.numChannels)
						{
							ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
							return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
						}

						bufferAL->AttachSource(m_sourceName);
					}
					break;
				}
			}
			m_pBuffer = pBuffer;
		}

		return HQ_OK;
	}

	HQReturnVal Enable3DPositioning(HQBool enable)
	{
		if(m_positioning == enable)//no changes
			return HQ_OK;
		if (enable)
		{
			alSourcei(m_sourceName, AL_SOURCE_RELATIVE, AL_FALSE);
			alSourcefv(m_sourceName, AL_POSITION, m_position);
		}
		else//disable 3d positioning
		{
			alGetSourcefv(m_sourceName, AL_POSITION, m_position);//cache position in 3d space
			alSourcei(m_sourceName, AL_SOURCE_RELATIVE, AL_TRUE);
			alSource3f(m_sourceName, AL_POSITION, 0.f , 0.f , 0.f);
		}

		m_positioning = enable;

		return HQ_OK;
	}

	///default position is {0 , 0 , 0}
	HQReturnVal SetPosition(const hqfloat32 position[3]) 
	{
#if defined _DEBUG || defined DEBUG
		if (m_positioning == HQ_FALSE)
			return HQ_FAILED;
#endif
		alSourcefv(m_sourceName, AL_POSITION, position);
		return HQ_OK;
	}
	///default velocity is {0 , 0 , 0}
	HQReturnVal SetVelocity(const hqfloat32 velocity[3]) 
	{
		alSourcefv(m_sourceName, AL_VELOCITY, velocity);
		return HQ_OK;
	}
	///default direction is {0 , 0 , 0} => omnidirectional
	HQReturnVal SetDirection(const hqfloat32 direction[3]) 
	{
		alSourcefv(m_sourceName, AL_DIRECTION, direction);
		return HQ_OK;
	}

	HQReturnVal SetInnerAngle(hqfloat32 innerAngle) 
	{
#if defined _DEBUG || defined DEBUH
		if (innerAngle < 0.0f || innerAngle > 360.f)
			return HQ_FAILED_INVALID_PARAMETER;
#endif
		alSourcef(m_sourceName, AL_CONE_INNER_ANGLE , innerAngle);
		return HQ_OK;
	}

	HQReturnVal SetOuterAngle(hqfloat32 outerAngle)
	{
#if defined _DEBUG || defined DEBUH
		if (outerAngle < 0.0f || outerAngle > 360.f)
			return HQ_FAILED_INVALID_PARAMETER;
#endif
		alSourcef(m_sourceName, AL_CONE_OUTER_ANGLE , outerAngle);
		return HQ_OK;
	}


	HQReturnVal SetMaxDistance(hqfloat32 maxDistance) 
	{
#if defined _DEBUG || defined DEBUH
		if (maxDistance < 0.0f)
			return HQ_FAILED_INVALID_PARAMETER;
#endif
		alSourcef(m_sourceName, AL_MAX_DISTANCE, maxDistance);

		return HQ_OK;
	}
	
	HQReturnVal SetVolume(hqfloat32 volume)
	{
#if defined _DEBUG || defined DEBUH
		if (volume < 0.0f || volume > 1.f)
			return HQ_FAILED_INVALID_PARAMETER;
#endif
		alSourcef(m_sourceName, AL_GAIN, volume);
		return HQ_OK;
	}

	HQReturnVal Play(HQBool continuous) 
	{
		ALint state ;
		alGetSourcei(m_sourceName, AL_SOURCE_STATE, &state);
		if (state == AL_PLAYING)//source is playing
			return HQ_FAILED_PLAYING;
		
		HQBaseAudioBuffer *pRawBuffer = m_pBuffer.GetRawPointer();
		if (pRawBuffer != NULL && pRawBuffer->m_type == HQBaseAudioBuffer::STREAM_BUFFER)
		{
			//stream mode need special care
			HQAudioStreamBufferAL * bufferAL = (HQAudioStreamBufferAL*)pRawBuffer;
			
			bufferAL->PrepareToPlay();
			bufferAL->EnableLoop(continuous == HQ_TRUE);
		}
		else
			alSourcei(m_sourceName, AL_LOOPING, continuous);//looping mode or not

		alSourcePlay(m_sourceName);

		m_continuous = continuous;

		return HQ_OK;
	}
	HQReturnVal Pause()
	{
		ALint state;
		alGetSourcei(m_sourceName, AL_SOURCE_STATE, &state);
		if (state == AL_PLAYING)//source is playing
			alSourcePause(m_sourceName);
		return HQ_OK;
	}

	HQReturnVal Resume()
	{
		ALint state;
		alGetSourcei(m_sourceName, AL_SOURCE_STATE, &state);
		if (state == AL_PAUSED)//source is paused
		{
			this->Play(m_continuous);
		}
		return HQ_OK;
	}
	HQReturnVal Stop()
	{
		alSourceStop(m_sourceName);

		HQBaseAudioBuffer *pRawBuffer = m_pBuffer.GetRawPointer();
		if (pRawBuffer != NULL && pRawBuffer->m_type == HQBaseAudioBuffer::STREAM_BUFFER)
		{
			//detach all buffers
			alSourcei(m_sourceName, AL_BUFFER, AL_NONE);//detach old buffers

			HQAudioStreamBufferAL * bufferAL = (HQAudioStreamBufferAL*)pRawBuffer;
			bufferAL->Stop();
		}

		return HQ_OK;
	}
	HQAudioSourceController::State GetCurrentState()
	{
		ALint state ;
		alGetSourcei(m_sourceName, AL_SOURCE_STATE, &state);
		switch(state)
		{
		case AL_INITIAL:
			return HQAudioSourceController::STOPPED;
		case AL_PLAYING:
			return HQAudioSourceController::PLAYING;
		case AL_PAUSED:
			return HQAudioSourceController::PAUSED;
		case AL_STOPPED:
			return HQAudioSourceController::STOPPED;
		default:
			return HQAudioSourceController::UNDEFINED;
		}
	}

	HQReturnVal UpdateStream()
	{
		HQBaseAudioBuffer *pRawBuffer = m_pBuffer.GetRawPointer();

		if (pRawBuffer != NULL)
		{
			switch (pRawBuffer->m_type)
			{
			case  HQBaseAudioBuffer::NORMAL_BUFFER:
				{
					ge_pAudioDevice->Log("Error : UpdateStream() is called when the source is not attached to any stream buffer!");
					return HQ_FAILED;
				}
				break;
			case  HQBaseAudioBuffer::STREAM_BUFFER:
				{
					HQAudioStreamBufferAL* bufferAL = (HQAudioStreamBufferAL*)pRawBuffer;
					bufferAL->UpdateStream();
				}
				break;
			}
		}
		return HQ_OK;
	}
	
	const HQSharedPtr<HQBaseAudioBuffer> &GetAttachedBuffer() {return m_pBuffer;}
protected:
	ALuint m_sourceName;
	HQBool m_positioning;
	HQBool m_continuous;
private:
	hqfloat32 m_position[3];//cached position
	HQSharedPtr<HQBaseAudioBuffer> m_pBuffer;
	const HQAudioSourceInfo m_info;
};


/*-----------HQAudioSourceControllerLHAL - left handed coordinates using source--------------------*/
class HQAudioSourceControllerLHAL: public  HQAudioSourceControllerAL{
public:
	HQAudioSourceControllerLHAL(const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer) 
		:HQAudioSourceControllerAL(info, pBuffer)
	{
	}

	///default position is {0 , 0 , 0}
	HQReturnVal SetPosition(const hqfloat32 position[3]) 
	{
#if defined _DEBUG || defined DEBUG
		if (m_positioning == HQ_FALSE)
			return HQ_FAILED;
#endif
		alSource3f(m_sourceName, AL_POSITION, position[0], position[1], - position[2]);
		return HQ_OK;
	}
	///default velocity is {0 , 0 , 0}
	HQReturnVal SetVelocity(const hqfloat32 velocity[3]) 
	{
		alSource3f(m_sourceName, AL_VELOCITY, velocity[0], velocity[1], -velocity[2]);
		return HQ_OK;
	}
	///default direction is {0 , 0 , 0} => omnidirectional
	HQReturnVal SetDirection(const hqfloat32 direction[3]) 
	{
		alSource3f(m_sourceName, AL_DIRECTION, direction[0], direction[1], -direction[2]);
		return HQ_OK;
	}
};

#endif
