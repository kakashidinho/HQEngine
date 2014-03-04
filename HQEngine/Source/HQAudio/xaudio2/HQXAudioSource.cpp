/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "HQXAudioSource.h"

HQXAudioSourceController::HQXAudioSourceController(IXAudio2 *pXAudio2, const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer) 
	:m_positioning(HQ_TRUE) ,  m_info(info), m_state(STOPPED),
	m_3DAudioCalflags(X3DAUDIO_CALCULATE_MATRIX | X3DAUDIO_CALCULATE_DOPPLER)
{
	UINT numOutputChannels =  ((HQXAudioDevice*) ge_pAudioDevice)->GetMasterVoiceInputChannels();
	m_pDefaultMatrixCoefficients = HQ_NEW FLOAT32[numOutputChannels * m_info.numChannels];

	//create source voice
	WAVEFORMATEX * waveFmt = CreateWaveFormat(info);
	if (waveFmt == NULL|| pXAudio2->CreateSourceVoice(&m_xSource, waveFmt, 0,
		XAUDIO2_DEFAULT_FREQ_RATIO, &m_callback) != S_OK )
	{
		if (waveFmt != NULL)
			HQ_DELETE (waveFmt);
		throw std::bad_alloc();
	}
	
	if (waveFmt != NULL)
		HQ_DELETE (waveFmt);

	m_callback.SetSource(m_xSource);

	//get default matrix coefficients
	m_xSource->GetOutputMatrix(
		((HQXAudioDevice*) ge_pAudioDevice)->GetMasterVoice(),
		 m_info.numChannels,
		 numOutputChannels,
		m_pDefaultMatrixCoefficients
		);
	
	if (m_info.numChannels == 1)//3d positioning only supported when source is not multi channels
	{
		/*---------------emitter info-----------------------*/
		m_xCone = HQ_NEW X3DAUDIO_CONE;
		m_xCone->InnerAngle = m_xCone->OuterAngle = X3DAUDIO_2PI;
		m_xCone->InnerLPF = m_xCone->OuterLPF = m_xCone->InnerReverb = m_xCone->OuterReverb = 0.f;
		m_xCone->InnerVolume = 1.f;
		m_xCone->OuterVolume = 0.f;
		
		m_xEmitter.ChannelCount = 1;
		/*-----doesn't use--------*/
		m_xEmitter.ChannelRadius = 0.f;
		m_xEmitter.pChannelAzimuths = HQ_NEW FLOAT32[1];
		m_xEmitter.pChannelAzimuths[0] = 0.f;
		m_xEmitter.InnerRadius = m_xEmitter.InnerRadiusAngle = 0.f;
		/*--------------*/
		m_xEmitter.pCone = NULL;//omnidirectional
		m_xEmitter.pVolumeCurve = NULL;
		m_xEmitter.pLFECurve = NULL;
		m_xEmitter.pReverbCurve = NULL;
		m_xEmitter.pLPFDirectCurve = NULL;
		m_xEmitter.pLPFReverbCurve = NULL;
		/*--------------------*/
		m_xEmitter.Position.x = m_xEmitter.Position.y = m_xEmitter.Position.z = 0.f;
		m_xEmitter.Velocity.x = m_xEmitter.Velocity.y = m_xEmitter.Velocity.z = 0.f;
		m_xEmitter.OrientTop = m_xEmitter.OrientFront = m_xEmitter.Position;//zero vector
		/*------------------*/
		m_xEmitter.CurveDistanceScaler = 1.f;
		m_xEmitter.DopplerScaler = 1.f;
		
		/*---------dsp setting----------------*/
		ZeroMemory(&m_xDSPSettings, sizeof (X3DAUDIO_DSP_SETTINGS));
		m_xDSPSettings.pMatrixCoefficients = HQ_NEW FLOAT32[numOutputChannels];
		m_xDSPSettings.SrcChannelCount = 1;
		m_xDSPSettings.DstChannelCount = numOutputChannels;
	}
	else//multi channels
	{
		/*------doesn't use 3d positioning mode--------*/
		m_xEmitter.pChannelAzimuths = NULL;
		m_xDSPSettings.pMatrixCoefficients = NULL;
		m_xCone = NULL;
		m_positioning = HQ_FALSE;
	}

	AttachBuffer(pBuffer);

	this->SetAudioSetting();
}
HQXAudioSourceController::~HQXAudioSourceController() 
{
	SafeDeleteArray(m_xEmitter.pChannelAzimuths);
	SafeDeleteArray(m_xDSPSettings.pMatrixCoefficients);
	SafeDeleteArray(m_pDefaultMatrixCoefficients);
	SafeDelete(m_xCone);

	if (m_xSource != NULL)
		m_xSource->DestroyVoice();
}

void HQXAudioSourceController::SetAudioSetting()
{
	if (m_positioning == HQ_TRUE)//3d positioning
	{
		X3DAudioCalculate(
			((HQXAudioDevice*) ge_pAudioDevice)->Get3DAudioHandle(),
			&((HQXAudioDevice*) ge_pAudioDevice)->GetListener(),
			&m_xEmitter,
			m_3DAudioCalflags,
			&m_xDSPSettings)
			;

		m_xSource->SetOutputMatrix(((HQXAudioDevice*) ge_pAudioDevice)->GetMasterVoice(),
			1 ,
			((HQXAudioDevice*) ge_pAudioDevice)->GetMasterVoiceInputChannels() ,
			m_xDSPSettings.pMatrixCoefficients);

		m_xSource->SetFrequencyRatio(m_xDSPSettings.DopplerFactor);

	}
	else
	{
		//disable 3d positioning audio
		m_xSource->SetOutputMatrix(((HQXAudioDevice*) ge_pAudioDevice)->GetMasterVoice(),
			m_info.numChannels ,
			((HQXAudioDevice*) ge_pAudioDevice)->GetMasterVoiceInputChannels() ,
			m_pDefaultMatrixCoefficients);

		m_xSource->SetFrequencyRatio(1.f);

	}


}

HQReturnVal HQXAudioSourceController::AttachBuffer(HQSharedPtr<HQBaseAudioBuffer> &pBuffer)
{
	HQBaseAudioBuffer *pRawBuffer = pBuffer.GetRawPointer();
	if (m_pBuffer != pRawBuffer)
	{
		m_xSource->Stop();
		m_xSource->FlushSourceBuffers();//remove old buffer

		if (m_pBuffer != NULL && m_pBuffer->m_type == HQBaseAudioBuffer::STREAM_BUFFER)
		{
			//detach old buffer
			HQXAudioStreamBuffer *xOldBuffer = (HQXAudioStreamBuffer*)m_pBuffer.GetRawPointer();
			xOldBuffer->AttachSource(NULL);
		}

		if (pRawBuffer != NULL)
		{

			switch (pRawBuffer->m_type)
			{
			case HQBaseAudioBuffer::NORMAL_BUFFER:
				{
					HQXAudioBuffer* xBuffer = (HQXAudioBuffer*)pRawBuffer;

					if (xBuffer->GetDataInfo().bits != m_info.bits || 
							xBuffer->GetDataInfo().channels != m_info.numChannels)
					{
						ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
						return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
					}
					m_xSource->SetSourceSampleRate(xBuffer->GetDataInfo().sampleRate);
					if (FAILED(m_xSource->SubmitSourceBuffer(&xBuffer->GetBuffer())))
						return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;

					m_callback.SwitchDelegate(false);
				}
				break;
			case HQBaseAudioBuffer::STREAM_BUFFER:
				{
					HQXAudioStreamBuffer *xBuffer = (HQXAudioStreamBuffer*)pRawBuffer;

					if (xBuffer->GetAttachedSource() != NULL)
					{
						ge_pAudioDevice->Log ("Error : Streaming buffer cannot attach to multiple sources!");
						return HQ_FAILED;
					}

					if (xBuffer->GetInfo().bits != m_info.bits || 
							xBuffer->GetInfo().channels != m_info.numChannels)
					{
						ge_pAudioDevice->Log ("Error : Buffer is not compatible with Source!");
						return HQ_FAILED_AUDIO_SOURCE_BUFFER_MISMATCH;
					}
					m_xSource->SetSourceSampleRate(xBuffer->GetInfo().sampleRate);

					xBuffer->AttachSource(m_xSource);

					m_callback.SwitchDelegate(true);

				}
				break;
			}
		}//if (pRawBuffer != NULL)

		m_pBuffer = pBuffer;
	}//if (m_pBuffer != pRawBuffer)

	return HQ_OK;
}

HQReturnVal HQXAudioSourceController::Enable3DPositioning(HQBool enable)
{
	if(m_positioning == enable)//no changes
		return HQ_OK;
	if (enable && m_info.numChannels != 1)//multichannels not supported in 3D positioning mode
	{
		ge_pAudioDevice->Log ("Error : Can't enable 3D positioning mode because this source is multi channels!");
		return HQ_FAILED;
	}

	m_positioning = enable;

	this->SetAudioSetting();

	return HQ_OK;
}

///default position is {0 , 0 , 0}
HQReturnVal HQXAudioSourceController::SetPosition(const hqfloat32 position[3]) 
{
#if defined _DEBUG || defined DEBUG
	if (m_positioning == HQ_FALSE)
		return HQ_FAILED;
#endif
	memcpy(&m_xEmitter.Position , position, 3 * sizeof(hqfloat32));
	this->SetAudioSetting();
	return HQ_OK;
}
///default velocity is {0 , 0 , 0}
HQReturnVal HQXAudioSourceController::SetVelocity(const hqfloat32 velocity[3]) 
{
	memcpy(&m_xEmitter.Velocity , velocity, 3 * sizeof(hqfloat32));
	this->SetAudioSetting();
	return HQ_OK;
}
///default direction is {0 , 0 , 0} => omnidirectional
HQReturnVal HQXAudioSourceController::SetDirection(const hqfloat32 direction[3]) 
{
	HQ_DECL_STACK_VECTOR4_CTOR_PARAMS(directionVec, (direction[0], direction[1], direction[2]));
	hqfloat32 lenSqr = directionVec.LengthSqr();
	if (lenSqr != 0)
	{
		if (lenSqr < 1.0f - 0.00001f || lenSqr > 1.0f + 0.00001f)//normalize if not unit length
		{
			hqfloat32 _1overLen = 1.0f / sqrtf(lenSqr);
			directionVec *= _1overLen;
		}

		memcpy(&m_xEmitter.OrientFront , directionVec, 3 * sizeof(hqfloat32));

		m_xEmitter.pCone = m_xCone;
	}
	else//omnidirectional
	{
		m_xEmitter.pCone = NULL;

	}
	this->SetAudioSetting();
	return HQ_OK;
}


HQReturnVal HQXAudioSourceController::SetInnerAngle(hqfloat32 innerAngle) 
{
#if defined _DEBUG || defined DEBUH
	if (innerAngle < 0.0f || innerAngle > 360.f)
		return HQ_FAILED_INVALID_PARAMETER;
#endif
	m_xCone->InnerAngle = HQToRadian( innerAngle) ;

	this->SetAudioSetting();
	return HQ_OK;
}

HQReturnVal HQXAudioSourceController::SetOuterAngle(hqfloat32 outerAngle)
{
#if defined _DEBUG || defined DEBUH
	if (outerAngle < 0.0f || outerAngle > 360.f)
		return HQ_FAILED_INVALID_PARAMETER;
#endif
	m_xCone->OuterAngle = HQToRadian( outerAngle );

	this->SetAudioSetting();
	return HQ_OK;
}


HQReturnVal HQXAudioSourceController::SetMaxDistance(hqfloat32 maxDistance) 
{
#if defined _DEBUG || defined DEBUH
	if (maxDistance < 0.0f)
		return HQ_FAILED_INVALID_PARAMETER;
#endif
	ge_pAudioDevice->Log("Error : SetMaxDistance() is not supported!");
	return HQ_FAILED;
}

HQReturnVal HQXAudioSourceController::SetVolume(hqfloat32 volume)
{
#if defined _DEBUG || defined DEBUH
	if (volume < 0.0f || volume > 1.f)
		return HQ_FAILED_INVALID_PARAMETER;
#endif
	m_xSource->SetVolume(volume);
	return HQ_OK;
}

HQReturnVal HQXAudioSourceController::Play(HQBool continuous) 
{
	if (!m_callback.IsPlaying())
	{
		m_callback.StartPlaying(continuous);
	
		m_xSource->Start();

		m_state = PLAYING;

	}

	return HQ_OK;
}
HQReturnVal HQXAudioSourceController::Pause()
{
	m_xSource->Stop();
	m_callback.Invalidate();

	m_state = PAUSED;
	return HQ_OK;
}
HQReturnVal HQXAudioSourceController::Stop()
{
	if (m_callback.IsPlaying())//source is still playing
	{
		m_xSource->Stop();
		m_xSource->FlushSourceBuffers();
		m_callback.Invalidate();

		//reset buffer
		HQBaseAudioBuffer *pRawBuffer = m_pBuffer.GetRawPointer();

		if (pRawBuffer != NULL)
		{
			switch (pRawBuffer->m_type)
			{
			case  HQBaseAudioBuffer::NORMAL_BUFFER:
				{
					HQXAudioBuffer* xBuffer = (HQXAudioBuffer*)pRawBuffer;
					m_xSource->SubmitSourceBuffer(&xBuffer->GetBuffer());
				}
				break;
			case  HQBaseAudioBuffer::STREAM_BUFFER:
				{
					HQXAudioStreamBuffer * xBuffer = (HQXAudioStreamBuffer*)pRawBuffer;
					xBuffer->RestartStream(false);//do not re push used buffer list, because the used buffer will automatically be added by OnBufferEnd
				}
				break;
			}
		}
	}
	m_state = STOPPED;
	return HQ_OK;
}
HQAudioSourceController::State HQXAudioSourceController::GetCurrentState()
{
	if (m_state == PLAYING && !m_callback.IsPlaying())
		return m_state = STOPPED;
	return m_state;
}

HQReturnVal HQXAudioSourceController::UpdateStream()
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
				HQXAudioStreamBuffer* xBuffer = (HQXAudioStreamBuffer*)pRawBuffer;
				xBuffer->UpdateStream();
			}
			break;
		}
	}
	return HQ_OK;
}

WAVEFORMATEX * HQXAudioSourceController::CreateWaveFormat(const HQAudioSourceInfo &info)
{
	WAVEFORMATEX * waveFmt = NULL;
	switch(info.formatType)
	{
	case HQ_AUDIO_PCM:
		if (info.numChannels == 1 || info.numChannels == 2)
		{
			waveFmt = HQ_NEW WAVEFORMATEX();
			waveFmt->cbSize = 0;
			waveFmt->nChannels = info.numChannels;
			waveFmt->nSamplesPerSec = 8000;
			waveFmt->wBitsPerSample = info.bits;
			waveFmt->nBlockAlign = info.numChannels * info.bits / 8;
			waveFmt->nAvgBytesPerSec = waveFmt->nSamplesPerSec * waveFmt->nBlockAlign;
			waveFmt->wFormatTag = WAVE_FORMAT_PCM;
		}
	}

	return waveFmt;
}



/*-----------HQXAudioSourceControllerRH - right handed coordinates using source--------------------*/

///default position is {0 , 0 , 0}
HQReturnVal HQXAudioSourceControllerRH::SetPosition(const hqfloat32 position[3]) 
{
#if defined _DEBUG || defined DEBUG
	if (m_positioning == HQ_FALSE)
		return HQ_FAILED;
#endif
	m_xEmitter.Position.x = position[0];
	m_xEmitter.Position.y = position[1];
	m_xEmitter.Position.z = -position[2];

	this->SetAudioSetting();
	return HQ_OK;
}
///default velocity is {0 , 0 , 0}
HQReturnVal HQXAudioSourceControllerRH::SetVelocity(const hqfloat32 velocity[3]) 
{
	m_xEmitter.Velocity.x = velocity[0];
	m_xEmitter.Velocity.y = velocity[1];
	m_xEmitter.Velocity.z = -velocity[2];

	this->SetAudioSetting();
	return HQ_OK;
}
///default direction is {0 , 0 , 0} => omnidirectional
HQReturnVal HQXAudioSourceControllerRH::SetDirection(const hqfloat32 direction[3]) 
{
	HQ_DECL_STACK_VECTOR4_CTOR_PARAMS(directionVec, (direction[0], direction[1], direction[2]));
	hqfloat32 lenSqr = directionVec.LengthSqr();
	if (lenSqr != 0)
	{
		if (lenSqr < 1.0f - 0.00001f || lenSqr > 1.0f + 0.00001f)//normalize if not unit length
		{
			hqfloat32 _1overLen = 1.0f / sqrtf(lenSqr);
			directionVec *= _1overLen;
		}

		m_xEmitter.OrientFront.x = directionVec.x;
		m_xEmitter.OrientFront.y = directionVec.y;
		m_xEmitter.OrientFront.z = -directionVec.z;

		m_xEmitter.pCone = m_xCone;
	}
	else//omnidirectional
	{
		m_xEmitter.pCone = NULL;

	}
	this->SetAudioSetting();
	return HQ_OK;
}
