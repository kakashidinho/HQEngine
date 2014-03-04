/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#pragma once
#include "HQXAudioBuffer.h"


struct HQXVoiceCallback : public IXAudio2VoiceCallback
{
public:
	//normal delegate
	struct Delegate{
		STDMETHOD_(void, OnLoopEnd) (void * pBufferContext) {    
			if (!parent->m_loop)//play only once
			{
				parent->m_xsource->Stop();
				InterlockedExchange(&parent->m_playing, HQ_FALSE);
			}
		}

		//Unused methods 
		STDMETHOD_(void, OnStreamEnd) () {  }
		STDMETHOD_(void, OnVoiceProcessingPassEnd) () { }
		STDMETHOD_(void, OnVoiceProcessingPassStart)(UINT32 SamplesRequired) {    }
		STDMETHOD_(void, OnBufferEnd) (void * pBufferContext)    { }
		STDMETHOD_(void, OnBufferStart) (void * pBufferContext) {    }
		STDMETHOD_(void, OnVoiceError) (void * pBufferContext, HRESULT Error) { }

		HQXVoiceCallback *parent;
	};


	///stream mode delegate
	struct StreamModeDelegate: public Delegate
	{
		STDMETHOD_(void, OnLoopEnd) (void * pBufferContext) {   
		}

		STDMETHOD_(void, OnBufferEnd) (void * pBufferContext)    { HQXAudioStreamBuffer::OnSubBufferUsed(pBufferContext); }

		STDMETHOD_(void, OnStreamEnd) () {  
			parent->m_xsource->Stop();
			if (parent->m_loop)//play in loop mode
			{
				//restart
				parent->m_xsource->Start();
			}
			else//play only once
			{
				InterlockedExchange(&parent->m_playing, HQ_FALSE);
			}
		}
	};


	HQXVoiceCallback() : m_playing(HQ_FALSE)
	{
		m_normalDelegate.parent = m_streamDelegate.parent = this;

		m_pCurrentDelegate = &m_normalDelegate;
	}

	void SetSource(IXAudio2SourceVoice *xsource) {m_xsource = xsource;}

    STDMETHOD_(void, OnLoopEnd) (void * pBufferContext) {    
		m_pCurrentDelegate->OnLoopEnd(pBufferContext);
	}

    STDMETHOD_(void, OnStreamEnd) () {  
		m_pCurrentDelegate->OnStreamEnd();
	}
    STDMETHOD_(void, OnVoiceProcessingPassEnd) () { 
		m_pCurrentDelegate->OnVoiceProcessingPassEnd();
	}
    STDMETHOD_(void, OnVoiceProcessingPassStart)(UINT32 SamplesRequired) {   
		m_pCurrentDelegate->OnVoiceProcessingPassStart(SamplesRequired);
	}
    STDMETHOD_(void, OnBufferEnd) (void * pBufferContext)    {
		m_pCurrentDelegate->OnBufferEnd(pBufferContext);
	}
	STDMETHOD_(void, OnBufferStart) (void * pBufferContext) {  
		m_pCurrentDelegate->OnBufferStart(pBufferContext);
	}
    STDMETHOD_(void, OnVoiceError) (void * pBufferContext, HRESULT Error) {
		m_pCurrentDelegate->OnVoiceError(pBufferContext, Error);
	}

	HQBool m_loop;//true if continuous playing
	inline HQBool IsPlaying() const {
		register LONG playing;
		InterlockedExchange(&playing, m_playing);
		return (HQBool)playing;
	}
	inline void Invalidate() {
		m_playing = HQ_FALSE;
	}
	void StartPlaying(HQBool loop)
	{
		m_loop = loop;
		m_playing = HQ_TRUE;
	}

	void SwitchDelegate(bool streamMode)
	{
		if (streamMode)
			m_pCurrentDelegate = &m_streamDelegate;
		else
			m_pCurrentDelegate = &m_normalDelegate;
	}
protected:
	IXAudio2SourceVoice *m_xsource;
	volatile LONG m_playing;

	Delegate m_normalDelegate;
	StreamModeDelegate m_streamDelegate;
	Delegate *m_pCurrentDelegate;
};


/*-----------HQXAudioSourceController - left handed coordinates using source--------------------*/
class HQXAudioSourceController: public  HQBaseAudioSourceController{
public:

	HQXAudioSourceController(IXAudio2 *pXAudio2, const HQAudioSourceInfo &info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer) ;
	~HQXAudioSourceController() ;
	
	const HQSharedPtr<HQBaseAudioBuffer> &GetAttachedBuffer() {return m_pBuffer;}
 
	void SetAudioSetting();

	HQReturnVal AttachBuffer(HQSharedPtr<HQBaseAudioBuffer> &pBuffer);

	HQReturnVal Enable3DPositioning(HQBool enable);

	///default position is {0 , 0 , 0}
	HQReturnVal SetPosition(const hqfloat32 position[3]) ;
	///default velocity is {0 , 0 , 0}
	HQReturnVal SetVelocity(const hqfloat32 velocity[3]) ;
	///default direction is {0 , 0 , 0} => omnidirectional
	HQReturnVal SetDirection(const hqfloat32 direction[3]) ;


	HQReturnVal SetInnerAngle(hqfloat32 innerAngle) ;

	HQReturnVal SetOuterAngle(hqfloat32 outerAngle);


	HQReturnVal SetMaxDistance(hqfloat32 maxDistance) ;
	
	HQReturnVal SetVolume(hqfloat32 volume);

	HQReturnVal Play(HQBool continuous) ;
	HQReturnVal Pause();
	HQReturnVal Stop();
	HQAudioSourceController::State GetCurrentState();

	HQReturnVal UpdateStream();
protected:
	IXAudio2SourceVoice * m_xSource;
	X3DAUDIO_EMITTER m_xEmitter;
	X3DAUDIO_CONE *m_xCone;

	HQBool m_positioning;
private:
	WAVEFORMATEX * CreateWaveFormat(const HQAudioSourceInfo &info);

	HQSharedPtr<HQBaseAudioBuffer> m_pBuffer;
	
	UINT32 m_3DAudioCalflags;
	X3DAUDIO_DSP_SETTINGS m_xDSPSettings;
	FLOAT32 *m_pDefaultMatrixCoefficients;
	HQAudioSourceInfo m_info;

	HQXVoiceCallback m_callback;

	State m_state;
};


/*-----------HQXAudioSourceControllerRH - right handed coordinates using source--------------------*/
class HQXAudioSourceControllerRH: public  HQXAudioSourceController{
public:
	HQXAudioSourceControllerRH(IXAudio2 *pXAudio2, const HQAudioSourceInfo& info, HQSharedPtr<HQBaseAudioBuffer> &pBuffer) 
		:HQXAudioSourceController(pXAudio2, info, pBuffer)
	{
	}

	
	///default position is {0 , 0 , 0}
	HQReturnVal SetPosition(const hqfloat32 position[3]) ;
	///default velocity is {0 , 0 , 0}
	HQReturnVal SetVelocity(const hqfloat32 velocity[3]) ;
	///default direction is {0 , 0 , 0} => omnidirectional
	HQReturnVal SetDirection(const hqfloat32 direction[3]) ;

};
