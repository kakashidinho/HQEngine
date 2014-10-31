/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "HQAudioBufferAL.h"

#define BUFFER_STILL_ALIVE 0x1 
#define BUFFER_FIRST_ATTACH_STATE 0x2
#define BUFFER_LOOP_ENABLE 0x4
#define BUFFER_STREAM_RESTART 0x8
//#define BUFFER_SOURCE_STARTING 0x10
#define BUFFER_STREAM_STOP 0x20
#define BUFFER_SOURCE_NEED_RESTART 0x40
#define BUFFER_SOURCE_MUST_STOP 0x80

/*-------------------------------------------*/
HQAudioStreamBufferAL::HQAudioStreamBufferAL(
		const char *fileName, 
		ALenum alFormat, 
		HQBaseAudioDevice::AudioInfo &info,
		HQStreamBufferSupplier *supplier,
		size_t numSubBuffers,
		size_t subBufferSize)
		: HQBaseAudioBuffer(STREAM_BUFFER), 
		HQThread("HQAudioStreamBufferAL Thread"),
		m_formatAL (alFormat),
		m_info(info) , m_state(BUFFER_STILL_ALIVE | BUFFER_FIRST_ATTACH_STATE), m_streamPosition(0),
		m_supplier(supplier), m_numSubBuffers(numSubBuffers), m_subBufferSize (subBufferSize),
		m_subBuffers(HQ_NEW ALuint[numSubBuffers]),
		m_removedSubBuffersCache(HQ_NEW ALuint[numSubBuffers]), 
		m_subBufferClientData (HQ_NEW SubBufferData[numSubBuffers]),
		m_subBufferListMemPool(HQ_NEW HQPoolMemoryManager(sizeof(SubBufferClientDataListType::LinkedListNodeType), numSubBuffers)),
		m_usedBufferClienDataList (m_subBufferListMemPool),
		m_unusedBufferClienDataList(m_subBufferListMemPool),
		m_removedSubBuffers(HQ_NEW HQPoolMemoryManager(sizeof(SubBufferALListType::LinkedListNodeType), numSubBuffers))
{
	//specify buffer format
	this->SetHQFormatType(alFormat);

	//init sub buffers
	m_subBuffers[0] = 0;

	alGenBuffers(m_numSubBuffers, m_subBuffers);
	if (AL_OUT_OF_MEMORY == alGetError())
			throw std::bad_alloc();

	//init sub buffers storage
	for (size_t i = 0; i < m_numSubBuffers; ++i)
	{
		m_subBufferClientData[i].data = HQ_NEW hqubyte8 [m_subBufferSize];//create client data buffer
		m_subBufferClientData[i].dataSize = 0;

		alBufferData(m_subBuffers[i] , m_formatAL , m_subBufferClientData[i].data, m_subBufferSize, m_info.sampleRate);
		if (AL_OUT_OF_MEMORY == alGetError())
			throw std::bad_alloc();
	}

	m_playTime = (hqfloat32) ((double)info.samples / info.sampleRate);//playback duration


	m_attachedSource = 0;

	FillSubBuffers();

	Start();//start worker thread
}

HQAudioStreamBufferAL::~HQAudioStreamBufferAL()
{
	m_lock.Lock();

	m_state &= ~BUFFER_STILL_ALIVE;

	m_usedBufferClienDataList.RemoveAll();
	m_unusedBufferClienDataList.RemoveAll();

	m_lock.Signal();//signal the worker thread to stop running

	m_lock.Unlock();

	HQ_DELETE(m_supplier);

	if (m_subBuffers != NULL)
	{
		if (m_subBuffers[0] != 0)
			alDeleteBuffers(m_numSubBuffers, m_subBuffers);
		delete[] (m_subBuffers);
	}

	for (size_t i = 0; i < m_numSubBuffers; ++i)
	{
		delete[] m_subBufferClientData[i].data; 
	}
	
	delete[] (m_subBufferClientData);

	delete[] (m_removedSubBuffersCache);

	this->Join();//join with worker thread

}

void HQAudioStreamBufferAL::AttachSource(ALuint  source)
{
	m_lock.Lock();
	m_attachedSource = source;
	
	if (BUFFER_FIRST_ATTACH_STATE & m_state)//first attachment
	{
		//submit all sub buffers
		if (m_attachedSource != 0)
		{
			size_t idx = 0;

			SubBufferClientDataListType::LinkedListNodeType* node = m_unusedBufferClienDataList.GetRoot();
			while (m_unusedBufferClienDataList.GetSize() > 0)
			{
				SubBufferData * clientData = node->m_element;

				//copy client data to buffer
				alBufferData(m_subBuffers[idx] , m_formatAL , clientData->data, clientData->dataSize, m_info.sampleRate);
				//queue buffer
				alSourceQueueBuffers(m_attachedSource, 1, &m_subBuffers[idx]);

				SubBufferClientDataListType::LinkedListNodeType* oldNode = node;

				node = node->m_pNext;
				
				//remove from un-used buffer client data list
				m_unusedBufferClienDataList.RemoveAt(oldNode);

				//push to used buffer client data list
				m_usedBufferClienDataList.PushBack(clientData);
				
				idx ++;
			}

			m_removedSubBuffers.RemoveAll();
		}


		m_state &= ~BUFFER_FIRST_ATTACH_STATE;
	}
	else
	{
		RestartStreamNonLock(true);
	}

	m_lock.Unlock();
}

void HQAudioStreamBufferAL::FillSubBuffers()
{
	m_streamPosition = 0;

	m_supplier->Rewind();

	for (size_t i = 0; i < m_numSubBuffers; ++i)
		this->FillSubBuffer(m_subBufferClientData + i);

}

void HQAudioStreamBufferAL::FillSubBuffer(SubBufferData * bufferClientData)
{
	if (m_streamPosition >= m_info.size)
	{
		m_supplier ->Rewind();
		m_streamPosition = 0;//reset stream
	}

	//read data to intermediate buffer
	m_streamPosition += bufferClientData->dataSize = m_supplier->GetNextData(bufferClientData->data, m_subBufferSize);

	m_unusedBufferClienDataList.PushBack(bufferClientData);//push to unused buffer client data list
}

void HQAudioStreamBufferAL::RestartStream(bool repushUsedList)
{
	m_lock.Lock();
	RestartStreamNonLock(repushUsedList);
	m_lock.Unlock();
}

void HQAudioStreamBufferAL::RestartStreamNonLock(bool repushUsedList)
{
	//force to start stream at the beginning on next buffer filling
	m_streamPosition = m_info.size;
	m_state |= BUFFER_STREAM_RESTART;

	m_removedSubBuffers.RemoveAll();

	for (size_t i = 0; i < m_numSubBuffers; ++i)//all sub buffers are removed from source
		m_removedSubBuffers.PushBack(m_subBuffers[i]);

	if (repushUsedList)
	{
		//invalidate all sub buffers
		m_usedBufferClienDataList.RemoveAll();
		m_unusedBufferClienDataList.RemoveAll();

		for (size_t i = 0; i < m_numSubBuffers; ++i)
		{
			m_usedBufferClienDataList.PushBack(m_subBufferClientData + i);//push to used list so it can be re-use
		}
	}
	else
	{
		
		//transfer from un-used buffer to used buffer list 
		SubBufferClientDataListType::LinkedListNodeType* node = m_unusedBufferClienDataList.GetRoot();
		while (m_unusedBufferClienDataList.GetSize() > 0)
		{
			SubBufferData* bufferClienData = node->m_element;
			SubBufferClientDataListType::LinkedListNodeType* oldNode = node;

			node = node->m_pNext;
			
			m_unusedBufferClienDataList.RemoveAt(oldNode);//remove from un-used buffer client data list

			m_usedBufferClienDataList.PushBack(bufferClienData);//push to used buffer client data list

		}
	}


	m_lock.Signal();//signal worker thread to start loading sub buffers
}

void HQAudioStreamBufferAL::PrepareToPlay()
{
	m_state |= BUFFER_STREAM_RESTART;
	m_state &= ~BUFFER_SOURCE_MUST_STOP; 
}

void HQAudioStreamBufferAL::EnableLoop(bool loop)
{
	if (loop)
		m_state |= BUFFER_LOOP_ENABLE;
	else
		m_state &= ~BUFFER_LOOP_ENABLE;
}

void HQAudioStreamBufferAL::Stop()
{
	m_state |= BUFFER_SOURCE_MUST_STOP; 

	RestartStream(true);
}

void HQAudioStreamBufferAL::UpdateStream()
{
	HQ_ASSERT(m_attachedSource != 0);

	if (m_lock.TryLock())
	{
		//stream restart flag still not removed by worker thread while free buffer client data are available and there are equal or more than one removed buffer
		if ((m_state & BUFFER_STREAM_RESTART) != 0 &&
			m_usedBufferClienDataList.GetSize() > 0 &&
			m_removedSubBuffers.GetSize() > 0)
			m_lock.Signal();//wake the worker thread

		ALsizei removedSubBuffers = 0;

		//remove used buffers from source
		alGetSourcei(m_attachedSource, AL_BUFFERS_PROCESSED, &removedSubBuffers);

		HQ_ASSERT((size_t)removedSubBuffers <= m_numSubBuffers);

		if (removedSubBuffers > 0)
		{
			//get used buffers
			alSourceUnqueueBuffers(m_attachedSource, removedSubBuffers, m_removedSubBuffersCache);
			
			for (ALsizei i = 0; i < removedSubBuffers; ++i)
				m_removedSubBuffers.PushBack(m_removedSubBuffersCache[i]);//push to removed buffers list

			m_lock.Signal();
		}

		//copy un-used buffers' client data and queue the buffer
		SubBufferClientDataListType::LinkedListNodeType* node = m_unusedBufferClienDataList.GetRoot();
		SubBufferALListType::LinkedListNodeType *removedBufferNode = m_removedSubBuffers.GetRoot();
		//get current state of source
		ALint sourceState = 0;

		alGetSourcei(m_attachedSource, AL_SOURCE_STATE, &sourceState);

		while (m_unusedBufferClienDataList.GetSize() > 0 && m_removedSubBuffers.GetSize() > 0)
		{
			SubBufferData * clientData = node->m_element;

			//copy client data to buffer
			alBufferData(removedBufferNode->m_element , m_formatAL , clientData->data, clientData->dataSize, m_info.sampleRate);
			//queue buffer
			alSourceQueueBuffers(m_attachedSource, 1, &removedBufferNode->m_element);

			SubBufferClientDataListType::LinkedListNodeType* oldNode = node;
			SubBufferALListType::LinkedListNodeType * oldRemovedBuferNode = removedBufferNode;

			node = node->m_pNext;
			removedBufferNode = removedBufferNode->m_pNext;
			
			//remove from un-used buffer client data list
			m_unusedBufferClienDataList.RemoveAt(oldNode);

			//push to used client data list
			m_usedBufferClienDataList.PushBack(clientData);
			
			//remove the buffer from removed buffers list
			m_removedSubBuffers.RemoveAt(oldRemovedBuferNode);

			//need to restart source if it is stopped
			if (sourceState != AL_PLAYING && (m_state & BUFFER_SOURCE_MUST_STOP) == 0)//source is not playing
			{
				alSourcePlay(m_attachedSource);

				sourceState = AL_PLAYING;
			}
		}

		m_lock.Unlock();
	}
}

void HQAudioStreamBufferAL::Run()
{
	m_lock.Lock();

	while (m_state & BUFFER_STILL_ALIVE)
	{
		hquint32 numRefills = 0;

		while (numRefills < m_removedSubBuffers.GetSize() && m_usedBufferClienDataList.GetSize() > 0)
		{
			//get the first free buffer
			SubBufferData * clientData = m_usedBufferClienDataList.GetFront();
			
			if (m_streamPosition < m_info.size ||
				(m_state & BUFFER_STREAM_RESTART) != 0 ||
				(m_state & BUFFER_LOOP_ENABLE) != 0)
			{
				m_usedBufferClienDataList.PopFront();//remove from used client data list

				FillSubBuffer(clientData);//fill buffer and push to unused list

				m_state &= ~BUFFER_STREAM_RESTART;//remove restart flag
			}
			
			numRefills ++;

		}
		m_lock.Wait();

	}
	m_lock.Unlock();
}
