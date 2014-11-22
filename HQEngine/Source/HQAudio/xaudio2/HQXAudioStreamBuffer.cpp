/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "HQXAudioBuffer.h"

#define BUFFER_STILL_ALIVE 0x1 
#define BUFFER_FIRST_ATTACH_STATE 0x2

#pragma message ("TO DO: it is not safe when delete buffers before delete source because most of Xaudio2's methods are async")

/*-------------------------------------------*/
HQXAudioStreamBuffer::HQXAudioStreamBuffer(
		const char *fileName, 
		HQBaseAudioDevice::AudioInfo &info,
		HQStreamBufferSupplier *supplier,
		size_t numSubBuffers,
		size_t subBufferSize)
		: HQBaseAudioBuffer(fileName, STREAM_BUFFER), 
		HQThread("HQXAudioStreamBuffer Thread"),
		m_info(info) , m_state(BUFFER_STILL_ALIVE | BUFFER_FIRST_ATTACH_STATE), m_streamPosition(0),
		m_supplier(supplier), m_numSubBuffers(numSubBuffers), m_subBufferSize (subBufferSize),
		m_xSubBuffers(HQ_NEW XAUDIO2_BUFFER_EX[numSubBuffers]),
		m_subBufferListMemPool(HQ_NEW HQPoolMemoryManager(sizeof(SubBufferListType::LinkedListNodeType), numSubBuffers)),
		m_usedBufferList (m_subBufferListMemPool),
		m_unusedBufferList(m_subBufferListMemPool)
{
	//init buffer
	for (size_t i = 0; i < m_numSubBuffers; ++i)
	{
		m_xSubBuffers[i].pAudioData = m_xSubBuffers[i].pActualData = HQ_NEW BYTE[m_subBufferSize];
		m_xSubBuffers[i].AudioBytes = m_subBufferSize;
		m_xSubBuffers[i].Flags = 0;
		m_xSubBuffers[i].PlayBegin = 0;
		m_xSubBuffers[i].PlayLength = 0;
		m_xSubBuffers[i].LoopBegin = m_xSubBuffers[i].LoopLength = 0;
		m_xSubBuffers[i].LoopCount = 0;
		
		m_xSubBuffers[i].pParent = this;
		m_xSubBuffers[i].pContext = m_xSubBuffers + i;//point to itself
	}

	m_playTime = (hqfloat32) ((double)info.samples / info.sampleRate);//playback duration


	m_xAttachedSource = NULL;

	FillSubBuffers();

	Start();//start worker thread
}

HQXAudioStreamBuffer::~HQXAudioStreamBuffer()
{
	m_lock.Lock();

	m_state &= ~BUFFER_STILL_ALIVE;

	m_usedBufferList.RemoveAll();
	m_unusedBufferList.RemoveAll();

	m_lock.Signal();//signal the worker thread to stop running

	m_lock.Unlock();

	HQ_DELETE(m_supplier);

	if (m_xSubBuffers != NULL)
	{
		for (size_t i = 0; i < m_numSubBuffers; ++i)
		{
			delete[] (m_xSubBuffers[i].pActualData);
		}
		delete[] (m_xSubBuffers);
	}

	this->Join();//join with worker thread
}


void HQXAudioStreamBuffer::OnSubBufferUsed(void *pBufferContext)
{
	XAUDIO2_BUFFER_EX * subBuffer = (XAUDIO2_BUFFER_EX*)pBufferContext;

	subBuffer->pParent->OnSubBufferUsed(subBuffer);
}

void HQXAudioStreamBuffer::OnSubBufferUsed(XAUDIO2_BUFFER_EX* subBuffer)
{
	m_lock.Lock();

	m_usedBufferList.PushBack(subBuffer);//push to used list

	m_lock.Signal();

	m_lock.Unlock();
}

void HQXAudioStreamBuffer::AttachSource(IXAudio2SourceVoice * source)
{
	m_lock.Lock();
	m_xAttachedSource = source;
	
	if (BUFFER_FIRST_ATTACH_STATE & m_state)//first attachment
	{
		//submit all sub buffers
		if (m_xAttachedSource != NULL)
		{
			SubBufferListType::LinkedListNodeType* node = m_unusedBufferList.GetRoot();
			while (m_unusedBufferList.GetSize() > 0)
			{
				HRESULT hr = m_xAttachedSource->SubmitSourceBuffer(node->m_element);

				SubBufferListType::LinkedListNodeType* oldNode = node;

				node = node->m_pNext;
				
				if (!FAILED(hr))
				{
					//remove from un-used buffer list
					m_unusedBufferList.RemoveAt(oldNode);
				}

			}
		}


		m_state &= ~BUFFER_FIRST_ATTACH_STATE;
	}
	else
	{
		RestartStreamNonLock(false);
	}

	m_lock.Unlock();
}

void HQXAudioStreamBuffer::FillSubBuffers()
{
	m_streamPosition = 0;

	m_supplier->Rewind();

	for (size_t i = 0; i < m_numSubBuffers; ++i)
		this->FillSubBuffer(m_xSubBuffers + i);

}

void HQXAudioStreamBuffer::FillSubBuffer(XAUDIO2_BUFFER_EX* subBuffer)
{
	if (m_streamPosition >= m_info.size)
	{
		m_supplier ->Rewind();
		m_streamPosition = 0;//reset stream
	}
	m_streamPosition += subBuffer->AudioBytes = m_supplier->GetNextData(subBuffer->pActualData, m_subBufferSize);

	subBuffer->pAudioData = subBuffer->pActualData;//have to do this because pAudioData is const

	if (m_streamPosition >= m_info.size)//the last chunk
		subBuffer->Flags = XAUDIO2_END_OF_STREAM;
	else
		subBuffer->Flags = 0;

	m_unusedBufferList.PushBack(subBuffer);//push to unused buffer list
}

void HQXAudioStreamBuffer::RestartStream(bool repushUsedList)
{
	m_lock.Lock();
	RestartStreamNonLock(repushUsedList);
	m_lock.Unlock();
}

void HQXAudioStreamBuffer::RestartStreamNonLock(bool repushUsedList)
{
	//force to start stream at the beginning on next buffer filling
	m_streamPosition = m_info.size;


	if (repushUsedList)
	{
		//invalidate all sub buffers
		m_usedBufferList.RemoveAll();
		m_unusedBufferList.RemoveAll();

		for (size_t i = 0; i < m_numSubBuffers; ++i)
		{
			m_usedBufferList.PushBack(m_xSubBuffers + i);//push to used list so it can be re-use
		}
	}
	else
	{
		
		//transfer from un-used buffer to used buffer list 
		SubBufferListType::LinkedListNodeType* node = m_unusedBufferList.GetRoot();
		while (m_unusedBufferList.GetSize() > 0)
		{
			XAUDIO2_BUFFER_EX *buffer = node->m_element;
			SubBufferListType::LinkedListNodeType* oldNode = node;

			node = node->m_pNext;
			
			m_unusedBufferList.RemoveAt(oldNode);//remove from un-used buffer list

			m_usedBufferList.PushBack(buffer);//push to used buffer list

		}
	}


	m_lock.Signal();//signal worker thread to start loading sub buffer
}


void HQXAudioStreamBuffer::UpdateStream()
{
	HQ_ASSERT(m_xAttachedSource != NULL);

	if (m_lock.TryLock())
	{
		SubBufferListType::LinkedListNodeType* node = m_unusedBufferList.GetRoot();
		while (m_unusedBufferList.GetSize() > 0)
		{
			HRESULT hr = m_xAttachedSource->SubmitSourceBuffer(node->m_element);

			SubBufferListType::LinkedListNodeType* oldNode = node;

			node = node->m_pNext;
			
			if (!FAILED(hr))
			{
				//remove from un-used buffer list
				m_unusedBufferList.RemoveAt(oldNode);
			}

		}

		m_lock.Unlock();
	}
}

void HQXAudioStreamBuffer::Run()
{
	m_lock.Lock();

	while (m_state & BUFFER_STILL_ALIVE)
	{
		while (m_usedBufferList.GetSize() > 0)
		{
			//get the first free buffer
			XAUDIO2_BUFFER_EX* buffer = m_usedBufferList.GetFront();

			m_usedBufferList.PopFront();//remove from used list

			FillSubBuffer(buffer);//fill buffer and push to unused list


		}
		m_lock.Wait();

	}
	m_lock.Unlock();
}
