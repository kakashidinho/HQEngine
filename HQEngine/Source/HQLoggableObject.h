/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_LOG_OBJ_H
#define HQ_LOG_OBJ_H


#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "HQPrimitiveDataType.h"
#include "HQLogStream.h"

class HQLoggableObject
{
public :
	HQLoggableObject(const char *logPrefix , bool flushLog = false ,hq_uint32 bufferSize = 512)
	{
        if (bufferSize == 0)
            bufferSize = 1;
		this->m_flushLog = flushLog;
		this->m_pLogStream = NULL;
		this->m_logPrefix = HQ_NEW char[strlen(logPrefix) + 1];
		this->m_logBuffer = HQ_NEW char[bufferSize];
		this->m_bufferSize = bufferSize;
		strcpy(this->m_logPrefix , logPrefix);
	}
	HQLoggableObject (HQLogStream *pLogStream ,const char *logPrefix , bool flushLog = false,hq_uint32 bufferSize = 512)
	{
        if (bufferSize == 0)
            bufferSize = 1;
		this->m_flushLog = flushLog;
		this->m_pLogStream = pLogStream;
		this->m_logPrefix = HQ_NEW char[strlen(logPrefix) + 1];
		this->m_logBuffer = HQ_NEW char[bufferSize];
        this->m_bufferSize = bufferSize;
		strcpy(this->m_logPrefix , logPrefix);
	}
	~HQLoggableObject()
	{
		delete[] this->m_logPrefix;
		delete[] this->m_logBuffer;
	}
	
	void SetLogStream(HQLogStream *pLogStream)
	{
		this->m_pLogStream = pLogStream;
	}

	virtual void Log(const char *format, ...)
	{
		if(m_pLogStream == NULL)
			return;
		va_list args;
		va_start(args,format);
		int re = vsnprintf(this->m_logBuffer, m_bufferSize, format, args);
		if (re < 0 || (hquint32)re >= m_bufferSize)
		{
			m_logBuffer[m_bufferSize - 1] = '\0';//truncated log message
		}

		this->m_pLogStream->Log(m_logPrefix, m_logBuffer);
		if(m_flushLog)
			this->m_pLogStream->Flush();
		va_end(args);
	}

	virtual void LogMessage(const char *message)
	{
		if (m_pLogStream == NULL)
			return;
		m_pLogStream->Log(m_logPrefix, message);
		if (m_flushLog)
			this->m_pLogStream->Flush();
	}

protected:
	bool m_flushLog;
	HQLogStream * m_pLogStream;
	char *m_logPrefix;
	char *m_logBuffer;
	hquint32 m_bufferSize;
};

#endif
