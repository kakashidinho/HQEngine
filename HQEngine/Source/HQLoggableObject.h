/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
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
		this->m_flushLog = flushLog;
		this->m_pLogStream = NULL;
		this->m_logPrefix = HQ_NEW char[strlen(logPrefix) + 1];
		this->m_logBuffer = HQ_NEW char[bufferSize];
		strcpy(this->m_logPrefix , logPrefix);
	}
	HQLoggableObject (HQLogStream *pLogStream ,const char *logPrefix , bool flushLog = false,hq_uint32 bufferSize = 512)
	{
		this->m_flushLog = flushLog;
		this->m_pLogStream = pLogStream;
		this->m_logPrefix = HQ_NEW char[strlen(logPrefix) + 1];
		this->m_logBuffer = HQ_NEW char[bufferSize];
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
		vsprintf(this->m_logBuffer,format,args);
		this->m_pLogStream->Log(m_logPrefix, m_logBuffer);
		if(m_flushLog)
			this->m_pLogStream->Flush();
		va_end(args);
	}

protected:
	bool m_flushLog;
	HQLogStream * m_pLogStream;
	char *m_logPrefix;
	char *m_logBuffer;
};

#endif
