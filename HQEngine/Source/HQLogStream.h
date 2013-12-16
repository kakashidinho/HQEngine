#ifndef HQ_LOG_STREAM_H
#define HQ_LOG_STREAM_H

#include "HQPlatformDef.h"

class HQLogStream
{
protected:
	virtual ~HQLogStream(){ };
public:

	virtual void Close()  = 0;// this method should destroy the object
	virtual void Flush() {};
	virtual void Log(const char *tag, const char * message) = 0;
};

#endif