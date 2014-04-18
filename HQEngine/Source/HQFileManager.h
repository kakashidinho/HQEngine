/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_FILE_MANAGER_H
#define HQ_FILE_MANAGER_H

#include "HQDataStream.h"

///
///file manager interface
///
class HQFileManager{
public:
	virtual HQDataReaderStream * OpenFileForRead(const char* name) = 0;
protected:
	virtual ~HQFileManager(){}
};


#endif