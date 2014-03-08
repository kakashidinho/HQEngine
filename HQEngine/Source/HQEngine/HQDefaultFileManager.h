/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DEFAULT_FILE_MANAGER_H
#define HQ_DEFAULT_FILE_MANAGER_H

#include "../HQFileManager.h"
#include "../HQLinkedList.h"

#include <string>

class HQDefaultFileManager: public HQFileManager {
public:
	~HQDefaultFileManager();
	virtual HQDataReaderStream * OpenFileForRead(const char* name);

	void AddLastSearchPath(const char* path);
	void AddFirstSearchPath(const char* path);

private:
	void Append(const std::string & parent, const char * name, std::string &appendedPath);

	HQLinkedList<std::string> m_searchPaths;
};

#endif