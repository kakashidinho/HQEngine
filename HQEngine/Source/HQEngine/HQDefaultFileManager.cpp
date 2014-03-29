/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "HQDefaultDataStream.h"
#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#include "winstore/HQWinStoreFileSystem.h"
#endif
#include "HQDefaultFileManager.h"
HQDefaultFileManager::HQDefaultFileManager()
{
	m_searchPaths.PushBack("");
}

HQDefaultFileManager::~HQDefaultFileManager()
{
}

HQDataReaderStream * HQDefaultFileManager::OpenFileForRead(const char* name)
{
	HQLinkedList<std::string>::Iterator ite;
	m_searchPaths.GetIterator(ite);
	std::string fullPath;

	//iterate to every search paths
	for (; !ite.IsAtEnd(); ++ite) {
		this->Append((*ite), name, fullPath);

#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
		HQDataReaderStream * stream = HQWinStoreFileSystem::OpenFileForRead(fullPath.c_str());
#else
		HQSTDDataReader * stream = HQ_NEW HQSTDDataReader(fullPath.c_str());
#endif
		if (stream != NULL && !stream->Good())
		{
			stream->Release();
		}
		else if (stream != NULL)
			return stream;
	}

	return NULL;
}

void HQDefaultFileManager::AddLastSearchPath(const char* path)
{
	m_searchPaths.PushBack(path);
}

void HQDefaultFileManager::AddFirstSearchPath(const char* path)
{
	m_searchPaths.PushFront(path);
}

#define REMOVE_DOUBLE_DOT 0

void HQDefaultFileManager::Append(const std::string & parent, const char * name, std::string &fixedPath)
{
	size_t intendedPathPos = 0;
	size_t fixedPathPos = 0;
	size_t dotCount = 0;
	size_t len ;
	bool needSlash = false;
#if REMOVE_DOUBLE_DOT
	HQLinkedList<size_t> slashPositions; //positions of '\\' or '/' characters
#endif
#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
	char slash = '\\';
#else
	char slash = '/';
#endif
	

	len = strlen(name);
	if (parent != "")
	{
		if (parent[parent.size()-1] != '\\' && parent[parent.size()-1] != '/')
		{
			len += parent.size() + 1;
			needSlash = true;
		}
		else
			len += parent.size();
	}

	fixedPath.resize(len);
	//copy the parent path to the beginning
	for (size_t i = 0; i < parent.size(); ++i)
	{
		fixedPath[i] = parent[i];

		if (parent[i] == '\\' || parent[i] == '/')
		{
#if REMOVE_DOUBLE_DOT
			slashPositions.PushBack(i);
#endif
			fixedPath[i] = slash;
		}
	}
	fixedPathPos = parent.size();
	if (needSlash)
	{
		fixedPath[fixedPathPos] = slash;
#if REMOVE_DOUBLE_DOT
		slashPositions.PushBack(fixedPathPos);
#endif
		fixedPathPos += 1;
	}

	/*-----------concatenate the intended path----------*/

	while (name[intendedPathPos] != '\0')
	{
		if (name[intendedPathPos] == '/' || name[intendedPathPos] == '\\')
		{
#if REMOVE_DOUBLE_DOT
			if (dotCount == 2)//this is "\..\" pop the last folder path
			{
				if (slashPositions.GetSize() >= 2)
				{
					//pop the last folder
					slashPositions.PopBack();
					fixedPathPos = slashPositions.GetBack();
				}
				else if (slashPositions.GetSize() > 0)
				{
					//pop the last folder
					slashPositions.PopBack();
					fixedPathPos = (size_t)-1;
				}
			}
			else
#endif//#if REMOVE_DOUBLE_DOT
			{
				fixedPath[fixedPathPos] = slash;
#if REMOVE_DOUBLE_DOT
				slashPositions.PushBack(fixedPathPos);//push the slash position to positions list
#endif
			}

			dotCount = 0;//reset dot count
		}
		else
		{
			fixedPath[fixedPathPos] = name[intendedPathPos]; 
			if (name[intendedPathPos] == '.')
			{
				dotCount++;
			}
			else
			{
				dotCount = 0;//reset dot count
			}
		}

		++intendedPathPos;
		if (fixedPathPos == (size_t)-1)
			fixedPathPos = 0;
		else
			++fixedPathPos;
	}

	fixedPath[fixedPathPos] = '\0';
	fixedPath.resize(fixedPathPos);
}