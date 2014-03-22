/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_RES_PARSER_H
#define HQ_ENGINE_RES_PARSER_H

#include "HQEngineCommonInternal.h"

#include <string>

class HQEngineCommonResParserNode {
public:
	struct ValueType{

		enum TypeEnum {
			INTEGER_TYPE,
			FLOAT_TYPE,
			STRING_TYPE,
			INVALID_VALUE,
		};

		TypeEnum type;

		const char * GetAsString () const { return type == STRING_TYPE? string: NULL; }
		const char * GetAsStringEmptyIfNone () const;///return empty string if not found
		const hqint32 * GetAsIntPt () const { return type == INTEGER_TYPE? &ivalue: NULL; }
		const hqfloat64 * GetAsFloatPt () const { return type == FLOAT_TYPE? &fvalue: NULL; }

		union{
			hqint32 ivalue;
			hqfloat64 fvalue;
			const char * string;///Note: value is stored by reference. Be careful don't delete string if it is still needed
		};
	};

	HQEngineCommonResParserNode(const char *typeName, int sourceLine);
	HQEngineCommonResParserNode(int sourceLine);
	HQEngineCommonResParserNode();
	virtual ~HQEngineCommonResParserNode();

	static void DeleteTree(HQEngineCommonResParserNode * root);
	
	int GetSourceLine() const {return m_sourceLine;}
	void SetType(const char* typeName) {m_typeName = typeName;}
	void SetSourceLine(int line) {m_sourceLine = line;}
	const char * GetType() const {return m_typeName.c_str();};
	const ValueType& GetAttribute(const char *attributeName) const;//returned value has INVALID_VALUE if not found
	const ValueType* GetAttributePtr(const char *attributeName) const;
	const char* GetStrAttribute(const char *attributeName) const;
	const char* GetStrAttributeEmptyIfNone(const char *attributeName) const; ///return empty string if not found
	const hqint32* GetIntAttributePtr(const char *attributeName) const;
	const hqfloat64* GetFloatAttributePtr(const char *attributeName) const;
	const HQEngineCommonResParserNode* GetNextSibling(const char *siblingTypeName = NULL) const ;
	const HQEngineCommonResParserNode* GetFirstChild(const char *childName = NULL) const ;

	void AddChild(HQEngineCommonResParserNode* child);
	void SetAttribute(const char *name, ValueType value );
	void SetAttribute(const char *name, hqint32 value );
	void SetAttribute(const char *name, hqfloat64 value );
	void SetAttribute(const char *name, const char* value );///Note: value is stored by reference. Be careful don't delete value if it is still needed

private:
	int m_sourceLine;
	std::string m_typeName;

	HQEngineStringHashTable<ValueType> m_attributes; 
	HQEngineCommonResParserNode* m_nextSibling;
	HQEngineCommonResParserNode* m_firstChild;
};

typedef HQEngineCommonResParserNode HQEngineResParserNode;
typedef HQEngineCommonResParserNode HQEngineEffectParserNode;

#endif