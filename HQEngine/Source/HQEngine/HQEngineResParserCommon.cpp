/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"

#include  "HQEngineResParserCommon.h"

const char * HQEngineEffectParserNode::ValueType::GetAsStringEmptyIfNone () const
{
	const char * str = GetAsString();
	if (str == NULL)
	{
		static const char empty[] = "";
		return empty;
	}
	return str;
}

/*--------------------------------------*/
HQEngineCommonResParserNode::HQEngineCommonResParserNode(const char *typeName, int sourceLine)
:m_nextSibling(NULL), m_firstChild(NULL)
{
	m_typeName = typeName;
	m_sourceLine = sourceLine;
}

HQEngineCommonResParserNode::HQEngineCommonResParserNode( int sourceLine)
:m_nextSibling(NULL), m_firstChild(NULL)
{
	m_sourceLine = sourceLine;
}

HQEngineCommonResParserNode::HQEngineCommonResParserNode()
:m_nextSibling(NULL), m_firstChild(NULL)
{
	m_sourceLine = -1;
}



HQEngineCommonResParserNode::~HQEngineCommonResParserNode()
{
}

/*static*/ void HQEngineCommonResParserNode::DeleteTree(HQEngineCommonResParserNode * node)
{
	if (node == NULL)
		return;
	DeleteTree(node->m_firstChild);
	DeleteTree(node->m_nextSibling);

	delete node;
}


const HQEngineCommonResParserNode* HQEngineCommonResParserNode::GetNextSibling(const char *siblingTypeName) const 
{
	if (siblingTypeName == NULL)
		return m_nextSibling;
	HQEngineCommonResParserNode* sibling = m_nextSibling;
	while (sibling != NULL)
	{
		if (!strcmp(sibling->GetType(), siblingTypeName))
			return sibling;
		sibling = sibling->m_nextSibling;
	}

	return NULL;
}

const HQEngineCommonResParserNode* HQEngineCommonResParserNode::GetFirstChild(const char *childName) const {
	if (childName == NULL)
		return m_firstChild;

	HQEngineCommonResParserNode* child = m_firstChild;
	while (child != NULL)
	{
		if (!strcmp(child->GetType(), childName))
			return child;
		child = child->m_nextSibling;
	}

	return NULL;
}

void HQEngineCommonResParserNode::AddChild(HQEngineCommonResParserNode* child){
	HQEngineCommonResParserNode* prev_child = m_firstChild;
	while (prev_child != NULL && prev_child->m_nextSibling != NULL)
	{
		prev_child = prev_child->m_nextSibling;
	}

	if (prev_child == NULL)
		m_firstChild = child;//this is the first child
	else
		prev_child->m_nextSibling = child;

	child->m_nextSibling = NULL;
}

void HQEngineCommonResParserNode::SetAttribute(const char *name, HQEngineCommonResParserNode::ValueType value ){
	ValueType * old_val = m_attributes.GetItemPointer(name);

	if (old_val == NULL)
		m_attributes.Add(name, value);
	else
		*old_val = value;
}

void HQEngineCommonResParserNode::SetAttribute(const char *name, hqint32 value ){
	HQEngineCommonResParserNode::ValueType unionVal;

	unionVal.type = HQEngineCommonResParserNode::ValueType::INTEGER_TYPE;
	unionVal.ivalue = value;
	
	this->SetAttribute(name, unionVal);
}

void HQEngineCommonResParserNode::SetAttribute(const char *name, hqfloat64 value ){
	HQEngineCommonResParserNode::ValueType unionVal;

	unionVal.type = HQEngineCommonResParserNode::ValueType::FLOAT_TYPE;
	unionVal.fvalue = value;
	
	this->SetAttribute(name, unionVal);
}

void HQEngineCommonResParserNode::SetAttribute(const char *name, const char * value ){
	HQEngineCommonResParserNode::ValueType unionVal;

	unionVal.type = HQEngineCommonResParserNode::ValueType::STRING_TYPE;
	unionVal.string = value;
	
	this->SetAttribute(name, unionVal);
}

const HQEngineCommonResParserNode::ValueType* HQEngineCommonResParserNode::GetAttributePtr(const char* attributeName) const
{
	return m_attributes.GetItemPointer(attributeName);
}

const HQEngineCommonResParserNode::ValueType& HQEngineCommonResParserNode::GetAttribute(const char *attributeName) const
{
	bool found;
	const ValueType& value = m_attributes.GetItem(attributeName, found);

	if (!found)
	{
		static ValueType invalid;
		invalid.type = ValueType::INVALID_VALUE;
		return invalid;
	}
	return value;
}

const char* HQEngineCommonResParserNode::GetStrAttribute(const char *attributeName) const
{
	const HQEngineCommonResParserNode::ValueType* value = this->GetAttributePtr(attributeName);
	if (value == NULL || value->type != HQEngineCommonResParserNode::ValueType::STRING_TYPE)
		return NULL;
	return value->string;
}

const char* HQEngineCommonResParserNode::GetStrAttributeEmptyIfNone(const char *attributeName) const
{
	const HQEngineCommonResParserNode::ValueType& value = this->GetAttribute(attributeName);
	return value.GetAsStringEmptyIfNone();
}

const hqint32* HQEngineCommonResParserNode::GetIntAttributePtr(const char *attributeName) const
{
	const HQEngineCommonResParserNode::ValueType* value = this->GetAttributePtr(attributeName);
	if (value == NULL || value->type != HQEngineCommonResParserNode::ValueType::INTEGER_TYPE)
		return NULL;
	return &value->ivalue;
}

const hqfloat64* HQEngineCommonResParserNode::GetFloatAttributePtr(const char *attributeName) const
{
	const HQEngineCommonResParserNode::ValueType* value = this->GetAttributePtr(attributeName);
	if (value == NULL || value->type != HQEngineCommonResParserNode::ValueType::FLOAT_TYPE)
		return NULL;
	return &value->fvalue;
}