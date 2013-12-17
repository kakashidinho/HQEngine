/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
#include <string.h>//for memset

/*--------HQBSPTree-------*/
HQBSPTree::HQBSPTree()
{
	this->numPoly = 0;
	this->rootNode = HQ_NEW HQBSPTreeNode();
}
HQBSPTree::HQBSPTree(HQBSPTreeNode *rootNode)
{
	this->numPoly = 0;
	this->rootNode = rootNode;
}
HQBSPTree::~HQBSPTree()
{
	SafeDelete(rootNode);
}

void HQBSPTree::BuildTree(const HQPolygon3D *headNode , hq_uint32 numPoly)
{
	this->numPoly = 0;
	this->rootNode->Clear();
	if(headNode == NULL || numPoly == 0)
		return;

	for(hq_uint32 i = 0; i < numPoly ; ++i)
	{
		rootNode->AddPolygon(&headNode[i]);
	}

	rootNode->tree = this;
	rootNode->CreateChilds();
}

/*---------HQBSPTreeNode-----*/
HQBSPTreeNode::HQBSPTreeNode() : HQPolygonList()
{
	this->tree = NULL;
	this->parentNode = NULL;
	this->frontNode = NULL;
	this->backNode = NULL;

	memset(&boundingBox,0,sizeof(HQAABB));
}

HQBSPTreeNode::HQBSPTreeNode(const HQBSPTreeNode &src)
{
}

void HQBSPTreeNode::Clear()
{
	SafeDelete(frontNode);
	SafeDelete(backNode);
	ClearPolyList();
}

void HQBSPTreeNode::CalBoundingBox()
{
	if(headNode == NULL)
		return;
	this->boundingBox = headNode->GetPolygon()->GetBoundingBox();
	HQPolyListNode *pNode = headNode->pNext;
	while(pNode != NULL)
	{
		const HQAABB & aabb = pNode->GetPolygon()->GetBoundingBox();
		
		if(aabb.vMin.x < this->boundingBox.vMin.x)
			this->boundingBox.vMin.x = aabb.vMin.x;
		if(aabb.vMin.y < this->boundingBox.vMin.y)
			this->boundingBox.vMin.y = aabb.vMin.y;
		if(aabb.vMin.z < this->boundingBox.vMin.z)
			this->boundingBox.vMin.z = aabb.vMin.z;

		if(aabb.vMax.x > this->boundingBox.vMax.x)
			this->boundingBox.vMax.x = aabb.vMax.x;
		if(aabb.vMax.y > this->boundingBox.vMax.y)
			this->boundingBox.vMax.y = aabb.vMax.y;
		if(aabb.vMax.z > this->boundingBox.vMax.z)
			this->boundingBox.vMax.z = aabb.vMax.z;

		pNode = pNode->pNext;
	}
}

void HQBSPTreeNode::CreateChilds()
{
	this->CalBoundingBox();
	if(!FindBestSplitter())//this is leaf node, polygons in list form a convex group
	{
		this->tree->numPoly += this->numPoly;
		return;
	}
	

	frontNode = HQ_NEW HQBSPTreeNode();
	backNode = HQ_NEW HQBSPTreeNode();
	
	frontNode->tree = backNode->tree = this->tree;
	frontNode->parentNode = backNode->parentNode = this;

	HQPolyListNode *pNode = headNode;
	

	//chia danh sách polygon làm 2 nhánh sau và trước mặt splitter
	while(pNode != NULL)
	{
		const HQPolygon3D* poly = pNode->GetPolygon();

		HQPlane::Side side = splitter.CheckSide(*poly);

		switch(side)
		{
		case HQPlane::FRONT_PLANE:
			frontNode->AddPolygon(poly);
			break;
		case HQPlane::BACK_PLANE:
			backNode->AddPolygon(poly);
			break;
		case HQPlane::IN_PLANE://nằm trên mặt splitter
			{
				hq_float32 dot = splitter.N * poly->GetPlane().N;
				if(dot >= 0.0f)//pháp vector polygon cùng hướng với pháp vector của splitter
					frontNode->AddPolygon(poly);
				else
					backNode->AddPolygon(poly);
			}
			break;
		case HQPlane::BOTH_SIDE://cắt splitter
			{
#ifdef HQ_EXPLICIT_ALIGN
				HQ_DECL_STACK_VAR_ARRAY(HQPolygon3D, polys, 2);
				HQPolygon3D &front = polys[0];
				HQPolygon3D &back = polys[1];
#else
				HQPolygon3D front,back;
#endif
				poly->Clip(splitter , &front, &back);//chia polygon làm 2 polygon phía sau và trước splitter
				backNode->AddPolygon(&back);
				frontNode->AddPolygon(&front);
			}
			break;
		}

		pNode = pNode->pNext;
	}
	
	this->ClearPolyList();

	frontNode->CreateChilds();
	backNode->CreateChilds();
}

bool HQBSPTreeNode::FindBestSplitter()
{
	hq_uint32 minScore = 0xffffffff;
	hq_uint32 nFront ;// số polygon phía trước mặt đang xét
	hq_uint32 nBack ;//số polygon phía sau mặt đang xét
	hq_uint32 nSplit;//số polygon cắt mặt đang xét
	
	HQPolyListNode* pBestSplitter = NULL;
	HQPolyListNode *pCurrentNode = headNode;
	while(pCurrentNode != NULL)
	{
		nFront = nBack = nSplit = 0;
		if(pCurrentNode->GetPolygon()->GetFlag() != 1)//chưa chọn làm splitter trước đó
		{
			const HQPlane &plane = pCurrentNode->GetPolygon()->GetPlane();
			//kiểm tra các polygon còn lại nằm phía nào so với mặt phẳng chứa polygon đang xét
			HQPolyListNode *pOtherNode = headNode;
			while(pOtherNode != NULL)
			{
				if (pOtherNode != pCurrentNode)
				{
					const HQPolygon3D *poly = pOtherNode->GetPolygon();
					HQPlane::Side side = plane.CheckSide(*poly);//kiểm tra polygon này nằm phía nào

					switch(side)
					{
					case HQPlane::FRONT_PLANE:
						nFront ++;
						break;
					case HQPlane::BACK_PLANE:
						nBack ++;
						break;
					case HQPlane::BOTH_SIDE:
						nSplit ++;
						break;
					}

					hq_uint32 score = (nFront > nBack)?(nFront - nBack): (nBack - nFront) + 3 * nSplit;
					if(score < minScore)
					{
						if((nFront > 0 && nBack > 0) || nSplit > 0)
						{
							pBestSplitter = pCurrentNode;
							minScore = score;
						}
					}
				}

				pOtherNode = pOtherNode->pNext;
			}//other node
		}
		pCurrentNode = pCurrentNode->pNext;
	}//current node

	if(pBestSplitter == NULL)
		return false;
	
	pBestSplitter->GetPolygon()->SetFlag(1);//đánh dấu polygon này đã chọn làm splitter
	this->splitter = pBestSplitter->GetPolygon()->GetPlane();

	return true;
}

bool HQBSPTreeNode::Collision(const HQRay3D & ray , hq_float32 *pT , hq_float32 maxT)//<pT> lưu thời điểm cắt
{
	if(IsLeaf())
	{
		HQPolyListNode *pNode = headNode;
		while(pNode != NULL)
		{
			//kiểm tra cắt với từng polygon
			const HQPolygon3D* poly = pNode->GetPolygon();

			if(ray.Intersect(*poly , pT , maxT))
				return true;

			pNode = pNode->pNext;
		}
		return false;
	}
	
	HQPlane::Side side = splitter.CheckSide(ray.O);
	hq_float32 t;
	if (ray.Intersect(splitter , &t ,maxT , false))//tia cắt mặt splitter
	{
		HQ_DECL_STACK_VAR(HQRay3D, clippedRay);//tia kết quả từ việc cắt tia <ray> bằng mặt splitter
		//clippedRay.O = ray.O + t * ray.D;
		HQVector4Mul(t, &ray.D, &clippedRay.O);
		clippedRay.O += ray.O;

		clippedRay.D = ray.D;

		if(side == HQPlane::BACK_PLANE)//gốc của tia ở phía mặt sau của splitter
			return ( backNode->Collision(ray , pT, t) || //kiểm tra nhánh sau splitter trước
				frontNode->Collision(clippedRay , pT , maxT - t));//kiểm tra nhánh trước splitter bằng tia kết quả từ việc cắt tia <ray>
		else
			return ( frontNode->Collision(ray , pT, t) || //kiểm tra nhánh trước splitter trước
				backNode->Collision(clippedRay , pT , maxT - t));
	}
	else//tia song song mặt splitter
	{
		if(side == HQPlane::BACK_PLANE)//gốc tia thuộc vùng mặt sau 
			return  backNode->Collision(ray , pT, maxT);
		else
			return frontNode->Collision(ray , pT, maxT);
	}
}

void HQBSPTreeNode::TraverseFtoB(const HQVector4& eye , const HQPlane * frustum , HQPolygonList &listOut)//traverse front to back.<listOut> will store polygons in front to back order. 
{
	if (this->boundingBox.Cull(frustum , 6) == HQ_CULLED)//nhánh này hoàn toàn nằm ngoài thể tích nhìn
		return;

	if(IsLeaf())
	{
		HQPolyListNode *pNode = headNode;
		while(pNode != NULL)
		{
			const HQPolygon3D* poly = pNode->GetPolygon();

			listOut.AddPolygon(poly);

			pNode = pNode->pNext;
		}
	}
	else
	{
		HQPlane::Side side = splitter.CheckSide(eye);
		if(side == HQPlane::BACK_PLANE)
		{
			backNode->TraverseFtoB(eye , frustum , listOut);
			frontNode->TraverseFtoB(eye , frustum , listOut);
		}
		else
		{
			frontNode->TraverseFtoB(eye , frustum , listOut);
			backNode->TraverseFtoB(eye , frustum , listOut);
		}
	}
}
void HQBSPTreeNode::TraverseBtoF(const HQVector4& eye , const HQPlane * frustum , HQPolygonList &listOut)//traverse front to back.<listOut> will store polygons in back to front order. 
{
	if (this->boundingBox.Cull(frustum , 6) == HQ_CULLED)//nhánh này hoàn toàn nằm ngoài thể tích nhìn
		return;
	if(IsLeaf())
	{
		HQPolyListNode *pNode = headNode;
		while(pNode != NULL)
		{
			const HQPolygon3D* poly = pNode->GetPolygon();

			listOut.AddPolygon(poly);

			pNode = pNode->pNext;
		}
	}
	else
	{
		HQPlane::Side side = splitter.CheckSide(eye);
		if(side == HQPlane::BACK_PLANE)
		{
			frontNode->TraverseBtoF(eye , frustum , listOut);
			backNode->TraverseBtoF(eye , frustum , listOut);
		}
		else
		{
			backNode->TraverseBtoF(eye , frustum , listOut);
			frontNode->TraverseBtoF(eye , frustum , listOut);
		}
	}
}
