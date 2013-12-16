#ifndef _VERTEX_STREAM_MAN_
#define _VERTEX_STREAM_MAN_

#include "d3d11.h"
#include "../HQVertexStreamManager.h"
#include "HQLoggableObject.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "HQShaderD3D11.h"
#include "../HQItemManager.h"

#define MAX_VERTEX_ATTRIBS 16

struct HQBufferD3D11
{
	HQBufferD3D11(bool isDynamic , hq_uint32 size)
	{
		this->pD3DBuffer = NULL;
		this->isDynamic = isDynamic;
		this->size = size;
	}
	virtual ~HQBufferD3D11()
	{
		SafeRelease(this->pD3DBuffer);
	}
	
	ID3D11Buffer *pD3DBuffer;
	bool isDynamic;
	hq_uint32 size;
};



struct HQIndexBufferD3D11 : public HQBufferD3D11
{
	HQIndexBufferD3D11(bool dynamic   , hq_uint32 size, HQIndexDataType dataType) 
		: HQBufferD3D11(dynamic , size)
	{
		switch (dataType)
		{
		case HQ_IDT_UINT:
			this->d3dDataType = DXGI_FORMAT_R32_UINT;
			break;
		default:
			this->d3dDataType = DXGI_FORMAT_R16_UINT;
		}
	}

	DXGI_FORMAT d3dDataType;
};


struct HQVertexInputLayoutD3D11
{
	HQVertexInputLayoutD3D11() {
		this->pD3DLayout = NULL;
	}
	~HQVertexInputLayoutD3D11() {
		SafeRelease(pD3DLayout);
	}

	ID3D11InputLayout * pD3DLayout;
};

struct HQVertexStreamD3D11
{
	HQVertexStreamD3D11() {stride = 0;}
	HQSharedPtr<HQBufferD3D11> vertexBuffer;
	hq_uint32 stride;
};

class HQVertexStreamManagerD3D11: public HQVertexStreamManager , public HQLoggableObject
{
private:
	ID3D11Device * pD3DDevice;
	ID3D11DeviceContext *pD3DContext;
	ID3D11Buffer * pCleaVBuffer;//vertex buffer for clearing viewport
	ID3D11InputLayout *pClearInputLayout;//input layout for clearing viewport

	HQShaderManagerD3D11 *pShaderMan;

	HQItemManager<HQBufferD3D11> vertexBuffers;
	HQItemManager<HQIndexBufferD3D11> indexBuffers;
	HQItemManager<HQVertexInputLayoutD3D11> inputLayouts;

	HQSharedPtr<HQIndexBufferD3D11> activeIndexBuffer;
	HQSharedPtr<HQVertexInputLayoutD3D11> activeInputLayout;

	HQVertexStreamD3D11 streams[MAX_VERTEX_ATTRIBS];
	

	void ConvertToElementDesc(const HQVertexAttribDesc &vAttribDesc ,D3D11_INPUT_ELEMENT_DESC &vElement);

public:
	HQVertexStreamManagerD3D11(ID3D11Device* pD3DDevice , 
								ID3D11DeviceContext *pD3DContext, 
								HQShaderManagerD3D11 *pShaderMan,
								HQLogStream* logFileStream , bool flushLog);
	~HQVertexStreamManagerD3D11() ;
	

	HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,hq_uint32 *pVertexBufferID);
	HQReturnVal CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , hq_uint32 *pIndexBufferID);

	HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												hq_uint32 vertexShaderID , 
												hq_uint32 *pInputLayoutID);

	HQReturnVal SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) ;

	HQReturnVal SetIndexBuffer(hq_uint32 indexBufferID );

	HQReturnVal SetVertexInputLayout(hq_uint32 inputLayoutID) ;
	
	HQReturnVal MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) ;
	HQReturnVal UnmapVertexBuffer(hq_uint32 vertexBufferID) ;
	HQReturnVal MapIndexBuffer( HQMapType mapType , void **ppData) ;
	HQReturnVal UnmapIndexBuffer() ;
	
	HQReturnVal UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData);
	HQReturnVal UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData);
	HQReturnVal UpdateIndexBuffer(hq_uint32 indexBufferID, hq_uint32 offset , hq_uint32 size , const void * pData);

	HQReturnVal RemoveVertexBuffer(hq_uint32 vertexBufferID) ;
	HQReturnVal RemoveIndexBuffer(hq_uint32 indexBufferID) ;
	HQReturnVal RemoveVertexInputLayout(hq_uint32 inputLayoutID) ;
	void RemoveAllVertexBuffer() ;
	void RemoveAllIndexBuffer() ;
	void RemoveAllVertexInputLayout() ;
	
#if HQ_D3D_CLEAR_VP_USE_GS
	void ChangeClearVBuffer(HQColorui color , hq_float32 depth);
#endif
	void BeginClearViewport();
	void EndClearViewport();
};


#endif