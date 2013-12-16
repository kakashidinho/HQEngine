#pragma once

#include "Direct3DBase.h"

struct ModelViewProjectionConstantBuffer
{
	DirectX::XMFLOAT4X4 model;
	DirectX::XMFLOAT4X4 view;
	DirectX::XMFLOAT4X4 projection;
};

struct VertexPositionColor
{
	DirectX::XMFLOAT3 pos;
	DirectX::XMFLOAT3 color;
};

// This class renders a simple spinning cube.
ref class CubeRenderer sealed : public Direct3DBase
{
public:
	CubeRenderer();

	// Direct3DBase methods.
	virtual void CreateDeviceResources() override;
	virtual void CreateWindowSizeDependentResources() override;
	virtual void Render() override;
	
	// Method for updating time-dependent objects.
	void Update(float timeTotal, float timeDelta);

private:

	void CreateDeviceResourcesOnOtherThread();

	bool m_loadingComplete;

	hquint32 m_inputLayout;
	hquint32 m_vertexBuffer;
	hquint32 m_indexBuffer;
	hquint32 m_vertexShader;
	hquint32 m_pixelShader;
	hquint32 m_shaderProgram;
	hquint32 m_constantBuffer;

	uint32 m_indexCount;
	uint32 m_vertexCount;
	ModelViewProjectionConstantBuffer m_constantBufferData;
};
