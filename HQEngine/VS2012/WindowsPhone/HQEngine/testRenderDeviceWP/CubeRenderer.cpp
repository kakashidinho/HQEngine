#include "pch.h"
#include "CubeRenderer.h"
#include "HQUtil.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Windows::Foundation;
using namespace Windows::UI::Core;

CubeRenderer::CubeRenderer() :
	m_loadingComplete(false),
	m_indexCount(0)
{
}

void CubeRenderer::CreateDeviceResourcesOnOtherThread()
{
	HQReturnVal re;
	//create shaders
	re = m_device->GetShaderManager()->CreateShaderFromByteCodeFile(HQ_VERTEX_SHADER, "SimpleVertexShader.cso", &m_vertexShader);
	re = m_device->GetShaderManager()->CreateShaderFromByteCodeFile(HQ_PIXEL_SHADER, "SimplePixelShader.cso", &m_pixelShader);

	re = m_device->GetShaderManager()->CreateProgram(m_vertexShader, m_pixelShader, NOT_USE_GSHADER, NULL, &m_shaderProgram);

	//create input layout
	HQVertexAttribDescArray<2> attribDesc;

	attribDesc.SetPosition(0, 0, 0, HQ_VADT_FLOAT3);
	attribDesc.SetColor(1, 0, 12, HQ_VADT_FLOAT3);

	re = m_device->GetVertexStreamManager()->CreateVertexInputLayout(attribDesc, attribDesc.GetNumAttribs(), m_vertexShader, &m_inputLayout);

	//create uniform buffer
	re = m_device->GetShaderManager()->CreateUniformBuffer(sizeof(ModelViewProjectionConstantBuffer), NULL, false, &m_constantBuffer);
	
	//create vertex buffer
	VertexPositionColor cubeVertices[] = 
	{
		{XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0.0f, 0.0f, 0.0f)},
		{XMFLOAT3(-0.5f, -0.5f,  0.5f), XMFLOAT3(0.0f, 0.0f, 1.0f)},
		{XMFLOAT3(-0.5f,  0.5f, -0.5f), XMFLOAT3(0.0f, 1.0f, 0.0f)},
		{XMFLOAT3(-0.5f,  0.5f,  0.5f), XMFLOAT3(0.0f, 1.0f, 1.0f)},
		{XMFLOAT3( 0.5f, -0.5f, -0.5f), XMFLOAT3(1.0f, 0.0f, 0.0f)},
		{XMFLOAT3( 0.5f, -0.5f,  0.5f), XMFLOAT3(1.0f, 0.0f, 1.0f)},
		{XMFLOAT3( 0.5f,  0.5f, -0.5f), XMFLOAT3(1.0f, 1.0f, 0.0f)},
		{XMFLOAT3( 0.5f,  0.5f,  0.5f), XMFLOAT3(1.0f, 1.0f, 1.0f)},
	};

	m_vertexCount = ARRAYSIZE(cubeVertices);

	re = m_device->GetVertexStreamManager()->CreateVertexBuffer(
		cubeVertices,
		sizeof(cubeVertices),
		false,
		false,
		&m_vertexBuffer);

	//create index buffer

	unsigned short cubeIndices[] = 
	{
		0,2,1, // -x
		1,2,3,

		4,5,6, // +x
		5,7,6,

		0,1,5, // -y
		0,5,4,

		2,6,7, // +y
		2,7,3,

		0,4,6, // -z
		0,6,2,

		1,3,7, // +z
		1,7,5,
	};

	m_indexCount = ARRAYSIZE(cubeIndices);

	m_device->GetVertexStreamManager()->CreateIndexBuffer(
		cubeIndices,
		sizeof(cubeIndices),
		false,
		HQ_IDT_USHORT,
		&m_indexBuffer);
}

void CubeRenderer::CreateDeviceResources()
{
	Direct3DBase::CreateDeviceResources();


	auto task = Concurrency::create_task([this] () {
		
		this->CreateDeviceResourcesOnOtherThread();
		
	});

	task.then([this] () {
		m_loadingComplete = true;
	});
}

void CubeRenderer::CreateWindowSizeDependentResources()
{
	Direct3DBase::CreateWindowSizeDependentResources();

	float aspectRatio = m_windowBounds.Width / m_windowBounds.Height;
	float fovAngleY = 70.0f * XM_PI / 180.0f;
}

void CubeRenderer::Update(float timeTotal, float timeDelta)
{
	(void) timeDelta; // Unused parameter.

	XMVECTOR eye = XMVectorSet(0.0f, 0.7f, 1.5f, 0.0f);
	XMVECTOR at = XMVectorSet(0.0f, -0.1f, 0.0f, 0.0f);
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	XMStoreFloat4x4(&m_constantBufferData.view, XMMatrixTranspose(XMMatrixLookAtRH(eye, at, up)));
	XMStoreFloat4x4(&m_constantBufferData.model, XMMatrixTranspose(XMMatrixRotationY(timeTotal * XM_PIDIV4)));
}

void CubeRenderer::Render()
{
	HQColor midnightBlue = { 0.098f, 0.098f, (rand() % 1000) / 1000.0f, 1.000f };
	m_device->SetClearColor(midnightBlue);
	m_device->SetClearDepthVal(1.0f);

	m_device->Clear(HQ_TRUE, HQ_TRUE, HQ_FALSE, HQ_FALSE);

	// Only draw the cube once it is loaded (loading is asynchronous).
	if (!m_loadingComplete)
	{
		m_device->DisplayBackBuffer();
		return;
	}
	

	m_device->BeginRender(HQ_FALSE, HQ_FALSE, HQ_FALSE, HQ_TRUE);

	m_device->GetShaderManager()->UpdateUniformBuffer(m_constantBuffer, &m_constantBufferData);


	UINT stride = sizeof(VertexPositionColor);
	UINT offset = 0;

	m_device->GetVertexStreamManager()->SetVertexBuffer(m_vertexBuffer, 0, stride);
	m_device->GetVertexStreamManager()->SetIndexBuffer(m_indexBuffer);

	m_device->SetPrimitiveMode(HQ_PRI_TRIANGLES);

	m_device->GetVertexStreamManager()->SetVertexInputLayout(m_inputLayout);

	m_device->GetShaderManager()->ActiveProgram(m_shaderProgram);

	m_device->GetShaderManager()->SetUniformBuffer(HQ_VERTEX_SHADER | 0, m_constantBuffer);

	m_device->DrawIndexed(m_vertexCount, m_indexCount, 0);

	m_device->EndRender();
}
