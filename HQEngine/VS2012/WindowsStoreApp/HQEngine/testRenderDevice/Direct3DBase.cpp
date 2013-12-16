#include "pch.h"
#include "Direct3DBase.h"
#include "HQUtil.h"

#include "HQEngine\winstore\HQWinStoreUtil.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Windows::UI::Core;
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;

// Constructor.
Direct3DBase::Direct3DBase()
	: m_renderer(true)
{
	m_renderer.CreateD3DDevice11(true);
	m_device = m_renderer.GetDevice();
}

// Initialize the Direct3D resources required to run.
void Direct3DBase::Initialize(CoreWindow^ window)
{
	m_window = window;
	
	CreateDeviceResources();
	CreateWindowSizeDependentResources();
}

// Recreate all device resources and set them back to the current state.
void Direct3DBase::HandleDeviceLost()
{
	// TO DO
}

// These are the resources that depend on the device.
void Direct3DBase::CreateDeviceResources()
{ 
	auto logFile = HQCreateFileLogStream("log.txt");

	m_device->Init(m_window, "setting.txt", logFile);
	m_device->GetStateManager()->SetFaceCulling(HQ_CULL_CCW);
}

// Allocate all memory resources that change on a window SizeChanged event.
void Direct3DBase::CreateWindowSizeDependentResources()
{ 
	// Store the window bounds so the next time we get a SizeChanged event we can
	// avoid rebuilding everything if the size is identical.
	m_windowBounds = m_window->Bounds;

	// Calculate the necessary swap chain and render target size in pixels.
	float windowWidth = m_device->GetWidth();
	float windowHeight = m_device->GetHeight();

	// The width and height of the swap chain must be based on the window's
	// landscape-oriented width and height. If the window is in a portrait
	// orientation, the dimensions must be reversed.
	m_orientation = DisplayProperties::CurrentOrientation;
	bool swapDimensions =
		m_orientation == DisplayOrientations::Portrait ||
		m_orientation == DisplayOrientations::PortraitFlipped;
	m_renderTargetSize.Width = swapDimensions ? windowHeight : windowWidth;
	m_renderTargetSize.Height = swapDimensions ? windowWidth : windowHeight;
	
	// Set the proper orientation for the swap chain, and generate the
	// 3D matrix transformation for rendering to the rotated swap chain.
	DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_UNSPECIFIED;
	switch (m_orientation)
	{
		case DisplayOrientations::Landscape:
			rotation = DXGI_MODE_ROTATION_IDENTITY;
			m_orientationTransform3D = XMFLOAT4X4( // 0-degree Z-rotation
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f
				);
			break;

		case DisplayOrientations::Portrait:
			rotation = DXGI_MODE_ROTATION_ROTATE270;
			m_orientationTransform3D = XMFLOAT4X4( // 90-degree Z-rotation
				0.0f, 1.0f, 0.0f, 0.0f,
				-1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f
				);
			break;

		case DisplayOrientations::LandscapeFlipped:
			rotation = DXGI_MODE_ROTATION_ROTATE180;
			m_orientationTransform3D = XMFLOAT4X4( // 180-degree Z-rotation
				-1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, -1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f
				);
			break;

		case DisplayOrientations::PortraitFlipped:
			rotation = DXGI_MODE_ROTATION_ROTATE90;
			m_orientationTransform3D = XMFLOAT4X4( // 270-degree Z-rotation
				0.0f, -1.0f, 0.0f, 0.0f,
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f
				);
			break;

		default:
			throw ref new Platform::FailureException();
	}

	m_device->OnWindowSizeChanged(m_renderTargetSize.Width, m_renderTargetSize.Height);
}

// This method is called in the event handler for the SizeChanged event.
void Direct3DBase::UpdateForWindowSizeChange()
{
	if (m_window->Bounds.Width  != m_windowBounds.Width ||
		m_window->Bounds.Height != m_windowBounds.Height ||
		m_orientation != DisplayProperties::CurrentOrientation)
	{
		CreateWindowSizeDependentResources();
	}
}

// Method to deliver the final image to the display.
void Direct3DBase::Present()
{
	m_device->DisplayBackBuffer();
}

// Method to convert a length in device-independent pixels (DIPs) to a length in physical pixels.
float Direct3DBase::ConvertDipsToPixels(float dips)
{
	static const float dipsPerInch = 96.0f;
	return floor(dips * DisplayProperties::LogicalDpi / dipsPerInch + 0.5f); // Round to nearest integer.
}
