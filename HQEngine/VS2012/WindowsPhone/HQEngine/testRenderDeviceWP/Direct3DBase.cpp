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

	m_device->OnWindowSizeChanged(m_renderTargetSize.Width, m_renderTargetSize.Height);
}

// This method is called in the event handler for the SizeChanged event.
void Direct3DBase::UpdateForWindowSizeChange()
{
	if (m_window->Bounds.Width  != m_windowBounds.Width ||
		m_window->Bounds.Height != m_windowBounds.Height )
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

void Direct3DBase::ReleaseResourcesForSuspending()
{
	// Phone applications operate in a memory-constrained environment, so when entering
	// the background it is a good idea to free memory-intensive objects that will be
	// easy to restore upon reactivation. The swapchain and backbuffer are good candidates
	// here, as they consume a large amount of memory and can be reinitialized quickly.
	
}

