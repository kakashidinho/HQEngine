#pragma once

#include "DirectXHelper.h"

#include "HQRenderer.h"

// Helper class that initializes DirectX APIs for 3D rendering.
ref class Direct3DBase abstract
{
internal:
	Direct3DBase();

public:
	virtual void Initialize(Windows::UI::Core::CoreWindow^ window);
	virtual void HandleDeviceLost();
	virtual void CreateDeviceResources();
	virtual void CreateWindowSizeDependentResources();
	virtual void UpdateForWindowSizeChange();
	virtual void Render() = 0;
	virtual void Present();
	virtual float ConvertDipsToPixels(float dips);

protected private:

	// Cached renderer properties.
	D3D_FEATURE_LEVEL m_featureLevel;
	Windows::Foundation::Size m_renderTargetSize;
	Windows::Foundation::Rect m_windowBounds;
	Platform::Agile<Windows::UI::Core::CoreWindow> m_window;
	Windows::Graphics::Display::DisplayOrientations m_orientation;

	// Transform used for display orientation.
	DirectX::XMFLOAT4X4 m_orientationTransform3D;

	HQRenderer m_renderer;
	HQRenderDevice * m_device;
};
