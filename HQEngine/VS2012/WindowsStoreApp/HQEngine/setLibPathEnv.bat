set CDIR=%~dp0

reg add HKCU\Environment  /v HQENGINE_WINRT_X86_LIB_DEB_PATH /t REG_SZ /d %CDIR%\Debug\HQEngine
reg add HKCU\Environment  /v HQENGINE_WINRT_X86_LIB_REL_PATH /t REG_SZ /d %CDIR%\Release\HQEngine
reg add HKCU\Environment  /v HQENGINE_WINRT_ARM_LIB_DEB_PATH /t REG_SZ /d %CDIR%ARM\Debug\HQEngine
reg add HKCU\Environment  /v HQENGINE_WINRT_ARM_LIB_REL_PATH /t REG_SZ /d %CDIR%ARM\Release\HQEngine