set CDIR=%~dp0

reg add HKCU\Environment  /v HQENGINE_WINPHONE_X86_LIB_DEB_PATH /t REG_SZ /d %CDIR%\Debug\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_X86_LIB_REL_PATH /t REG_SZ /d %CDIR%\Release\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_ARM_LIB_DEB_PATH /t REG_SZ /d %CDIR%ARM\Debug\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_ARM_LIB_REL_PATH /t REG_SZ /d %CDIR%ARM\Release\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_X86_STATIC_LIB_DEB_PATH /t REG_SZ /d %CDIR%\StaticDebug\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_X86_STATIC_LIB_REL_PATH /t REG_SZ /d %CDIR%\StaticRelease\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_ARM_STATIC_LIB_DEB_PATH /t REG_SZ /d %CDIR%ARM\StaticDebug\HQEngineWP
reg add HKCU\Environment  /v HQENGINE_WINPHONE_ARM_STATIC_LIB_REL_PATH /t REG_SZ /d %CDIR%ARM\StaticRelease\HQEngineWP