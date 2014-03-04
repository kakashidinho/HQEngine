/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_CG_FUNC_PTR_H
#define HQ_CG_FUNC_PTR_H

#define CG_EXPLICIT
#define CGGL_EXPLICIT
#include "Cg/cg.h"
#include "Cg/cgGL.h"


typedef CGerror (CGENTRY* pfcgGetError) (void);
typedef void (CGENTRY *pfcgSetErrorCallback)(CGerrorCallbackFunc func);
typedef const char* (CGENTRY* pfcgGetErrorString) (CGerror error);
typedef CGparameter (CGENTRY* pfcgGetNamedParameter)(CGprogram program, const char *name);
typedef CGtype (CGENTRY *pfcgGetParameterType)(CGparameter param);
typedef CGcontext (CGENTRY *pfcgCreateContext)(void);
typedef void (CGENTRY *pfcgDestroyContext)(CGcontext context);
typedef CGprogram (CGENTRY *pfcgCreateProgramFromFile)(CGcontext context, CGenum program_type, const char *program_file, CGprofile profile, const char *entry, const char **args);
typedef CGprogram (CGENTRY *pfcgCreateProgram)(CGcontext context, CGenum program_type, const char *program, CGprofile profile, const char *entry, const char **args);
typedef CGprogram (CGENTRY *pfcgCombinePrograms2)(const CGprogram exe1, const CGprogram exe2);
typedef CGprogram (CGENTRY *pfcgCombinePrograms3)(const CGprogram exe1, const CGprogram exe2, const CGprogram exe3);
typedef void (CGENTRY *pfcgDestroyProgram)(CGprogram program);
typedef void (CGENTRY *pfcgSetParameterValueir)(CGparameter param, int nelements, const int *vals);
typedef void (CGENTRY *pfcgSetParameterValuefr)(CGparameter param, int nelements, const float *vals);
typedef int (CGGLENTRY *pfcgGetArrayTotalSize)(CGparameter param);
typedef void (CGGLENTRY *pfcgSetParameter1iv) ( CGparameter param, const int * v );
typedef void (CGGLENTRY *pfcgSetParameter2iv) ( CGparameter param, const int * v );
typedef void (CGGLENTRY *pfcgSetParameter3iv) ( CGparameter param, const int * v );
typedef void (CGGLENTRY *pfcgSetParameter4iv) ( CGparameter param, const int * v );
typedef void (CGGLENTRY *pfcgSetParameter1fv) ( CGparameter param, const float * v );
typedef void (CGGLENTRY *pfcgSetParameter2fv) ( CGparameter param, const float * v );
typedef void (CGGLENTRY *pfcgSetParameter3fv) ( CGparameter param, const float * v );
typedef void (CGGLENTRY *pfcgSetParameter4fv) ( CGparameter param, const float * v );
typedef const char * (CGGLENTRY *pfcgGetString) ( CGenum _enum );
typedef CGbool (CGGLENTRY *pfcgGLIsProfileSupported)(CGprofile profile);
typedef CGprofile (CGENTRY *pfcgGLGetLatestProfile)(CGGLenum profile_type);
typedef void (CGGLENTRY *pfcgGLSetOptimalOptions)(CGprofile profile);
typedef void (CGGLENTRY *pfcgGLSetDebugMode)(CGbool debug);
typedef void (CGGLENTRY *pfcgGLLoadProgram)(CGprogram program);
typedef void (CGGLENTRY *pfcgGLUnbindProgram)(CGprofile profile);
typedef void (CGGLENTRY *pfcgGLDisableProfile)(CGprofile profile);
typedef void (CGGLENTRY *pfcgGLBindProgram)(CGprogram program);
typedef void (CGGLENTRY *pfcgGLEnableProgramProfiles) ( CGprogram program );
typedef void (CGGLENTRY *pfcgGLSetContextGLSLVersion) ( CGcontext context, CGGLglslversion version );
typedef CGGLglslversion (CGGLENTRY *pfcgGLGetGLSLVersion) (const char *version_string);
typedef CGGLglslversion (CGGLENTRY *pfcgGLGetContextGLSLVersion) ( CGcontext context );


#ifdef WIN32
#	define HQ_GET_CG_FUNC_PTR(lib , name) name = (pf##name) GetProcAddress(lib , #name)
#else
#	define HQ_GET_CG_FUNC_PTR(lib , name) name = reinterpret_cast<pf##name>( dlsym(lib , #name) )
#endif



#endif // HQ_CG_FUNC_PTR_H
