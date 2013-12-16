/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.opengl.GLSurfaceView
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_OPENGL_GLSURFACEVIEW_HPP_DECL
#define J2CPP_ANDROID_OPENGL_GLSURFACEVIEW_HPP_DECL


namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace egl { class EGL10; } } } } }
namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace egl { class EGLDisplay; } } } } }
namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace egl { class EGLSurface; } } } } }
namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace egl { class EGLConfig; } } } } }
namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace egl { class EGLContext; } } } } }
namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace opengles { class GL; } } } } }
namespace j2cpp { namespace javax { namespace microedition { namespace khronos { namespace opengles { class GL10; } } } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Runnable; } } }
namespace j2cpp { namespace android { namespace opengl { namespace GLSurfaceView_ { class EGLContextFactory; } } } }
namespace j2cpp { namespace android { namespace opengl { namespace GLSurfaceView_ { class Renderer; } } } }
namespace j2cpp { namespace android { namespace opengl { namespace GLSurfaceView_ { class EGLWindowSurfaceFactory; } } } }
namespace j2cpp { namespace android { namespace opengl { namespace GLSurfaceView_ { class EGLConfigChooser; } } } }
namespace j2cpp { namespace android { namespace opengl { namespace GLSurfaceView_ { class GLWrapper; } } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace view { class SurfaceView; } } }
namespace j2cpp { namespace android { namespace view { class SurfaceHolder; } } }
namespace j2cpp { namespace android { namespace view { namespace SurfaceHolder_ { class Callback; } } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/opengl/GLSurfaceView.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/view/SurfaceHolder.hpp>
#include <android/view/SurfaceView.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/Runnable.hpp>
#include <javax/microedition/khronos/egl/EGL10.hpp>
#include <javax/microedition/khronos/egl/EGLConfig.hpp>
#include <javax/microedition/khronos/egl/EGLContext.hpp>
#include <javax/microedition/khronos/egl/EGLDisplay.hpp>
#include <javax/microedition/khronos/egl/EGLSurface.hpp>
#include <javax/microedition/khronos/opengles/GL.hpp>
#include <javax/microedition/khronos/opengles/GL10.hpp>


namespace j2cpp {

namespace android { namespace opengl {

	class GLSurfaceView;
	namespace GLSurfaceView_ {

		class EGLContextFactory;
		class EGLContextFactory
			: public object<EGLContextFactory>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)
			J2CPP_DECLARE_METHOD(1)

			explicit EGLContextFactory(jobject jobj)
			: object<EGLContextFactory>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			local_ref< javax::microedition::khronos::egl::EGLContext > createContext(local_ref< javax::microedition::khronos::egl::EGL10 >  const&, local_ref< javax::microedition::khronos::egl::EGLDisplay >  const&, local_ref< javax::microedition::khronos::egl::EGLConfig >  const&);
			void destroyContext(local_ref< javax::microedition::khronos::egl::EGL10 >  const&, local_ref< javax::microedition::khronos::egl::EGLDisplay >  const&, local_ref< javax::microedition::khronos::egl::EGLContext >  const&);
		}; //class EGLContextFactory

		class Renderer;
		class Renderer
			: public object<Renderer>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)
			J2CPP_DECLARE_METHOD(1)
			J2CPP_DECLARE_METHOD(2)

			explicit Renderer(jobject jobj)
			: object<Renderer>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			void onSurfaceCreated(local_ref< javax::microedition::khronos::opengles::GL10 >  const&, local_ref< javax::microedition::khronos::egl::EGLConfig >  const&);
			void onSurfaceChanged(local_ref< javax::microedition::khronos::opengles::GL10 >  const&, jint, jint);
			void onDrawFrame(local_ref< javax::microedition::khronos::opengles::GL10 >  const&);
		}; //class Renderer

		class EGLWindowSurfaceFactory;
		class EGLWindowSurfaceFactory
			: public object<EGLWindowSurfaceFactory>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)
			J2CPP_DECLARE_METHOD(1)

			explicit EGLWindowSurfaceFactory(jobject jobj)
			: object<EGLWindowSurfaceFactory>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			local_ref< javax::microedition::khronos::egl::EGLSurface > createWindowSurface(local_ref< javax::microedition::khronos::egl::EGL10 >  const&, local_ref< javax::microedition::khronos::egl::EGLDisplay >  const&, local_ref< javax::microedition::khronos::egl::EGLConfig >  const&, local_ref< java::lang::Object >  const&);
			void destroySurface(local_ref< javax::microedition::khronos::egl::EGL10 >  const&, local_ref< javax::microedition::khronos::egl::EGLDisplay >  const&, local_ref< javax::microedition::khronos::egl::EGLSurface >  const&);
		}; //class EGLWindowSurfaceFactory

		class EGLConfigChooser;
		class EGLConfigChooser
			: public object<EGLConfigChooser>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)

			explicit EGLConfigChooser(jobject jobj)
			: object<EGLConfigChooser>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			local_ref< javax::microedition::khronos::egl::EGLConfig > chooseConfig(local_ref< javax::microedition::khronos::egl::EGL10 >  const&, local_ref< javax::microedition::khronos::egl::EGLDisplay >  const&);
		}; //class EGLConfigChooser

		class GLWrapper;
		class GLWrapper
			: public object<GLWrapper>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)

			explicit GLWrapper(jobject jobj)
			: object<GLWrapper>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			local_ref< javax::microedition::khronos::opengles::GL > wrap(local_ref< javax::microedition::khronos::opengles::GL >  const&);
		}; //class GLWrapper

	} //namespace GLSurfaceView_

	class GLSurfaceView
		: public object<GLSurfaceView>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)
		J2CPP_DECLARE_METHOD(6)
		J2CPP_DECLARE_METHOD(7)
		J2CPP_DECLARE_METHOD(8)
		J2CPP_DECLARE_METHOD(9)
		J2CPP_DECLARE_METHOD(10)
		J2CPP_DECLARE_METHOD(11)
		J2CPP_DECLARE_METHOD(12)
		J2CPP_DECLARE_METHOD(13)
		J2CPP_DECLARE_METHOD(14)
		J2CPP_DECLARE_METHOD(15)
		J2CPP_DECLARE_METHOD(16)
		J2CPP_DECLARE_METHOD(17)
		J2CPP_DECLARE_METHOD(18)
		J2CPP_DECLARE_METHOD(19)
		J2CPP_DECLARE_METHOD(20)
		J2CPP_DECLARE_FIELD(0)
		J2CPP_DECLARE_FIELD(1)
		J2CPP_DECLARE_FIELD(2)
		J2CPP_DECLARE_FIELD(3)

		typedef GLSurfaceView_::EGLContextFactory EGLContextFactory;
		typedef GLSurfaceView_::Renderer Renderer;
		typedef GLSurfaceView_::EGLWindowSurfaceFactory EGLWindowSurfaceFactory;
		typedef GLSurfaceView_::EGLConfigChooser EGLConfigChooser;
		typedef GLSurfaceView_::GLWrapper GLWrapper;

		explicit GLSurfaceView(jobject jobj)
		: object<GLSurfaceView>(jobj)
		{
		}

		operator local_ref<android::view::SurfaceView>() const;
		operator local_ref<android::view::SurfaceHolder_::Callback>() const;


		GLSurfaceView(local_ref< android::content::Context > const&);
		GLSurfaceView(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		void setGLWrapper(local_ref< android::opengl::GLSurfaceView_::GLWrapper >  const&);
		void setDebugFlags(jint);
		jint getDebugFlags();
		void setRenderer(local_ref< android::opengl::GLSurfaceView_::Renderer >  const&);
		void setEGLContextFactory(local_ref< android::opengl::GLSurfaceView_::EGLContextFactory >  const&);
		void setEGLWindowSurfaceFactory(local_ref< android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory >  const&);
		void setEGLConfigChooser(local_ref< android::opengl::GLSurfaceView_::EGLConfigChooser >  const&);
		void setEGLConfigChooser(jboolean);
		void setEGLConfigChooser(jint, jint, jint, jint, jint, jint);
		void setRenderMode(jint);
		jint getRenderMode();
		void requestRender();
		void surfaceCreated(local_ref< android::view::SurfaceHolder >  const&);
		void surfaceDestroyed(local_ref< android::view::SurfaceHolder >  const&);
		void surfaceChanged(local_ref< android::view::SurfaceHolder >  const&, jint, jint, jint);
		void onPause();
		void onResume();
		void queueEvent(local_ref< java::lang::Runnable >  const&);

		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), jint > RENDERMODE_WHEN_DIRTY;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(1), J2CPP_FIELD_SIGNATURE(1), jint > RENDERMODE_CONTINUOUSLY;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(2), J2CPP_FIELD_SIGNATURE(2), jint > DEBUG_CHECK_GL_ERROR;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(3), J2CPP_FIELD_SIGNATURE(3), jint > DEBUG_LOG_GL_CALLS;
	}; //class GLSurfaceView

} //namespace opengl
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_OPENGL_GLSURFACEVIEW_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_OPENGL_GLSURFACEVIEW_HPP_IMPL
#define J2CPP_ANDROID_OPENGL_GLSURFACEVIEW_HPP_IMPL

namespace j2cpp {




android::opengl::GLSurfaceView_::EGLContextFactory::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

local_ref< javax::microedition::khronos::egl::EGLContext > android::opengl::GLSurfaceView_::EGLContextFactory::createContext(local_ref< javax::microedition::khronos::egl::EGL10 > const &a0, local_ref< javax::microedition::khronos::egl::EGLDisplay > const &a1, local_ref< javax::microedition::khronos::egl::EGLConfig > const &a2)
{
	return call_method<
		android::opengl::GLSurfaceView_::EGLContextFactory::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::EGLContextFactory::J2CPP_METHOD_NAME(0),
		android::opengl::GLSurfaceView_::EGLContextFactory::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< javax::microedition::khronos::egl::EGLContext >
	>(get_jobject(), a0, a1, a2);
}

void android::opengl::GLSurfaceView_::EGLContextFactory::destroyContext(local_ref< javax::microedition::khronos::egl::EGL10 > const &a0, local_ref< javax::microedition::khronos::egl::EGLDisplay > const &a1, local_ref< javax::microedition::khronos::egl::EGLContext > const &a2)
{
	return call_method<
		android::opengl::GLSurfaceView_::EGLContextFactory::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::EGLContextFactory::J2CPP_METHOD_NAME(1),
		android::opengl::GLSurfaceView_::EGLContextFactory::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0, a1, a2);
}


J2CPP_DEFINE_CLASS(android::opengl::GLSurfaceView_::EGLContextFactory,"android/opengl/GLSurfaceView$EGLContextFactory")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::EGLContextFactory,0,"createContext","(Ljavax/microedition/khronos/egl/EGL10;Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLConfig;)Ljavax/microedition/khronos/egl/EGLContext;")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::EGLContextFactory,1,"destroyContext","(Ljavax/microedition/khronos/egl/EGL10;Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLContext;)V")


android::opengl::GLSurfaceView_::Renderer::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

void android::opengl::GLSurfaceView_::Renderer::onSurfaceCreated(local_ref< javax::microedition::khronos::opengles::GL10 > const &a0, local_ref< javax::microedition::khronos::egl::EGLConfig > const &a1)
{
	return call_method<
		android::opengl::GLSurfaceView_::Renderer::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::Renderer::J2CPP_METHOD_NAME(0),
		android::opengl::GLSurfaceView_::Renderer::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject(), a0, a1);
}

void android::opengl::GLSurfaceView_::Renderer::onSurfaceChanged(local_ref< javax::microedition::khronos::opengles::GL10 > const &a0, jint a1, jint a2)
{
	return call_method<
		android::opengl::GLSurfaceView_::Renderer::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::Renderer::J2CPP_METHOD_NAME(1),
		android::opengl::GLSurfaceView_::Renderer::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0, a1, a2);
}

void android::opengl::GLSurfaceView_::Renderer::onDrawFrame(local_ref< javax::microedition::khronos::opengles::GL10 > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView_::Renderer::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::Renderer::J2CPP_METHOD_NAME(2),
		android::opengl::GLSurfaceView_::Renderer::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::opengl::GLSurfaceView_::Renderer,"android/opengl/GLSurfaceView$Renderer")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::Renderer,0,"onSurfaceCreated","(Ljavax/microedition/khronos/opengles/GL10;Ljavax/microedition/khronos/egl/EGLConfig;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::Renderer,1,"onSurfaceChanged","(Ljavax/microedition/khronos/opengles/GL10;II)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::Renderer,2,"onDrawFrame","(Ljavax/microedition/khronos/opengles/GL10;)V")


android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

local_ref< javax::microedition::khronos::egl::EGLSurface > android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::createWindowSurface(local_ref< javax::microedition::khronos::egl::EGL10 > const &a0, local_ref< javax::microedition::khronos::egl::EGLDisplay > const &a1, local_ref< javax::microedition::khronos::egl::EGLConfig > const &a2, local_ref< java::lang::Object > const &a3)
{
	return call_method<
		android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::J2CPP_METHOD_NAME(0),
		android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< javax::microedition::khronos::egl::EGLSurface >
	>(get_jobject(), a0, a1, a2, a3);
}

void android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::destroySurface(local_ref< javax::microedition::khronos::egl::EGL10 > const &a0, local_ref< javax::microedition::khronos::egl::EGLDisplay > const &a1, local_ref< javax::microedition::khronos::egl::EGLSurface > const &a2)
{
	return call_method<
		android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::J2CPP_METHOD_NAME(1),
		android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0, a1, a2);
}


J2CPP_DEFINE_CLASS(android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory,"android/opengl/GLSurfaceView$EGLWindowSurfaceFactory")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory,0,"createWindowSurface","(Ljavax/microedition/khronos/egl/EGL10;Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLConfig;Ljava/lang/Object;)Ljavax/microedition/khronos/egl/EGLSurface;")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory,1,"destroySurface","(Ljavax/microedition/khronos/egl/EGL10;Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLSurface;)V")


android::opengl::GLSurfaceView_::EGLConfigChooser::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

local_ref< javax::microedition::khronos::egl::EGLConfig > android::opengl::GLSurfaceView_::EGLConfigChooser::chooseConfig(local_ref< javax::microedition::khronos::egl::EGL10 > const &a0, local_ref< javax::microedition::khronos::egl::EGLDisplay > const &a1)
{
	return call_method<
		android::opengl::GLSurfaceView_::EGLConfigChooser::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::EGLConfigChooser::J2CPP_METHOD_NAME(0),
		android::opengl::GLSurfaceView_::EGLConfigChooser::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< javax::microedition::khronos::egl::EGLConfig >
	>(get_jobject(), a0, a1);
}


J2CPP_DEFINE_CLASS(android::opengl::GLSurfaceView_::EGLConfigChooser,"android/opengl/GLSurfaceView$EGLConfigChooser")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::EGLConfigChooser,0,"chooseConfig","(Ljavax/microedition/khronos/egl/EGL10;Ljavax/microedition/khronos/egl/EGLDisplay;)Ljavax/microedition/khronos/egl/EGLConfig;")


android::opengl::GLSurfaceView_::GLWrapper::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

local_ref< javax::microedition::khronos::opengles::GL > android::opengl::GLSurfaceView_::GLWrapper::wrap(local_ref< javax::microedition::khronos::opengles::GL > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView_::GLWrapper::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView_::GLWrapper::J2CPP_METHOD_NAME(0),
		android::opengl::GLSurfaceView_::GLWrapper::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< javax::microedition::khronos::opengles::GL >
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::opengl::GLSurfaceView_::GLWrapper,"android/opengl/GLSurfaceView$GLWrapper")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView_::GLWrapper,0,"wrap","(Ljavax/microedition/khronos/opengles/GL;)Ljavax/microedition/khronos/opengles/GL;")



android::opengl::GLSurfaceView::operator local_ref<android::view::SurfaceView>() const
{
	return local_ref<android::view::SurfaceView>(get_jobject());
}

android::opengl::GLSurfaceView::operator local_ref<android::view::SurfaceHolder_::Callback>() const
{
	return local_ref<android::view::SurfaceHolder_::Callback>(get_jobject());
}


android::opengl::GLSurfaceView::GLSurfaceView(local_ref< android::content::Context > const &a0)
: object<android::opengl::GLSurfaceView>(
	call_new_object<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(0),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



android::opengl::GLSurfaceView::GLSurfaceView(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::opengl::GLSurfaceView>(
	call_new_object<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(1),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}


void android::opengl::GLSurfaceView::setGLWrapper(local_ref< android::opengl::GLSurfaceView_::GLWrapper > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(2),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::setDebugFlags(jint a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(3),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

jint android::opengl::GLSurfaceView::getDebugFlags()
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(4),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

void android::opengl::GLSurfaceView::setRenderer(local_ref< android::opengl::GLSurfaceView_::Renderer > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(5),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::setEGLContextFactory(local_ref< android::opengl::GLSurfaceView_::EGLContextFactory > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(6),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::setEGLWindowSurfaceFactory(local_ref< android::opengl::GLSurfaceView_::EGLWindowSurfaceFactory > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(7),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(7), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::setEGLConfigChooser(local_ref< android::opengl::GLSurfaceView_::EGLConfigChooser > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(8),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::setEGLConfigChooser(jboolean a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(9),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::setEGLConfigChooser(jint a0, jint a1, jint a2, jint a3, jint a4, jint a5)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(10),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject(), a0, a1, a2, a3, a4, a5);
}

void android::opengl::GLSurfaceView::setRenderMode(jint a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(11),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(11), 
		void
	>(get_jobject(), a0);
}

jint android::opengl::GLSurfaceView::getRenderMode()
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(12),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(12), 
		jint
	>(get_jobject());
}

void android::opengl::GLSurfaceView::requestRender()
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(13),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(13), 
		void
	>(get_jobject());
}

void android::opengl::GLSurfaceView::surfaceCreated(local_ref< android::view::SurfaceHolder > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(14),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(14), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::surfaceDestroyed(local_ref< android::view::SurfaceHolder > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(15),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(15), 
		void
	>(get_jobject(), a0);
}

void android::opengl::GLSurfaceView::surfaceChanged(local_ref< android::view::SurfaceHolder > const &a0, jint a1, jint a2, jint a3)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(16),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(16), 
		void
	>(get_jobject(), a0, a1, a2, a3);
}

void android::opengl::GLSurfaceView::onPause()
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(17),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(17), 
		void
	>(get_jobject());
}

void android::opengl::GLSurfaceView::onResume()
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(18),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(18), 
		void
	>(get_jobject());
}

void android::opengl::GLSurfaceView::queueEvent(local_ref< java::lang::Runnable > const &a0)
{
	return call_method<
		android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
		android::opengl::GLSurfaceView::J2CPP_METHOD_NAME(19),
		android::opengl::GLSurfaceView::J2CPP_METHOD_SIGNATURE(19), 
		void
	>(get_jobject(), a0);
}



static_field<
	android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
	android::opengl::GLSurfaceView::J2CPP_FIELD_NAME(0),
	android::opengl::GLSurfaceView::J2CPP_FIELD_SIGNATURE(0),
	jint
> android::opengl::GLSurfaceView::RENDERMODE_WHEN_DIRTY;

static_field<
	android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
	android::opengl::GLSurfaceView::J2CPP_FIELD_NAME(1),
	android::opengl::GLSurfaceView::J2CPP_FIELD_SIGNATURE(1),
	jint
> android::opengl::GLSurfaceView::RENDERMODE_CONTINUOUSLY;

static_field<
	android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
	android::opengl::GLSurfaceView::J2CPP_FIELD_NAME(2),
	android::opengl::GLSurfaceView::J2CPP_FIELD_SIGNATURE(2),
	jint
> android::opengl::GLSurfaceView::DEBUG_CHECK_GL_ERROR;

static_field<
	android::opengl::GLSurfaceView::J2CPP_CLASS_NAME,
	android::opengl::GLSurfaceView::J2CPP_FIELD_NAME(3),
	android::opengl::GLSurfaceView::J2CPP_FIELD_SIGNATURE(3),
	jint
> android::opengl::GLSurfaceView::DEBUG_LOG_GL_CALLS;


J2CPP_DEFINE_CLASS(android::opengl::GLSurfaceView,"android/opengl/GLSurfaceView")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,0,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,1,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,2,"setGLWrapper","(Landroid/opengl/GLSurfaceView$GLWrapper;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,3,"setDebugFlags","(I)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,4,"getDebugFlags","()I")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,5,"setRenderer","(Landroid/opengl/GLSurfaceView$Renderer;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,6,"setEGLContextFactory","(Landroid/opengl/GLSurfaceView$EGLContextFactory;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,7,"setEGLWindowSurfaceFactory","(Landroid/opengl/GLSurfaceView$EGLWindowSurfaceFactory;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,8,"setEGLConfigChooser","(Landroid/opengl/GLSurfaceView$EGLConfigChooser;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,9,"setEGLConfigChooser","(Z)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,10,"setEGLConfigChooser","(IIIIII)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,11,"setRenderMode","(I)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,12,"getRenderMode","()I")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,13,"requestRender","()V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,14,"surfaceCreated","(Landroid/view/SurfaceHolder;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,15,"surfaceDestroyed","(Landroid/view/SurfaceHolder;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,16,"surfaceChanged","(Landroid/view/SurfaceHolder;III)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,17,"onPause","()V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,18,"onResume","()V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,19,"queueEvent","(Ljava/lang/Runnable;)V")
J2CPP_DEFINE_METHOD(android::opengl::GLSurfaceView,20,"onDetachedFromWindow","()V")
J2CPP_DEFINE_FIELD(android::opengl::GLSurfaceView,0,"RENDERMODE_WHEN_DIRTY","I")
J2CPP_DEFINE_FIELD(android::opengl::GLSurfaceView,1,"RENDERMODE_CONTINUOUSLY","I")
J2CPP_DEFINE_FIELD(android::opengl::GLSurfaceView,2,"DEBUG_CHECK_GL_ERROR","I")
J2CPP_DEFINE_FIELD(android::opengl::GLSurfaceView,3,"DEBUG_LOG_GL_CALLS","I")

} //namespace j2cpp

#endif //J2CPP_ANDROID_OPENGL_GLSURFACEVIEW_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
