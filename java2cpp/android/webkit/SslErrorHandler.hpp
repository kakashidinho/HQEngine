/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.webkit.SslErrorHandler
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WEBKIT_SSLERRORHANDLER_HPP_DECL
#define J2CPP_ANDROID_WEBKIT_SSLERRORHANDLER_HPP_DECL


namespace j2cpp { namespace android { namespace os { class Handler; } } }
namespace j2cpp { namespace android { namespace os { class Message; } } }


#include <android/os/Handler.hpp>
#include <android/os/Message.hpp>


namespace j2cpp {

namespace android { namespace webkit {

	class SslErrorHandler;
	class SslErrorHandler
		: public object<SslErrorHandler>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)

		explicit SslErrorHandler(jobject jobj)
		: object<SslErrorHandler>(jobj)
		{
		}

		operator local_ref<android::os::Handler>() const;


		void handleMessage(local_ref< android::os::Message >  const&);
		void proceed();
		void cancel();
	}; //class SslErrorHandler

} //namespace webkit
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_WEBKIT_SSLERRORHANDLER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WEBKIT_SSLERRORHANDLER_HPP_IMPL
#define J2CPP_ANDROID_WEBKIT_SSLERRORHANDLER_HPP_IMPL

namespace j2cpp {



android::webkit::SslErrorHandler::operator local_ref<android::os::Handler>() const
{
	return local_ref<android::os::Handler>(get_jobject());
}


void android::webkit::SslErrorHandler::handleMessage(local_ref< android::os::Message > const &a0)
{
	return call_method<
		android::webkit::SslErrorHandler::J2CPP_CLASS_NAME,
		android::webkit::SslErrorHandler::J2CPP_METHOD_NAME(1),
		android::webkit::SslErrorHandler::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0);
}

void android::webkit::SslErrorHandler::proceed()
{
	return call_method<
		android::webkit::SslErrorHandler::J2CPP_CLASS_NAME,
		android::webkit::SslErrorHandler::J2CPP_METHOD_NAME(2),
		android::webkit::SslErrorHandler::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject());
}

void android::webkit::SslErrorHandler::cancel()
{
	return call_method<
		android::webkit::SslErrorHandler::J2CPP_CLASS_NAME,
		android::webkit::SslErrorHandler::J2CPP_METHOD_NAME(3),
		android::webkit::SslErrorHandler::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(android::webkit::SslErrorHandler,"android/webkit/SslErrorHandler")
J2CPP_DEFINE_METHOD(android::webkit::SslErrorHandler,0,"<init>","()V")
J2CPP_DEFINE_METHOD(android::webkit::SslErrorHandler,1,"handleMessage","(Landroid/os/Message;)V")
J2CPP_DEFINE_METHOD(android::webkit::SslErrorHandler,2,"proceed","()V")
J2CPP_DEFINE_METHOD(android::webkit::SslErrorHandler,3,"cancel","()V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_WEBKIT_SSLERRORHANDLER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
