/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.apache.http.impl.cookie.BasicSecureHandler
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_BASICSECUREHANDLER_HPP_DECL
#define J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_BASICSECUREHANDLER_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace cookie { class CookieOrigin; } } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace cookie { class Cookie; } } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace cookie { class SetCookie; } } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace impl { namespace cookie { class AbstractCookieAttributeHandler; } } } } } }


#include <java/lang/String.hpp>
#include <org/apache/http/cookie/Cookie.hpp>
#include <org/apache/http/cookie/CookieOrigin.hpp>
#include <org/apache/http/cookie/SetCookie.hpp>
#include <org/apache/http/impl/cookie/AbstractCookieAttributeHandler.hpp>


namespace j2cpp {

namespace org { namespace apache { namespace http { namespace impl { namespace cookie {

	class BasicSecureHandler;
	class BasicSecureHandler
		: public object<BasicSecureHandler>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)

		explicit BasicSecureHandler(jobject jobj)
		: object<BasicSecureHandler>(jobj)
		{
		}

		operator local_ref<org::apache::http::impl::cookie::AbstractCookieAttributeHandler>() const;


		BasicSecureHandler();
		void parse(local_ref< org::apache::http::cookie::SetCookie >  const&, local_ref< java::lang::String >  const&);
		jboolean match(local_ref< org::apache::http::cookie::Cookie >  const&, local_ref< org::apache::http::cookie::CookieOrigin >  const&);
	}; //class BasicSecureHandler

} //namespace cookie
} //namespace impl
} //namespace http
} //namespace apache
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_BASICSECUREHANDLER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_BASICSECUREHANDLER_HPP_IMPL
#define J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_BASICSECUREHANDLER_HPP_IMPL

namespace j2cpp {



org::apache::http::impl::cookie::BasicSecureHandler::operator local_ref<org::apache::http::impl::cookie::AbstractCookieAttributeHandler>() const
{
	return local_ref<org::apache::http::impl::cookie::AbstractCookieAttributeHandler>(get_jobject());
}


org::apache::http::impl::cookie::BasicSecureHandler::BasicSecureHandler()
: object<org::apache::http::impl::cookie::BasicSecureHandler>(
	call_new_object<
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_METHOD_NAME(0),
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}


void org::apache::http::impl::cookie::BasicSecureHandler::parse(local_ref< org::apache::http::cookie::SetCookie > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_METHOD_NAME(1),
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0, a1);
}

jboolean org::apache::http::impl::cookie::BasicSecureHandler::match(local_ref< org::apache::http::cookie::Cookie > const &a0, local_ref< org::apache::http::cookie::CookieOrigin > const &a1)
{
	return call_method<
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_METHOD_NAME(2),
		org::apache::http::impl::cookie::BasicSecureHandler::J2CPP_METHOD_SIGNATURE(2), 
		jboolean
	>(get_jobject(), a0, a1);
}


J2CPP_DEFINE_CLASS(org::apache::http::impl::cookie::BasicSecureHandler,"org/apache/http/impl/cookie/BasicSecureHandler")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::BasicSecureHandler,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::BasicSecureHandler,1,"parse","(Lorg/apache/http/cookie/SetCookie;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::BasicSecureHandler,2,"match","(Lorg/apache/http/cookie/Cookie;Lorg/apache/http/cookie/CookieOrigin;)Z")

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_BASICSECUREHANDLER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
