/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.apache.http.impl.cookie.RFC2965Spec
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_RFC2965SPEC_HPP_DECL
#define J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_RFC2965SPEC_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace util { class List; } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace cookie { class CookieOrigin; } } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace cookie { class Cookie; } } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class Header; } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace impl { namespace cookie { class RFC2109Spec; } } } } } }


#include <java/lang/String.hpp>
#include <java/util/List.hpp>
#include <org/apache/http/Header.hpp>
#include <org/apache/http/cookie/Cookie.hpp>
#include <org/apache/http/cookie/CookieOrigin.hpp>
#include <org/apache/http/impl/cookie/RFC2109Spec.hpp>


namespace j2cpp {

namespace org { namespace apache { namespace http { namespace impl { namespace cookie {

	class RFC2965Spec;
	class RFC2965Spec
		: public object<RFC2965Spec>
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

		explicit RFC2965Spec(jobject jobj)
		: object<RFC2965Spec>(jobj)
		{
		}

		operator local_ref<org::apache::http::impl::cookie::RFC2109Spec>() const;


		RFC2965Spec();
		RFC2965Spec(local_ref< array< local_ref< java::lang::String >, 1> > const&, jboolean);
		local_ref< java::util::List > parse(local_ref< org::apache::http::Header >  const&, local_ref< org::apache::http::cookie::CookieOrigin >  const&);
		void validate(local_ref< org::apache::http::cookie::Cookie >  const&, local_ref< org::apache::http::cookie::CookieOrigin >  const&);
		jboolean match(local_ref< org::apache::http::cookie::Cookie >  const&, local_ref< org::apache::http::cookie::CookieOrigin >  const&);
		jint getVersion();
		local_ref< org::apache::http::Header > getVersionHeader();
	}; //class RFC2965Spec

} //namespace cookie
} //namespace impl
} //namespace http
} //namespace apache
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_RFC2965SPEC_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_RFC2965SPEC_HPP_IMPL
#define J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_RFC2965SPEC_HPP_IMPL

namespace j2cpp {



org::apache::http::impl::cookie::RFC2965Spec::operator local_ref<org::apache::http::impl::cookie::RFC2109Spec>() const
{
	return local_ref<org::apache::http::impl::cookie::RFC2109Spec>(get_jobject());
}


org::apache::http::impl::cookie::RFC2965Spec::RFC2965Spec()
: object<org::apache::http::impl::cookie::RFC2965Spec>(
	call_new_object<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(0),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



org::apache::http::impl::cookie::RFC2965Spec::RFC2965Spec(local_ref< array< local_ref< java::lang::String >, 1> > const &a0, jboolean a1)
: object<org::apache::http::impl::cookie::RFC2965Spec>(
	call_new_object<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(1),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}


local_ref< java::util::List > org::apache::http::impl::cookie::RFC2965Spec::parse(local_ref< org::apache::http::Header > const &a0, local_ref< org::apache::http::cookie::CookieOrigin > const &a1)
{
	return call_method<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(2),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::util::List >
	>(get_jobject(), a0, a1);
}

void org::apache::http::impl::cookie::RFC2965Spec::validate(local_ref< org::apache::http::cookie::Cookie > const &a0, local_ref< org::apache::http::cookie::CookieOrigin > const &a1)
{
	return call_method<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(3),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0, a1);
}

jboolean org::apache::http::impl::cookie::RFC2965Spec::match(local_ref< org::apache::http::cookie::Cookie > const &a0, local_ref< org::apache::http::cookie::CookieOrigin > const &a1)
{
	return call_method<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(4),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(4), 
		jboolean
	>(get_jobject(), a0, a1);
}


jint org::apache::http::impl::cookie::RFC2965Spec::getVersion()
{
	return call_method<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(6),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(6), 
		jint
	>(get_jobject());
}

local_ref< org::apache::http::Header > org::apache::http::impl::cookie::RFC2965Spec::getVersionHeader()
{
	return call_method<
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_NAME(7),
		org::apache::http::impl::cookie::RFC2965Spec::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< org::apache::http::Header >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(org::apache::http::impl::cookie::RFC2965Spec,"org/apache/http/impl/cookie/RFC2965Spec")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,1,"<init>","([java.lang.StringZ)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,2,"parse","(Lorg/apache/http/Header;Lorg/apache/http/cookie/CookieOrigin;)Ljava/util/List;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,3,"validate","(Lorg/apache/http/cookie/Cookie;Lorg/apache/http/cookie/CookieOrigin;)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,4,"match","(Lorg/apache/http/cookie/Cookie;Lorg/apache/http/cookie/CookieOrigin;)Z")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,5,"formatCookieAsVer","(Lorg/apache/http/util/CharArrayBuffer;Lorg/apache/http/cookie/Cookie;I)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,6,"getVersion","()I")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::RFC2965Spec,7,"getVersionHeader","()Lorg/apache/http/Header;")

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_RFC2965SPEC_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
