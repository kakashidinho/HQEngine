/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.apache.http.impl.cookie.DateParseException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_DATEPARSEEXCEPTION_HPP_DECL
#define J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_DATEPARSEEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace lang { class Exception; } } }


#include <java/lang/Exception.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace org { namespace apache { namespace http { namespace impl { namespace cookie {

	class DateParseException;
	class DateParseException
		: public object<DateParseException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit DateParseException(jobject jobj)
		: object<DateParseException>(jobj)
		{
		}

		operator local_ref<java::lang::Exception>() const;


		DateParseException();
		DateParseException(local_ref< java::lang::String > const&);
	}; //class DateParseException

} //namespace cookie
} //namespace impl
} //namespace http
} //namespace apache
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_DATEPARSEEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_DATEPARSEEXCEPTION_HPP_IMPL
#define J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_DATEPARSEEXCEPTION_HPP_IMPL

namespace j2cpp {



org::apache::http::impl::cookie::DateParseException::operator local_ref<java::lang::Exception>() const
{
	return local_ref<java::lang::Exception>(get_jobject());
}


org::apache::http::impl::cookie::DateParseException::DateParseException()
: object<org::apache::http::impl::cookie::DateParseException>(
	call_new_object<
		org::apache::http::impl::cookie::DateParseException::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::DateParseException::J2CPP_METHOD_NAME(0),
		org::apache::http::impl::cookie::DateParseException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



org::apache::http::impl::cookie::DateParseException::DateParseException(local_ref< java::lang::String > const &a0)
: object<org::apache::http::impl::cookie::DateParseException>(
	call_new_object<
		org::apache::http::impl::cookie::DateParseException::J2CPP_CLASS_NAME,
		org::apache::http::impl::cookie::DateParseException::J2CPP_METHOD_NAME(1),
		org::apache::http::impl::cookie::DateParseException::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(org::apache::http::impl::cookie::DateParseException,"org/apache/http/impl/cookie/DateParseException")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::DateParseException,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::cookie::DateParseException,1,"<init>","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_COOKIE_DATEPARSEEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
