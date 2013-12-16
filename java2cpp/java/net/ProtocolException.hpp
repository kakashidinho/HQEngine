/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.net.ProtocolException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NET_PROTOCOLEXCEPTION_HPP_DECL
#define J2CPP_JAVA_NET_PROTOCOLEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace io { class IOException; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/IOException.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace net {

	class ProtocolException;
	class ProtocolException
		: public object<ProtocolException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit ProtocolException(jobject jobj)
		: object<ProtocolException>(jobj)
		{
		}

		operator local_ref<java::io::IOException>() const;


		ProtocolException();
		ProtocolException(local_ref< java::lang::String > const&);
	}; //class ProtocolException

} //namespace net
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_NET_PROTOCOLEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NET_PROTOCOLEXCEPTION_HPP_IMPL
#define J2CPP_JAVA_NET_PROTOCOLEXCEPTION_HPP_IMPL

namespace j2cpp {



java::net::ProtocolException::operator local_ref<java::io::IOException>() const
{
	return local_ref<java::io::IOException>(get_jobject());
}


java::net::ProtocolException::ProtocolException()
: object<java::net::ProtocolException>(
	call_new_object<
		java::net::ProtocolException::J2CPP_CLASS_NAME,
		java::net::ProtocolException::J2CPP_METHOD_NAME(0),
		java::net::ProtocolException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::net::ProtocolException::ProtocolException(local_ref< java::lang::String > const &a0)
: object<java::net::ProtocolException>(
	call_new_object<
		java::net::ProtocolException::J2CPP_CLASS_NAME,
		java::net::ProtocolException::J2CPP_METHOD_NAME(1),
		java::net::ProtocolException::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(java::net::ProtocolException,"java/net/ProtocolException")
J2CPP_DEFINE_METHOD(java::net::ProtocolException,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::net::ProtocolException,1,"<init>","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_NET_PROTOCOLEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
