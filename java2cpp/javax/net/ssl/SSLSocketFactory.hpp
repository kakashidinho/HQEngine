/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: javax.net.ssl.SSLSocketFactory
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_NET_SSL_SSLSOCKETFACTORY_HPP_DECL
#define J2CPP_JAVAX_NET_SSL_SSLSOCKETFACTORY_HPP_DECL


namespace j2cpp { namespace java { namespace net { class Socket; } } }
namespace j2cpp { namespace javax { namespace net { class SocketFactory; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/lang/String.hpp>
#include <java/net/Socket.hpp>
#include <javax/net/SocketFactory.hpp>


namespace j2cpp {

namespace javax { namespace net { namespace ssl {

	class SSLSocketFactory;
	class SSLSocketFactory
		: public object<SSLSocketFactory>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)

		explicit SSLSocketFactory(jobject jobj)
		: object<SSLSocketFactory>(jobj)
		{
		}

		operator local_ref<javax::net::SocketFactory>() const;


		SSLSocketFactory();
		static local_ref< javax::net::SocketFactory > getDefault();
		local_ref< array< local_ref< java::lang::String >, 1> > getDefaultCipherSuites();
		local_ref< array< local_ref< java::lang::String >, 1> > getSupportedCipherSuites();
		local_ref< java::net::Socket > createSocket(local_ref< java::net::Socket >  const&, local_ref< java::lang::String >  const&, jint, jboolean);
	}; //class SSLSocketFactory

} //namespace ssl
} //namespace net
} //namespace javax

} //namespace j2cpp

#endif //J2CPP_JAVAX_NET_SSL_SSLSOCKETFACTORY_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_NET_SSL_SSLSOCKETFACTORY_HPP_IMPL
#define J2CPP_JAVAX_NET_SSL_SSLSOCKETFACTORY_HPP_IMPL

namespace j2cpp {



javax::net::ssl::SSLSocketFactory::operator local_ref<javax::net::SocketFactory>() const
{
	return local_ref<javax::net::SocketFactory>(get_jobject());
}


javax::net::ssl::SSLSocketFactory::SSLSocketFactory()
: object<javax::net::ssl::SSLSocketFactory>(
	call_new_object<
		javax::net::ssl::SSLSocketFactory::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_NAME(0),
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}


local_ref< javax::net::SocketFactory > javax::net::ssl::SSLSocketFactory::getDefault()
{
	return call_static_method<
		javax::net::ssl::SSLSocketFactory::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_NAME(1),
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< javax::net::SocketFactory >
	>();
}

local_ref< array< local_ref< java::lang::String >, 1> > javax::net::ssl::SSLSocketFactory::getDefaultCipherSuites()
{
	return call_method<
		javax::net::ssl::SSLSocketFactory::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_NAME(2),
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject());
}

local_ref< array< local_ref< java::lang::String >, 1> > javax::net::ssl::SSLSocketFactory::getSupportedCipherSuites()
{
	return call_method<
		javax::net::ssl::SSLSocketFactory::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_NAME(3),
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject());
}

local_ref< java::net::Socket > javax::net::ssl::SSLSocketFactory::createSocket(local_ref< java::net::Socket > const &a0, local_ref< java::lang::String > const &a1, jint a2, jboolean a3)
{
	return call_method<
		javax::net::ssl::SSLSocketFactory::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_NAME(4),
		javax::net::ssl::SSLSocketFactory::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::net::Socket >
	>(get_jobject(), a0, a1, a2, a3);
}


J2CPP_DEFINE_CLASS(javax::net::ssl::SSLSocketFactory,"javax/net/ssl/SSLSocketFactory")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLSocketFactory,0,"<init>","()V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLSocketFactory,1,"getDefault","()Ljavax/net/SocketFactory;")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLSocketFactory,2,"getDefaultCipherSuites","()[java.lang.String")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLSocketFactory,3,"getSupportedCipherSuites","()[java.lang.String")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLSocketFactory,4,"createSocket","(Ljava/net/Socket;Ljava/lang/String;IZ)Ljava/net/Socket;")

} //namespace j2cpp

#endif //J2CPP_JAVAX_NET_SSL_SSLSOCKETFACTORY_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
