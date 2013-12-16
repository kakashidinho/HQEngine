/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: javax.net.ssl.SSLServerSocket
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_NET_SSL_SSLSERVERSOCKET_HPP_DECL
#define J2CPP_JAVAX_NET_SSL_SSLSERVERSOCKET_HPP_DECL


namespace j2cpp { namespace java { namespace net { class ServerSocket; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/lang/String.hpp>
#include <java/net/ServerSocket.hpp>


namespace j2cpp {

namespace javax { namespace net { namespace ssl {

	class SSLServerSocket;
	class SSLServerSocket
		: public object<SSLServerSocket>
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

		explicit SSLServerSocket(jobject jobj)
		: object<SSLServerSocket>(jobj)
		{
		}

		operator local_ref<java::net::ServerSocket>() const;


		local_ref< array< local_ref< java::lang::String >, 1> > getEnabledCipherSuites();
		void setEnabledCipherSuites(local_ref< array< local_ref< java::lang::String >, 1> >  const&);
		local_ref< array< local_ref< java::lang::String >, 1> > getSupportedCipherSuites();
		local_ref< array< local_ref< java::lang::String >, 1> > getSupportedProtocols();
		local_ref< array< local_ref< java::lang::String >, 1> > getEnabledProtocols();
		void setEnabledProtocols(local_ref< array< local_ref< java::lang::String >, 1> >  const&);
		void setNeedClientAuth(jboolean);
		jboolean getNeedClientAuth();
		void setWantClientAuth(jboolean);
		jboolean getWantClientAuth();
		void setUseClientMode(jboolean);
		jboolean getUseClientMode();
		void setEnableSessionCreation(jboolean);
		jboolean getEnableSessionCreation();
	}; //class SSLServerSocket

} //namespace ssl
} //namespace net
} //namespace javax

} //namespace j2cpp

#endif //J2CPP_JAVAX_NET_SSL_SSLSERVERSOCKET_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_NET_SSL_SSLSERVERSOCKET_HPP_IMPL
#define J2CPP_JAVAX_NET_SSL_SSLSERVERSOCKET_HPP_IMPL

namespace j2cpp {



javax::net::ssl::SSLServerSocket::operator local_ref<java::net::ServerSocket>() const
{
	return local_ref<java::net::ServerSocket>(get_jobject());
}





local_ref< array< local_ref< java::lang::String >, 1> > javax::net::ssl::SSLServerSocket::getEnabledCipherSuites()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(4),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject());
}

void javax::net::ssl::SSLServerSocket::setEnabledCipherSuites(local_ref< array< local_ref< java::lang::String >, 1> > const &a0)
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(5),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0);
}

local_ref< array< local_ref< java::lang::String >, 1> > javax::net::ssl::SSLServerSocket::getSupportedCipherSuites()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(6),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject());
}

local_ref< array< local_ref< java::lang::String >, 1> > javax::net::ssl::SSLServerSocket::getSupportedProtocols()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(7),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject());
}

local_ref< array< local_ref< java::lang::String >, 1> > javax::net::ssl::SSLServerSocket::getEnabledProtocols()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(8),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject());
}

void javax::net::ssl::SSLServerSocket::setEnabledProtocols(local_ref< array< local_ref< java::lang::String >, 1> > const &a0)
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(9),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0);
}

void javax::net::ssl::SSLServerSocket::setNeedClientAuth(jboolean a0)
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(10),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject(), a0);
}

jboolean javax::net::ssl::SSLServerSocket::getNeedClientAuth()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(11),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(11), 
		jboolean
	>(get_jobject());
}

void javax::net::ssl::SSLServerSocket::setWantClientAuth(jboolean a0)
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(12),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(12), 
		void
	>(get_jobject(), a0);
}

jboolean javax::net::ssl::SSLServerSocket::getWantClientAuth()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(13),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(13), 
		jboolean
	>(get_jobject());
}

void javax::net::ssl::SSLServerSocket::setUseClientMode(jboolean a0)
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(14),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(14), 
		void
	>(get_jobject(), a0);
}

jboolean javax::net::ssl::SSLServerSocket::getUseClientMode()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(15),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(15), 
		jboolean
	>(get_jobject());
}

void javax::net::ssl::SSLServerSocket::setEnableSessionCreation(jboolean a0)
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(16),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(16), 
		void
	>(get_jobject(), a0);
}

jboolean javax::net::ssl::SSLServerSocket::getEnableSessionCreation()
{
	return call_method<
		javax::net::ssl::SSLServerSocket::J2CPP_CLASS_NAME,
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_NAME(17),
		javax::net::ssl::SSLServerSocket::J2CPP_METHOD_SIGNATURE(17), 
		jboolean
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(javax::net::ssl::SSLServerSocket,"javax/net/ssl/SSLServerSocket")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,0,"<init>","()V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,1,"<init>","(I)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,2,"<init>","(II)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,3,"<init>","(IILjava/net/InetAddress;)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,4,"getEnabledCipherSuites","()[java.lang.String")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,5,"setEnabledCipherSuites","([java.lang.String)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,6,"getSupportedCipherSuites","()[java.lang.String")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,7,"getSupportedProtocols","()[java.lang.String")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,8,"getEnabledProtocols","()[java.lang.String")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,9,"setEnabledProtocols","([java.lang.String)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,10,"setNeedClientAuth","(Z)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,11,"getNeedClientAuth","()Z")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,12,"setWantClientAuth","(Z)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,13,"getWantClientAuth","()Z")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,14,"setUseClientMode","(Z)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,15,"getUseClientMode","()Z")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,16,"setEnableSessionCreation","(Z)V")
J2CPP_DEFINE_METHOD(javax::net::ssl::SSLServerSocket,17,"getEnableSessionCreation","()Z")

} //namespace j2cpp

#endif //J2CPP_JAVAX_NET_SSL_SSLSERVERSOCKET_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
