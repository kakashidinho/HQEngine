/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.net.Socket
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NET_SOCKET_HPP_DECL
#define J2CPP_JAVA_NET_SOCKET_HPP_DECL


namespace j2cpp { namespace java { namespace net { class SocketAddress; } } }
namespace j2cpp { namespace java { namespace net { class InetAddress; } } }
namespace j2cpp { namespace java { namespace net { class Proxy; } } }
namespace j2cpp { namespace java { namespace net { class SocketImplFactory; } } }
namespace j2cpp { namespace java { namespace io { class InputStream; } } }
namespace j2cpp { namespace java { namespace io { class OutputStream; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { class SocketChannel; } } } }


#include <java/io/InputStream.hpp>
#include <java/io/OutputStream.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/net/InetAddress.hpp>
#include <java/net/Proxy.hpp>
#include <java/net/SocketAddress.hpp>
#include <java/net/SocketImplFactory.hpp>
#include <java/nio/channels/SocketChannel.hpp>


namespace j2cpp {

namespace java { namespace net {

	class Socket;
	class Socket
		: public object<Socket>
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
		J2CPP_DECLARE_METHOD(21)
		J2CPP_DECLARE_METHOD(22)
		J2CPP_DECLARE_METHOD(23)
		J2CPP_DECLARE_METHOD(24)
		J2CPP_DECLARE_METHOD(25)
		J2CPP_DECLARE_METHOD(26)
		J2CPP_DECLARE_METHOD(27)
		J2CPP_DECLARE_METHOD(28)
		J2CPP_DECLARE_METHOD(29)
		J2CPP_DECLARE_METHOD(30)
		J2CPP_DECLARE_METHOD(31)
		J2CPP_DECLARE_METHOD(32)
		J2CPP_DECLARE_METHOD(33)
		J2CPP_DECLARE_METHOD(34)
		J2CPP_DECLARE_METHOD(35)
		J2CPP_DECLARE_METHOD(36)
		J2CPP_DECLARE_METHOD(37)
		J2CPP_DECLARE_METHOD(38)
		J2CPP_DECLARE_METHOD(39)
		J2CPP_DECLARE_METHOD(40)
		J2CPP_DECLARE_METHOD(41)
		J2CPP_DECLARE_METHOD(42)
		J2CPP_DECLARE_METHOD(43)
		J2CPP_DECLARE_METHOD(44)
		J2CPP_DECLARE_METHOD(45)
		J2CPP_DECLARE_METHOD(46)
		J2CPP_DECLARE_METHOD(47)
		J2CPP_DECLARE_METHOD(48)
		J2CPP_DECLARE_METHOD(49)
		J2CPP_DECLARE_METHOD(50)

		explicit Socket(jobject jobj)
		: object<Socket>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		Socket();
		Socket(local_ref< java::net::Proxy > const&);
		Socket(local_ref< java::lang::String > const&, jint);
		Socket(local_ref< java::lang::String > const&, jint, local_ref< java::net::InetAddress > const&, jint);
		Socket(local_ref< java::lang::String > const&, jint, jboolean);
		Socket(local_ref< java::net::InetAddress > const&, jint);
		Socket(local_ref< java::net::InetAddress > const&, jint, local_ref< java::net::InetAddress > const&, jint);
		Socket(local_ref< java::net::InetAddress > const&, jint, jboolean);
		void close();
		local_ref< java::net::InetAddress > getInetAddress();
		local_ref< java::io::InputStream > getInputStream();
		jboolean getKeepAlive();
		local_ref< java::net::InetAddress > getLocalAddress();
		jint getLocalPort();
		local_ref< java::io::OutputStream > getOutputStream();
		jint getPort();
		jint getSoLinger();
		jint getReceiveBufferSize();
		jint getSendBufferSize();
		jint getSoTimeout();
		jboolean getTcpNoDelay();
		void setKeepAlive(jboolean);
		static void setSocketImplFactory(local_ref< java::net::SocketImplFactory >  const&);
		void setSendBufferSize(jint);
		void setReceiveBufferSize(jint);
		void setSoLinger(jboolean, jint);
		void setSoTimeout(jint);
		void setTcpNoDelay(jboolean);
		local_ref< java::lang::String > toString();
		void shutdownInput();
		void shutdownOutput();
		local_ref< java::net::SocketAddress > getLocalSocketAddress();
		local_ref< java::net::SocketAddress > getRemoteSocketAddress();
		jboolean isBound();
		jboolean isConnected();
		jboolean isClosed();
		void bind(local_ref< java::net::SocketAddress >  const&);
		void connect(local_ref< java::net::SocketAddress >  const&);
		void connect(local_ref< java::net::SocketAddress >  const&, jint);
		jboolean isInputShutdown();
		jboolean isOutputShutdown();
		void setReuseAddress(jboolean);
		jboolean getReuseAddress();
		void setOOBInline(jboolean);
		jboolean getOOBInline();
		void setTrafficClass(jint);
		jint getTrafficClass();
		void sendUrgentData(jint);
		local_ref< java::nio::channels::SocketChannel > getChannel();
		void setPerformancePreferences(jint, jint, jint);
	}; //class Socket

} //namespace net
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_NET_SOCKET_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NET_SOCKET_HPP_IMPL
#define J2CPP_JAVA_NET_SOCKET_HPP_IMPL

namespace j2cpp {



java::net::Socket::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}


java::net::Socket::Socket()
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(0),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::net::Socket::Socket(local_ref< java::net::Proxy > const &a0)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(1),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



java::net::Socket::Socket(local_ref< java::lang::String > const &a0, jint a1)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(2),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1)
)
{
}



java::net::Socket::Socket(local_ref< java::lang::String > const &a0, jint a1, local_ref< java::net::InetAddress > const &a2, jint a3)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(3),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(3)
	>(a0, a1, a2, a3)
)
{
}



java::net::Socket::Socket(local_ref< java::lang::String > const &a0, jint a1, jboolean a2)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(4),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(4)
	>(a0, a1, a2)
)
{
}



java::net::Socket::Socket(local_ref< java::net::InetAddress > const &a0, jint a1)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(5),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(5)
	>(a0, a1)
)
{
}



java::net::Socket::Socket(local_ref< java::net::InetAddress > const &a0, jint a1, local_ref< java::net::InetAddress > const &a2, jint a3)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(6),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(6)
	>(a0, a1, a2, a3)
)
{
}



java::net::Socket::Socket(local_ref< java::net::InetAddress > const &a0, jint a1, jboolean a2)
: object<java::net::Socket>(
	call_new_object<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(7),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(7)
	>(a0, a1, a2)
)
{
}



void java::net::Socket::close()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(9),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject());
}

local_ref< java::net::InetAddress > java::net::Socket::getInetAddress()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(10),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(10), 
		local_ref< java::net::InetAddress >
	>(get_jobject());
}

local_ref< java::io::InputStream > java::net::Socket::getInputStream()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(11),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::io::InputStream >
	>(get_jobject());
}

jboolean java::net::Socket::getKeepAlive()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(12),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(12), 
		jboolean
	>(get_jobject());
}

local_ref< java::net::InetAddress > java::net::Socket::getLocalAddress()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(13),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< java::net::InetAddress >
	>(get_jobject());
}

jint java::net::Socket::getLocalPort()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(14),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(14), 
		jint
	>(get_jobject());
}

local_ref< java::io::OutputStream > java::net::Socket::getOutputStream()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(15),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(15), 
		local_ref< java::io::OutputStream >
	>(get_jobject());
}

jint java::net::Socket::getPort()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(16),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(16), 
		jint
	>(get_jobject());
}

jint java::net::Socket::getSoLinger()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(17),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(17), 
		jint
	>(get_jobject());
}

jint java::net::Socket::getReceiveBufferSize()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(18),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(18), 
		jint
	>(get_jobject());
}

jint java::net::Socket::getSendBufferSize()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(19),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(19), 
		jint
	>(get_jobject());
}

jint java::net::Socket::getSoTimeout()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(20),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(20), 
		jint
	>(get_jobject());
}

jboolean java::net::Socket::getTcpNoDelay()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(21),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(21), 
		jboolean
	>(get_jobject());
}

void java::net::Socket::setKeepAlive(jboolean a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(22),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(22), 
		void
	>(get_jobject(), a0);
}

void java::net::Socket::setSocketImplFactory(local_ref< java::net::SocketImplFactory > const &a0)
{
	return call_static_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(23),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(23), 
		void
	>(a0);
}

void java::net::Socket::setSendBufferSize(jint a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(24),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(24), 
		void
	>(get_jobject(), a0);
}

void java::net::Socket::setReceiveBufferSize(jint a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(25),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(25), 
		void
	>(get_jobject(), a0);
}

void java::net::Socket::setSoLinger(jboolean a0, jint a1)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(26),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(26), 
		void
	>(get_jobject(), a0, a1);
}

void java::net::Socket::setSoTimeout(jint a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(27),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(27), 
		void
	>(get_jobject(), a0);
}

void java::net::Socket::setTcpNoDelay(jboolean a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(28),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(28), 
		void
	>(get_jobject(), a0);
}

local_ref< java::lang::String > java::net::Socket::toString()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(29),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(29), 
		local_ref< java::lang::String >
	>(get_jobject());
}

void java::net::Socket::shutdownInput()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(30),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(30), 
		void
	>(get_jobject());
}

void java::net::Socket::shutdownOutput()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(31),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(31), 
		void
	>(get_jobject());
}

local_ref< java::net::SocketAddress > java::net::Socket::getLocalSocketAddress()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(32),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(32), 
		local_ref< java::net::SocketAddress >
	>(get_jobject());
}

local_ref< java::net::SocketAddress > java::net::Socket::getRemoteSocketAddress()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(33),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(33), 
		local_ref< java::net::SocketAddress >
	>(get_jobject());
}

jboolean java::net::Socket::isBound()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(34),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(34), 
		jboolean
	>(get_jobject());
}

jboolean java::net::Socket::isConnected()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(35),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(35), 
		jboolean
	>(get_jobject());
}

jboolean java::net::Socket::isClosed()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(36),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(36), 
		jboolean
	>(get_jobject());
}

void java::net::Socket::bind(local_ref< java::net::SocketAddress > const &a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(37),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(37), 
		void
	>(get_jobject(), a0);
}

void java::net::Socket::connect(local_ref< java::net::SocketAddress > const &a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(38),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(38), 
		void
	>(get_jobject(), a0);
}

void java::net::Socket::connect(local_ref< java::net::SocketAddress > const &a0, jint a1)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(39),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(39), 
		void
	>(get_jobject(), a0, a1);
}

jboolean java::net::Socket::isInputShutdown()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(40),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(40), 
		jboolean
	>(get_jobject());
}

jboolean java::net::Socket::isOutputShutdown()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(41),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(41), 
		jboolean
	>(get_jobject());
}

void java::net::Socket::setReuseAddress(jboolean a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(42),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(42), 
		void
	>(get_jobject(), a0);
}

jboolean java::net::Socket::getReuseAddress()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(43),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(43), 
		jboolean
	>(get_jobject());
}

void java::net::Socket::setOOBInline(jboolean a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(44),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(44), 
		void
	>(get_jobject(), a0);
}

jboolean java::net::Socket::getOOBInline()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(45),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(45), 
		jboolean
	>(get_jobject());
}

void java::net::Socket::setTrafficClass(jint a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(46),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(46), 
		void
	>(get_jobject(), a0);
}

jint java::net::Socket::getTrafficClass()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(47),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(47), 
		jint
	>(get_jobject());
}

void java::net::Socket::sendUrgentData(jint a0)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(48),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(48), 
		void
	>(get_jobject(), a0);
}

local_ref< java::nio::channels::SocketChannel > java::net::Socket::getChannel()
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(49),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(49), 
		local_ref< java::nio::channels::SocketChannel >
	>(get_jobject());
}

void java::net::Socket::setPerformancePreferences(jint a0, jint a1, jint a2)
{
	return call_method<
		java::net::Socket::J2CPP_CLASS_NAME,
		java::net::Socket::J2CPP_METHOD_NAME(50),
		java::net::Socket::J2CPP_METHOD_SIGNATURE(50), 
		void
	>(get_jobject(), a0, a1, a2);
}


J2CPP_DEFINE_CLASS(java::net::Socket,"java/net/Socket")
J2CPP_DEFINE_METHOD(java::net::Socket,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::net::Socket,1,"<init>","(Ljava/net/Proxy;)V")
J2CPP_DEFINE_METHOD(java::net::Socket,2,"<init>","(Ljava/lang/String;I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,3,"<init>","(Ljava/lang/String;ILjava/net/InetAddress;I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,4,"<init>","(Ljava/lang/String;IZ)V")
J2CPP_DEFINE_METHOD(java::net::Socket,5,"<init>","(Ljava/net/InetAddress;I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,6,"<init>","(Ljava/net/InetAddress;ILjava/net/InetAddress;I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,7,"<init>","(Ljava/net/InetAddress;IZ)V")
J2CPP_DEFINE_METHOD(java::net::Socket,8,"<init>","(Ljava/net/SocketImpl;)V")
J2CPP_DEFINE_METHOD(java::net::Socket,9,"close","()V")
J2CPP_DEFINE_METHOD(java::net::Socket,10,"getInetAddress","()Ljava/net/InetAddress;")
J2CPP_DEFINE_METHOD(java::net::Socket,11,"getInputStream","()Ljava/io/InputStream;")
J2CPP_DEFINE_METHOD(java::net::Socket,12,"getKeepAlive","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,13,"getLocalAddress","()Ljava/net/InetAddress;")
J2CPP_DEFINE_METHOD(java::net::Socket,14,"getLocalPort","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,15,"getOutputStream","()Ljava/io/OutputStream;")
J2CPP_DEFINE_METHOD(java::net::Socket,16,"getPort","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,17,"getSoLinger","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,18,"getReceiveBufferSize","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,19,"getSendBufferSize","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,20,"getSoTimeout","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,21,"getTcpNoDelay","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,22,"setKeepAlive","(Z)V")
J2CPP_DEFINE_METHOD(java::net::Socket,23,"setSocketImplFactory","(Ljava/net/SocketImplFactory;)V")
J2CPP_DEFINE_METHOD(java::net::Socket,24,"setSendBufferSize","(I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,25,"setReceiveBufferSize","(I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,26,"setSoLinger","(ZI)V")
J2CPP_DEFINE_METHOD(java::net::Socket,27,"setSoTimeout","(I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,28,"setTcpNoDelay","(Z)V")
J2CPP_DEFINE_METHOD(java::net::Socket,29,"toString","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::Socket,30,"shutdownInput","()V")
J2CPP_DEFINE_METHOD(java::net::Socket,31,"shutdownOutput","()V")
J2CPP_DEFINE_METHOD(java::net::Socket,32,"getLocalSocketAddress","()Ljava/net/SocketAddress;")
J2CPP_DEFINE_METHOD(java::net::Socket,33,"getRemoteSocketAddress","()Ljava/net/SocketAddress;")
J2CPP_DEFINE_METHOD(java::net::Socket,34,"isBound","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,35,"isConnected","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,36,"isClosed","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,37,"bind","(Ljava/net/SocketAddress;)V")
J2CPP_DEFINE_METHOD(java::net::Socket,38,"connect","(Ljava/net/SocketAddress;)V")
J2CPP_DEFINE_METHOD(java::net::Socket,39,"connect","(Ljava/net/SocketAddress;I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,40,"isInputShutdown","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,41,"isOutputShutdown","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,42,"setReuseAddress","(Z)V")
J2CPP_DEFINE_METHOD(java::net::Socket,43,"getReuseAddress","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,44,"setOOBInline","(Z)V")
J2CPP_DEFINE_METHOD(java::net::Socket,45,"getOOBInline","()Z")
J2CPP_DEFINE_METHOD(java::net::Socket,46,"setTrafficClass","(I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,47,"getTrafficClass","()I")
J2CPP_DEFINE_METHOD(java::net::Socket,48,"sendUrgentData","(I)V")
J2CPP_DEFINE_METHOD(java::net::Socket,49,"getChannel","()Ljava/nio/channels/SocketChannel;")
J2CPP_DEFINE_METHOD(java::net::Socket,50,"setPerformancePreferences","(III)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_NET_SOCKET_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
