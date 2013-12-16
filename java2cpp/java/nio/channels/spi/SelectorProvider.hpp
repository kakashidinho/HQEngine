/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.nio.channels.spi.SelectorProvider
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NIO_CHANNELS_SPI_SELECTORPROVIDER_HPP_DECL
#define J2CPP_JAVA_NIO_CHANNELS_SPI_SELECTORPROVIDER_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { class Pipe; } } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { namespace spi { class AbstractSelector; } } } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { class Channel; } } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { class SocketChannel; } } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { class ServerSocketChannel; } } } }
namespace j2cpp { namespace java { namespace nio { namespace channels { class DatagramChannel; } } } }


#include <java/lang/Object.hpp>
#include <java/nio/channels/Channel.hpp>
#include <java/nio/channels/DatagramChannel.hpp>
#include <java/nio/channels/Pipe.hpp>
#include <java/nio/channels/ServerSocketChannel.hpp>
#include <java/nio/channels/SocketChannel.hpp>
#include <java/nio/channels/spi/AbstractSelector.hpp>


namespace j2cpp {

namespace java { namespace nio { namespace channels { namespace spi {

	class SelectorProvider;
	class SelectorProvider
		: public object<SelectorProvider>
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

		explicit SelectorProvider(jobject jobj)
		: object<SelectorProvider>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		static local_ref< java::nio::channels::spi::SelectorProvider > provider();
		local_ref< java::nio::channels::DatagramChannel > openDatagramChannel();
		local_ref< java::nio::channels::Pipe > openPipe();
		local_ref< java::nio::channels::spi::AbstractSelector > openSelector();
		local_ref< java::nio::channels::ServerSocketChannel > openServerSocketChannel();
		local_ref< java::nio::channels::SocketChannel > openSocketChannel();
		local_ref< java::nio::channels::Channel > inheritedChannel();
	}; //class SelectorProvider

} //namespace spi
} //namespace channels
} //namespace nio
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_NIO_CHANNELS_SPI_SELECTORPROVIDER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NIO_CHANNELS_SPI_SELECTORPROVIDER_HPP_IMPL
#define J2CPP_JAVA_NIO_CHANNELS_SPI_SELECTORPROVIDER_HPP_IMPL

namespace j2cpp {



java::nio::channels::spi::SelectorProvider::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}


local_ref< java::nio::channels::spi::SelectorProvider > java::nio::channels::spi::SelectorProvider::provider()
{
	return call_static_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(1),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::nio::channels::spi::SelectorProvider >
	>();
}

local_ref< java::nio::channels::DatagramChannel > java::nio::channels::spi::SelectorProvider::openDatagramChannel()
{
	return call_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(2),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::nio::channels::DatagramChannel >
	>(get_jobject());
}

local_ref< java::nio::channels::Pipe > java::nio::channels::spi::SelectorProvider::openPipe()
{
	return call_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(3),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::nio::channels::Pipe >
	>(get_jobject());
}

local_ref< java::nio::channels::spi::AbstractSelector > java::nio::channels::spi::SelectorProvider::openSelector()
{
	return call_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(4),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::nio::channels::spi::AbstractSelector >
	>(get_jobject());
}

local_ref< java::nio::channels::ServerSocketChannel > java::nio::channels::spi::SelectorProvider::openServerSocketChannel()
{
	return call_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(5),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(5), 
		local_ref< java::nio::channels::ServerSocketChannel >
	>(get_jobject());
}

local_ref< java::nio::channels::SocketChannel > java::nio::channels::spi::SelectorProvider::openSocketChannel()
{
	return call_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(6),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::nio::channels::SocketChannel >
	>(get_jobject());
}

local_ref< java::nio::channels::Channel > java::nio::channels::spi::SelectorProvider::inheritedChannel()
{
	return call_method<
		java::nio::channels::spi::SelectorProvider::J2CPP_CLASS_NAME,
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_NAME(7),
		java::nio::channels::spi::SelectorProvider::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::nio::channels::Channel >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::nio::channels::spi::SelectorProvider,"java/nio/channels/spi/SelectorProvider")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,1,"provider","()Ljava/nio/channels/spi/SelectorProvider;")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,2,"openDatagramChannel","()Ljava/nio/channels/DatagramChannel;")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,3,"openPipe","()Ljava/nio/channels/Pipe;")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,4,"openSelector","()Ljava/nio/channels/spi/AbstractSelector;")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,5,"openServerSocketChannel","()Ljava/nio/channels/ServerSocketChannel;")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,6,"openSocketChannel","()Ljava/nio/channels/SocketChannel;")
J2CPP_DEFINE_METHOD(java::nio::channels::spi::SelectorProvider,7,"inheritedChannel","()Ljava/nio/channels/Channel;")

} //namespace j2cpp

#endif //J2CPP_JAVA_NIO_CHANNELS_SPI_SELECTORPROVIDER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
