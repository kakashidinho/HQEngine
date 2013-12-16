/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.net.URI
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NET_URI_HPP_DECL
#define J2CPP_JAVA_NET_URI_HPP_DECL


namespace j2cpp { namespace java { namespace net { class URL; } } }
namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Comparable; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Comparable.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/net/URL.hpp>


namespace j2cpp {

namespace java { namespace net {

	class URI;
	class URI
		: public object<URI>
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

		explicit URI(jobject jobj)
		: object<URI>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::lang::Comparable>() const;
		operator local_ref<java::io::Serializable>() const;


		URI(local_ref< java::lang::String > const&);
		URI(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
		URI(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, jint, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
		URI(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
		URI(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
		jint compareTo(local_ref< java::net::URI >  const&);
		static local_ref< java::net::URI > create(local_ref< java::lang::String >  const&);
		jboolean equals(local_ref< java::lang::Object >  const&);
		local_ref< java::lang::String > getAuthority();
		local_ref< java::lang::String > getFragment();
		local_ref< java::lang::String > getHost();
		local_ref< java::lang::String > getPath();
		jint getPort();
		local_ref< java::lang::String > getQuery();
		local_ref< java::lang::String > getRawAuthority();
		local_ref< java::lang::String > getRawFragment();
		local_ref< java::lang::String > getRawPath();
		local_ref< java::lang::String > getRawQuery();
		local_ref< java::lang::String > getRawSchemeSpecificPart();
		local_ref< java::lang::String > getRawUserInfo();
		local_ref< java::lang::String > getScheme();
		local_ref< java::lang::String > getSchemeSpecificPart();
		local_ref< java::lang::String > getUserInfo();
		jint hashCode();
		jboolean isAbsolute();
		jboolean isOpaque();
		local_ref< java::net::URI > normalize();
		local_ref< java::net::URI > parseServerAuthority();
		local_ref< java::net::URI > relativize(local_ref< java::net::URI >  const&);
		local_ref< java::net::URI > resolve(local_ref< java::net::URI >  const&);
		local_ref< java::net::URI > resolve(local_ref< java::lang::String >  const&);
		local_ref< java::lang::String > toASCIIString();
		local_ref< java::lang::String > toString();
		local_ref< java::net::URL > toURL();
		jint compareTo(local_ref< java::lang::Object >  const&);
	}; //class URI

} //namespace net
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_NET_URI_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NET_URI_HPP_IMPL
#define J2CPP_JAVA_NET_URI_HPP_IMPL

namespace j2cpp {



java::net::URI::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::net::URI::operator local_ref<java::lang::Comparable>() const
{
	return local_ref<java::lang::Comparable>(get_jobject());
}

java::net::URI::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::net::URI::URI(local_ref< java::lang::String > const &a0)
: object<java::net::URI>(
	call_new_object<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(0),
		java::net::URI::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



java::net::URI::URI(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2)
: object<java::net::URI>(
	call_new_object<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(1),
		java::net::URI::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1, a2)
)
{
}



java::net::URI::URI(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2, jint a3, local_ref< java::lang::String > const &a4, local_ref< java::lang::String > const &a5, local_ref< java::lang::String > const &a6)
: object<java::net::URI>(
	call_new_object<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(2),
		java::net::URI::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1, a2, a3, a4, a5, a6)
)
{
}



java::net::URI::URI(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2, local_ref< java::lang::String > const &a3)
: object<java::net::URI>(
	call_new_object<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(3),
		java::net::URI::J2CPP_METHOD_SIGNATURE(3)
	>(a0, a1, a2, a3)
)
{
}



java::net::URI::URI(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2, local_ref< java::lang::String > const &a3, local_ref< java::lang::String > const &a4)
: object<java::net::URI>(
	call_new_object<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(4),
		java::net::URI::J2CPP_METHOD_SIGNATURE(4)
	>(a0, a1, a2, a3, a4)
)
{
}


jint java::net::URI::compareTo(local_ref< java::net::URI > const &a0)
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(5),
		java::net::URI::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject(), a0);
}

local_ref< java::net::URI > java::net::URI::create(local_ref< java::lang::String > const &a0)
{
	return call_static_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(6),
		java::net::URI::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::net::URI >
	>(a0);
}

jboolean java::net::URI::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(7),
		java::net::URI::J2CPP_METHOD_SIGNATURE(7), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::String > java::net::URI::getAuthority()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(8),
		java::net::URI::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getFragment()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(9),
		java::net::URI::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getHost()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(10),
		java::net::URI::J2CPP_METHOD_SIGNATURE(10), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getPath()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(11),
		java::net::URI::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::lang::String >
	>(get_jobject());
}

jint java::net::URI::getPort()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(12),
		java::net::URI::J2CPP_METHOD_SIGNATURE(12), 
		jint
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getQuery()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(13),
		java::net::URI::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getRawAuthority()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(14),
		java::net::URI::J2CPP_METHOD_SIGNATURE(14), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getRawFragment()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(15),
		java::net::URI::J2CPP_METHOD_SIGNATURE(15), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getRawPath()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(16),
		java::net::URI::J2CPP_METHOD_SIGNATURE(16), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getRawQuery()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(17),
		java::net::URI::J2CPP_METHOD_SIGNATURE(17), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getRawSchemeSpecificPart()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(18),
		java::net::URI::J2CPP_METHOD_SIGNATURE(18), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getRawUserInfo()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(19),
		java::net::URI::J2CPP_METHOD_SIGNATURE(19), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getScheme()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(20),
		java::net::URI::J2CPP_METHOD_SIGNATURE(20), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getSchemeSpecificPart()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(21),
		java::net::URI::J2CPP_METHOD_SIGNATURE(21), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::getUserInfo()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(22),
		java::net::URI::J2CPP_METHOD_SIGNATURE(22), 
		local_ref< java::lang::String >
	>(get_jobject());
}

jint java::net::URI::hashCode()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(23),
		java::net::URI::J2CPP_METHOD_SIGNATURE(23), 
		jint
	>(get_jobject());
}

jboolean java::net::URI::isAbsolute()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(24),
		java::net::URI::J2CPP_METHOD_SIGNATURE(24), 
		jboolean
	>(get_jobject());
}

jboolean java::net::URI::isOpaque()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(25),
		java::net::URI::J2CPP_METHOD_SIGNATURE(25), 
		jboolean
	>(get_jobject());
}

local_ref< java::net::URI > java::net::URI::normalize()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(26),
		java::net::URI::J2CPP_METHOD_SIGNATURE(26), 
		local_ref< java::net::URI >
	>(get_jobject());
}

local_ref< java::net::URI > java::net::URI::parseServerAuthority()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(27),
		java::net::URI::J2CPP_METHOD_SIGNATURE(27), 
		local_ref< java::net::URI >
	>(get_jobject());
}

local_ref< java::net::URI > java::net::URI::relativize(local_ref< java::net::URI > const &a0)
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(28),
		java::net::URI::J2CPP_METHOD_SIGNATURE(28), 
		local_ref< java::net::URI >
	>(get_jobject(), a0);
}

local_ref< java::net::URI > java::net::URI::resolve(local_ref< java::net::URI > const &a0)
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(29),
		java::net::URI::J2CPP_METHOD_SIGNATURE(29), 
		local_ref< java::net::URI >
	>(get_jobject(), a0);
}

local_ref< java::net::URI > java::net::URI::resolve(local_ref< java::lang::String > const &a0)
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(30),
		java::net::URI::J2CPP_METHOD_SIGNATURE(30), 
		local_ref< java::net::URI >
	>(get_jobject(), a0);
}

local_ref< java::lang::String > java::net::URI::toASCIIString()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(31),
		java::net::URI::J2CPP_METHOD_SIGNATURE(31), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::net::URI::toString()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(32),
		java::net::URI::J2CPP_METHOD_SIGNATURE(32), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::net::URL > java::net::URI::toURL()
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(33),
		java::net::URI::J2CPP_METHOD_SIGNATURE(33), 
		local_ref< java::net::URL >
	>(get_jobject());
}

jint java::net::URI::compareTo(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::net::URI::J2CPP_CLASS_NAME,
		java::net::URI::J2CPP_METHOD_NAME(34),
		java::net::URI::J2CPP_METHOD_SIGNATURE(34), 
		jint
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::net::URI,"java/net/URI")
J2CPP_DEFINE_METHOD(java::net::URI,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::net::URI,1,"<init>","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::net::URI,2,"<init>","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::net::URI,3,"<init>","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::net::URI,4,"<init>","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::net::URI,5,"compareTo","(Ljava/net/URI;)I")
J2CPP_DEFINE_METHOD(java::net::URI,6,"create","(Ljava/lang/String;)Ljava/net/URI;")
J2CPP_DEFINE_METHOD(java::net::URI,7,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::net::URI,8,"getAuthority","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,9,"getFragment","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,10,"getHost","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,11,"getPath","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,12,"getPort","()I")
J2CPP_DEFINE_METHOD(java::net::URI,13,"getQuery","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,14,"getRawAuthority","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,15,"getRawFragment","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,16,"getRawPath","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,17,"getRawQuery","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,18,"getRawSchemeSpecificPart","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,19,"getRawUserInfo","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,20,"getScheme","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,21,"getSchemeSpecificPart","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,22,"getUserInfo","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,23,"hashCode","()I")
J2CPP_DEFINE_METHOD(java::net::URI,24,"isAbsolute","()Z")
J2CPP_DEFINE_METHOD(java::net::URI,25,"isOpaque","()Z")
J2CPP_DEFINE_METHOD(java::net::URI,26,"normalize","()Ljava/net/URI;")
J2CPP_DEFINE_METHOD(java::net::URI,27,"parseServerAuthority","()Ljava/net/URI;")
J2CPP_DEFINE_METHOD(java::net::URI,28,"relativize","(Ljava/net/URI;)Ljava/net/URI;")
J2CPP_DEFINE_METHOD(java::net::URI,29,"resolve","(Ljava/net/URI;)Ljava/net/URI;")
J2CPP_DEFINE_METHOD(java::net::URI,30,"resolve","(Ljava/lang/String;)Ljava/net/URI;")
J2CPP_DEFINE_METHOD(java::net::URI,31,"toASCIIString","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,32,"toString","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::net::URI,33,"toURL","()Ljava/net/URL;")
J2CPP_DEFINE_METHOD(java::net::URI,34,"compareTo","(Ljava/lang/Object;)I")

} //namespace j2cpp

#endif //J2CPP_JAVA_NET_URI_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
