/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.Properties
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_PROPERTIES_HPP_DECL
#define J2CPP_JAVA_UTIL_PROPERTIES_HPP_DECL


namespace j2cpp { namespace java { namespace io { class PrintStream; } } }
namespace j2cpp { namespace java { namespace io { class PrintWriter; } } }
namespace j2cpp { namespace java { namespace io { class InputStream; } } }
namespace j2cpp { namespace java { namespace io { class OutputStream; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace util { class Enumeration; } } }
namespace j2cpp { namespace java { namespace util { class Hashtable; } } }


#include <java/io/InputStream.hpp>
#include <java/io/OutputStream.hpp>
#include <java/io/PrintStream.hpp>
#include <java/io/PrintWriter.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/util/Enumeration.hpp>
#include <java/util/Hashtable.hpp>


namespace j2cpp {

namespace java { namespace util {

	class Properties;
	class Properties
		: public object<Properties>
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
		J2CPP_DECLARE_FIELD(0)

		explicit Properties(jobject jobj)
		: object<Properties>(jobj)
		{
		}

		operator local_ref<java::util::Hashtable>() const;


		Properties();
		Properties(local_ref< java::util::Properties > const&);
		local_ref< java::lang::String > getProperty(local_ref< java::lang::String >  const&);
		local_ref< java::lang::String > getProperty(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void list(local_ref< java::io::PrintStream >  const&);
		void list(local_ref< java::io::PrintWriter >  const&);
		void load(local_ref< java::io::InputStream >  const&);
		local_ref< java::util::Enumeration > propertyNames();
		void save(local_ref< java::io::OutputStream >  const&, local_ref< java::lang::String >  const&);
		local_ref< java::lang::Object > setProperty(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void store(local_ref< java::io::OutputStream >  const&, local_ref< java::lang::String >  const&);
		void loadFromXML(local_ref< java::io::InputStream >  const&);
		void storeToXML(local_ref< java::io::OutputStream >  const&, local_ref< java::lang::String >  const&);
		void storeToXML(local_ref< java::io::OutputStream >  const&, local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);

	}; //class Properties

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_PROPERTIES_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_PROPERTIES_HPP_IMPL
#define J2CPP_JAVA_UTIL_PROPERTIES_HPP_IMPL

namespace j2cpp {



java::util::Properties::operator local_ref<java::util::Hashtable>() const
{
	return local_ref<java::util::Hashtable>(get_jobject());
}


java::util::Properties::Properties()
: object<java::util::Properties>(
	call_new_object<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(0),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::util::Properties::Properties(local_ref< java::util::Properties > const &a0)
: object<java::util::Properties>(
	call_new_object<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(1),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}


local_ref< java::lang::String > java::util::Properties::getProperty(local_ref< java::lang::String > const &a0)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(2),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::lang::String >
	>(get_jobject(), a0);
}

local_ref< java::lang::String > java::util::Properties::getProperty(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(3),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::lang::String >
	>(get_jobject(), a0, a1);
}

void java::util::Properties::list(local_ref< java::io::PrintStream > const &a0)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(4),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}

void java::util::Properties::list(local_ref< java::io::PrintWriter > const &a0)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(5),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0);
}

void java::util::Properties::load(local_ref< java::io::InputStream > const &a0)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(6),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject(), a0);
}

local_ref< java::util::Enumeration > java::util::Properties::propertyNames()
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(7),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::util::Enumeration >
	>(get_jobject());
}

void java::util::Properties::save(local_ref< java::io::OutputStream > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(8),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0, a1);
}

local_ref< java::lang::Object > java::util::Properties::setProperty(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(9),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0, a1);
}

void java::util::Properties::store(local_ref< java::io::OutputStream > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(10),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject(), a0, a1);
}

void java::util::Properties::loadFromXML(local_ref< java::io::InputStream > const &a0)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(11),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(11), 
		void
	>(get_jobject(), a0);
}

void java::util::Properties::storeToXML(local_ref< java::io::OutputStream > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(12),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(12), 
		void
	>(get_jobject(), a0, a1);
}

void java::util::Properties::storeToXML(local_ref< java::io::OutputStream > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2)
{
	return call_method<
		java::util::Properties::J2CPP_CLASS_NAME,
		java::util::Properties::J2CPP_METHOD_NAME(13),
		java::util::Properties::J2CPP_METHOD_SIGNATURE(13), 
		void
	>(get_jobject(), a0, a1, a2);
}



J2CPP_DEFINE_CLASS(java::util::Properties,"java/util/Properties")
J2CPP_DEFINE_METHOD(java::util::Properties,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::util::Properties,1,"<init>","(Ljava/util/Properties;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,2,"getProperty","(Ljava/lang/String;)Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::util::Properties,3,"getProperty","(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::util::Properties,4,"list","(Ljava/io/PrintStream;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,5,"list","(Ljava/io/PrintWriter;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,6,"load","(Ljava/io/InputStream;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,7,"propertyNames","()Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(java::util::Properties,8,"save","(Ljava/io/OutputStream;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,9,"setProperty","(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Properties,10,"store","(Ljava/io/OutputStream;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,11,"loadFromXML","(Ljava/io/InputStream;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,12,"storeToXML","(Ljava/io/OutputStream;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::util::Properties,13,"storeToXML","(Ljava/io/OutputStream;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_FIELD(java::util::Properties,0,"defaults","Ljava/util/Properties;")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_PROPERTIES_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION