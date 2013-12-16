/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.StringBufferInputStream
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_STRINGBUFFERINPUTSTREAM_HPP_DECL
#define J2CPP_JAVA_IO_STRINGBUFFERINPUTSTREAM_HPP_DECL


namespace j2cpp { namespace java { namespace io { class InputStream; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/InputStream.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace io {

	class StringBufferInputStream;
	class StringBufferInputStream
		: public object<StringBufferInputStream>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)
		J2CPP_DECLARE_FIELD(0)
		J2CPP_DECLARE_FIELD(1)
		J2CPP_DECLARE_FIELD(2)

		explicit StringBufferInputStream(jobject jobj)
		: object<StringBufferInputStream>(jobj)
		{
		}

		operator local_ref<java::io::InputStream>() const;


		StringBufferInputStream(local_ref< java::lang::String > const&);
		jint available();
		jint read();
		jint read(local_ref< array<jbyte,1> >  const&, jint, jint);
		void reset();
		jlong skip(jlong);

	}; //class StringBufferInputStream

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_STRINGBUFFERINPUTSTREAM_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_STRINGBUFFERINPUTSTREAM_HPP_IMPL
#define J2CPP_JAVA_IO_STRINGBUFFERINPUTSTREAM_HPP_IMPL

namespace j2cpp {



java::io::StringBufferInputStream::operator local_ref<java::io::InputStream>() const
{
	return local_ref<java::io::InputStream>(get_jobject());
}


java::io::StringBufferInputStream::StringBufferInputStream(local_ref< java::lang::String > const &a0)
: object<java::io::StringBufferInputStream>(
	call_new_object<
		java::io::StringBufferInputStream::J2CPP_CLASS_NAME,
		java::io::StringBufferInputStream::J2CPP_METHOD_NAME(0),
		java::io::StringBufferInputStream::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}


jint java::io::StringBufferInputStream::available()
{
	return call_method<
		java::io::StringBufferInputStream::J2CPP_CLASS_NAME,
		java::io::StringBufferInputStream::J2CPP_METHOD_NAME(1),
		java::io::StringBufferInputStream::J2CPP_METHOD_SIGNATURE(1), 
		jint
	>(get_jobject());
}

jint java::io::StringBufferInputStream::read()
{
	return call_method<
		java::io::StringBufferInputStream::J2CPP_CLASS_NAME,
		java::io::StringBufferInputStream::J2CPP_METHOD_NAME(2),
		java::io::StringBufferInputStream::J2CPP_METHOD_SIGNATURE(2), 
		jint
	>(get_jobject());
}

jint java::io::StringBufferInputStream::read(local_ref< array<jbyte,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::StringBufferInputStream::J2CPP_CLASS_NAME,
		java::io::StringBufferInputStream::J2CPP_METHOD_NAME(3),
		java::io::StringBufferInputStream::J2CPP_METHOD_SIGNATURE(3), 
		jint
	>(get_jobject(), a0, a1, a2);
}

void java::io::StringBufferInputStream::reset()
{
	return call_method<
		java::io::StringBufferInputStream::J2CPP_CLASS_NAME,
		java::io::StringBufferInputStream::J2CPP_METHOD_NAME(4),
		java::io::StringBufferInputStream::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject());
}

jlong java::io::StringBufferInputStream::skip(jlong a0)
{
	return call_method<
		java::io::StringBufferInputStream::J2CPP_CLASS_NAME,
		java::io::StringBufferInputStream::J2CPP_METHOD_NAME(5),
		java::io::StringBufferInputStream::J2CPP_METHOD_SIGNATURE(5), 
		jlong
	>(get_jobject(), a0);
}



J2CPP_DEFINE_CLASS(java::io::StringBufferInputStream,"java/io/StringBufferInputStream")
J2CPP_DEFINE_METHOD(java::io::StringBufferInputStream,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::io::StringBufferInputStream,1,"available","()I")
J2CPP_DEFINE_METHOD(java::io::StringBufferInputStream,2,"read","()I")
J2CPP_DEFINE_METHOD(java::io::StringBufferInputStream,3,"read","([BII)I")
J2CPP_DEFINE_METHOD(java::io::StringBufferInputStream,4,"reset","()V")
J2CPP_DEFINE_METHOD(java::io::StringBufferInputStream,5,"skip","(J)J")
J2CPP_DEFINE_FIELD(java::io::StringBufferInputStream,0,"buffer","Ljava/lang/String;")
J2CPP_DEFINE_FIELD(java::io::StringBufferInputStream,1,"count","I")
J2CPP_DEFINE_FIELD(java::io::StringBufferInputStream,2,"pos","I")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_STRINGBUFFERINPUTSTREAM_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
