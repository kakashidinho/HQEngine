/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.LineNumberReader
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_LINENUMBERREADER_HPP_DECL
#define J2CPP_JAVA_IO_LINENUMBERREADER_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Reader; } } }
namespace j2cpp { namespace java { namespace io { class BufferedReader; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/BufferedReader.hpp>
#include <java/io/Reader.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace io {

	class LineNumberReader;
	class LineNumberReader
		: public object<LineNumberReader>
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

		explicit LineNumberReader(jobject jobj)
		: object<LineNumberReader>(jobj)
		{
		}

		operator local_ref<java::io::BufferedReader>() const;


		LineNumberReader(local_ref< java::io::Reader > const&);
		LineNumberReader(local_ref< java::io::Reader > const&, jint);
		jint getLineNumber();
		void mark(jint);
		jint read();
		jint read(local_ref< array<jchar,1> >  const&, jint, jint);
		local_ref< java::lang::String > readLine();
		void reset();
		void setLineNumber(jint);
		jlong skip(jlong);
	}; //class LineNumberReader

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_LINENUMBERREADER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_LINENUMBERREADER_HPP_IMPL
#define J2CPP_JAVA_IO_LINENUMBERREADER_HPP_IMPL

namespace j2cpp {



java::io::LineNumberReader::operator local_ref<java::io::BufferedReader>() const
{
	return local_ref<java::io::BufferedReader>(get_jobject());
}


java::io::LineNumberReader::LineNumberReader(local_ref< java::io::Reader > const &a0)
: object<java::io::LineNumberReader>(
	call_new_object<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(0),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



java::io::LineNumberReader::LineNumberReader(local_ref< java::io::Reader > const &a0, jint a1)
: object<java::io::LineNumberReader>(
	call_new_object<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(1),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}


jint java::io::LineNumberReader::getLineNumber()
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(2),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(2), 
		jint
	>(get_jobject());
}

void java::io::LineNumberReader::mark(jint a0)
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(3),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

jint java::io::LineNumberReader::read()
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(4),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

jint java::io::LineNumberReader::read(local_ref< array<jchar,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(5),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject(), a0, a1, a2);
}

local_ref< java::lang::String > java::io::LineNumberReader::readLine()
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(6),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::lang::String >
	>(get_jobject());
}

void java::io::LineNumberReader::reset()
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(7),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(7), 
		void
	>(get_jobject());
}

void java::io::LineNumberReader::setLineNumber(jint a0)
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(8),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

jlong java::io::LineNumberReader::skip(jlong a0)
{
	return call_method<
		java::io::LineNumberReader::J2CPP_CLASS_NAME,
		java::io::LineNumberReader::J2CPP_METHOD_NAME(9),
		java::io::LineNumberReader::J2CPP_METHOD_SIGNATURE(9), 
		jlong
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::io::LineNumberReader,"java/io/LineNumberReader")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,0,"<init>","(Ljava/io/Reader;)V")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,1,"<init>","(Ljava/io/Reader;I)V")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,2,"getLineNumber","()I")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,3,"mark","(I)V")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,4,"read","()I")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,5,"read","([CII)I")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,6,"readLine","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,7,"reset","()V")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,8,"setLineNumber","(I)V")
J2CPP_DEFINE_METHOD(java::io::LineNumberReader,9,"skip","(J)J")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_LINENUMBERREADER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION