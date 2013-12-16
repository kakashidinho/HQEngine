/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.BufferedReader
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_BUFFEREDREADER_HPP_DECL
#define J2CPP_JAVA_IO_BUFFEREDREADER_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Reader; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/Reader.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace io {

	class BufferedReader;
	class BufferedReader
		: public object<BufferedReader>
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

		explicit BufferedReader(jobject jobj)
		: object<BufferedReader>(jobj)
		{
		}

		operator local_ref<java::io::Reader>() const;


		BufferedReader(local_ref< java::io::Reader > const&);
		BufferedReader(local_ref< java::io::Reader > const&, jint);
		void close();
		void mark(jint);
		jboolean markSupported();
		jint read();
		jint read(local_ref< array<jchar,1> >  const&, jint, jint);
		local_ref< java::lang::String > readLine();
		jboolean ready();
		void reset();
		jlong skip(jlong);
	}; //class BufferedReader

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_BUFFEREDREADER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_BUFFEREDREADER_HPP_IMPL
#define J2CPP_JAVA_IO_BUFFEREDREADER_HPP_IMPL

namespace j2cpp {



java::io::BufferedReader::operator local_ref<java::io::Reader>() const
{
	return local_ref<java::io::Reader>(get_jobject());
}


java::io::BufferedReader::BufferedReader(local_ref< java::io::Reader > const &a0)
: object<java::io::BufferedReader>(
	call_new_object<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(0),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



java::io::BufferedReader::BufferedReader(local_ref< java::io::Reader > const &a0, jint a1)
: object<java::io::BufferedReader>(
	call_new_object<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(1),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}


void java::io::BufferedReader::close()
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(2),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject());
}

void java::io::BufferedReader::mark(jint a0)
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(3),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

jboolean java::io::BufferedReader::markSupported()
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(4),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(4), 
		jboolean
	>(get_jobject());
}

jint java::io::BufferedReader::read()
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(5),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject());
}

jint java::io::BufferedReader::read(local_ref< array<jchar,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(6),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(6), 
		jint
	>(get_jobject(), a0, a1, a2);
}

local_ref< java::lang::String > java::io::BufferedReader::readLine()
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(7),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::lang::String >
	>(get_jobject());
}

jboolean java::io::BufferedReader::ready()
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(8),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(8), 
		jboolean
	>(get_jobject());
}

void java::io::BufferedReader::reset()
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(9),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject());
}

jlong java::io::BufferedReader::skip(jlong a0)
{
	return call_method<
		java::io::BufferedReader::J2CPP_CLASS_NAME,
		java::io::BufferedReader::J2CPP_METHOD_NAME(10),
		java::io::BufferedReader::J2CPP_METHOD_SIGNATURE(10), 
		jlong
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::io::BufferedReader,"java/io/BufferedReader")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,0,"<init>","(Ljava/io/Reader;)V")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,1,"<init>","(Ljava/io/Reader;I)V")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,2,"close","()V")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,3,"mark","(I)V")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,4,"markSupported","()Z")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,5,"read","()I")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,6,"read","([CII)I")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,7,"readLine","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,8,"ready","()Z")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,9,"reset","()V")
J2CPP_DEFINE_METHOD(java::io::BufferedReader,10,"skip","(J)J")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_BUFFEREDREADER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
