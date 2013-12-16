/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.FilterReader
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_FILTERREADER_HPP_DECL
#define J2CPP_JAVA_IO_FILTERREADER_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Reader; } } }


#include <java/io/Reader.hpp>


namespace j2cpp {

namespace java { namespace io {

	class FilterReader;
	class FilterReader
		: public object<FilterReader>
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
		J2CPP_DECLARE_FIELD(0)

		explicit FilterReader(jobject jobj)
		: object<FilterReader>(jobj)
		{
		}

		operator local_ref<java::io::Reader>() const;


		void close();
		void mark(jint);
		jboolean markSupported();
		jint read();
		jint read(local_ref< array<jchar,1> >  const&, jint, jint);
		jboolean ready();
		void reset();
		jlong skip(jlong);

	}; //class FilterReader

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_FILTERREADER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_FILTERREADER_HPP_IMPL
#define J2CPP_JAVA_IO_FILTERREADER_HPP_IMPL

namespace j2cpp {



java::io::FilterReader::operator local_ref<java::io::Reader>() const
{
	return local_ref<java::io::Reader>(get_jobject());
}


void java::io::FilterReader::close()
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(1),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject());
}

void java::io::FilterReader::mark(jint a0)
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(2),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}

jboolean java::io::FilterReader::markSupported()
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(3),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(3), 
		jboolean
	>(get_jobject());
}

jint java::io::FilterReader::read()
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(4),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

jint java::io::FilterReader::read(local_ref< array<jchar,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(5),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject(), a0, a1, a2);
}

jboolean java::io::FilterReader::ready()
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(6),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(6), 
		jboolean
	>(get_jobject());
}

void java::io::FilterReader::reset()
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(7),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(7), 
		void
	>(get_jobject());
}

jlong java::io::FilterReader::skip(jlong a0)
{
	return call_method<
		java::io::FilterReader::J2CPP_CLASS_NAME,
		java::io::FilterReader::J2CPP_METHOD_NAME(8),
		java::io::FilterReader::J2CPP_METHOD_SIGNATURE(8), 
		jlong
	>(get_jobject(), a0);
}



J2CPP_DEFINE_CLASS(java::io::FilterReader,"java/io/FilterReader")
J2CPP_DEFINE_METHOD(java::io::FilterReader,0,"<init>","(Ljava/io/Reader;)V")
J2CPP_DEFINE_METHOD(java::io::FilterReader,1,"close","()V")
J2CPP_DEFINE_METHOD(java::io::FilterReader,2,"mark","(I)V")
J2CPP_DEFINE_METHOD(java::io::FilterReader,3,"markSupported","()Z")
J2CPP_DEFINE_METHOD(java::io::FilterReader,4,"read","()I")
J2CPP_DEFINE_METHOD(java::io::FilterReader,5,"read","([CII)I")
J2CPP_DEFINE_METHOD(java::io::FilterReader,6,"ready","()Z")
J2CPP_DEFINE_METHOD(java::io::FilterReader,7,"reset","()V")
J2CPP_DEFINE_METHOD(java::io::FilterReader,8,"skip","(J)J")
J2CPP_DEFINE_FIELD(java::io::FilterReader,0,"in","Ljava/io/Reader;")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_FILTERREADER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
