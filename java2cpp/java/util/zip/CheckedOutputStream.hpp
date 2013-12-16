/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.zip.CheckedOutputStream
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_ZIP_CHECKEDOUTPUTSTREAM_HPP_DECL
#define J2CPP_JAVA_UTIL_ZIP_CHECKEDOUTPUTSTREAM_HPP_DECL


namespace j2cpp { namespace java { namespace io { class FilterOutputStream; } } }
namespace j2cpp { namespace java { namespace io { class OutputStream; } } }
namespace j2cpp { namespace java { namespace util { namespace zip { class Checksum; } } } }


#include <java/io/FilterOutputStream.hpp>
#include <java/io/OutputStream.hpp>
#include <java/util/zip/Checksum.hpp>


namespace j2cpp {

namespace java { namespace util { namespace zip {

	class CheckedOutputStream;
	class CheckedOutputStream
		: public object<CheckedOutputStream>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)

		explicit CheckedOutputStream(jobject jobj)
		: object<CheckedOutputStream>(jobj)
		{
		}

		operator local_ref<java::io::FilterOutputStream>() const;


		CheckedOutputStream(local_ref< java::io::OutputStream > const&, local_ref< java::util::zip::Checksum > const&);
		local_ref< java::util::zip::Checksum > getChecksum();
		void write(jint);
		void write(local_ref< array<jbyte,1> >  const&, jint, jint);
	}; //class CheckedOutputStream

} //namespace zip
} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_ZIP_CHECKEDOUTPUTSTREAM_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_ZIP_CHECKEDOUTPUTSTREAM_HPP_IMPL
#define J2CPP_JAVA_UTIL_ZIP_CHECKEDOUTPUTSTREAM_HPP_IMPL

namespace j2cpp {



java::util::zip::CheckedOutputStream::operator local_ref<java::io::FilterOutputStream>() const
{
	return local_ref<java::io::FilterOutputStream>(get_jobject());
}


java::util::zip::CheckedOutputStream::CheckedOutputStream(local_ref< java::io::OutputStream > const &a0, local_ref< java::util::zip::Checksum > const &a1)
: object<java::util::zip::CheckedOutputStream>(
	call_new_object<
		java::util::zip::CheckedOutputStream::J2CPP_CLASS_NAME,
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_NAME(0),
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}


local_ref< java::util::zip::Checksum > java::util::zip::CheckedOutputStream::getChecksum()
{
	return call_method<
		java::util::zip::CheckedOutputStream::J2CPP_CLASS_NAME,
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_NAME(1),
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::util::zip::Checksum >
	>(get_jobject());
}

void java::util::zip::CheckedOutputStream::write(jint a0)
{
	return call_method<
		java::util::zip::CheckedOutputStream::J2CPP_CLASS_NAME,
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_NAME(2),
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}

void java::util::zip::CheckedOutputStream::write(local_ref< array<jbyte,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::util::zip::CheckedOutputStream::J2CPP_CLASS_NAME,
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_NAME(3),
		java::util::zip::CheckedOutputStream::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0, a1, a2);
}


J2CPP_DEFINE_CLASS(java::util::zip::CheckedOutputStream,"java/util/zip/CheckedOutputStream")
J2CPP_DEFINE_METHOD(java::util::zip::CheckedOutputStream,0,"<init>","(Ljava/io/OutputStream;Ljava/util/zip/Checksum;)V")
J2CPP_DEFINE_METHOD(java::util::zip::CheckedOutputStream,1,"getChecksum","()Ljava/util/zip/Checksum;")
J2CPP_DEFINE_METHOD(java::util::zip::CheckedOutputStream,2,"write","(I)V")
J2CPP_DEFINE_METHOD(java::util::zip::CheckedOutputStream,3,"write","([BII)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_ZIP_CHECKEDOUTPUTSTREAM_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
