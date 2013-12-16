/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.ObjectOutput
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_OBJECTOUTPUT_HPP_DECL
#define J2CPP_JAVA_IO_OBJECTOUTPUT_HPP_DECL


namespace j2cpp { namespace java { namespace io { class DataOutput; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }


#include <java/io/DataOutput.hpp>
#include <java/lang/Object.hpp>


namespace j2cpp {

namespace java { namespace io {

	class ObjectOutput;
	class ObjectOutput
		: public object<ObjectOutput>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)

		explicit ObjectOutput(jobject jobj)
		: object<ObjectOutput>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::io::DataOutput>() const;


		void close();
		void flush();
		void write(local_ref< array<jbyte,1> >  const&);
		void write(local_ref< array<jbyte,1> >  const&, jint, jint);
		void write(jint);
		void writeObject(local_ref< java::lang::Object >  const&);
	}; //class ObjectOutput

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_OBJECTOUTPUT_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_OBJECTOUTPUT_HPP_IMPL
#define J2CPP_JAVA_IO_OBJECTOUTPUT_HPP_IMPL

namespace j2cpp {



java::io::ObjectOutput::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::io::ObjectOutput::operator local_ref<java::io::DataOutput>() const
{
	return local_ref<java::io::DataOutput>(get_jobject());
}

void java::io::ObjectOutput::close()
{
	return call_method<
		java::io::ObjectOutput::J2CPP_CLASS_NAME,
		java::io::ObjectOutput::J2CPP_METHOD_NAME(0),
		java::io::ObjectOutput::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject());
}

void java::io::ObjectOutput::flush()
{
	return call_method<
		java::io::ObjectOutput::J2CPP_CLASS_NAME,
		java::io::ObjectOutput::J2CPP_METHOD_NAME(1),
		java::io::ObjectOutput::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject());
}

void java::io::ObjectOutput::write(local_ref< array<jbyte,1> > const &a0)
{
	return call_method<
		java::io::ObjectOutput::J2CPP_CLASS_NAME,
		java::io::ObjectOutput::J2CPP_METHOD_NAME(2),
		java::io::ObjectOutput::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}

void java::io::ObjectOutput::write(local_ref< array<jbyte,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::ObjectOutput::J2CPP_CLASS_NAME,
		java::io::ObjectOutput::J2CPP_METHOD_NAME(3),
		java::io::ObjectOutput::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0, a1, a2);
}

void java::io::ObjectOutput::write(jint a0)
{
	return call_method<
		java::io::ObjectOutput::J2CPP_CLASS_NAME,
		java::io::ObjectOutput::J2CPP_METHOD_NAME(4),
		java::io::ObjectOutput::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}

void java::io::ObjectOutput::writeObject(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::io::ObjectOutput::J2CPP_CLASS_NAME,
		java::io::ObjectOutput::J2CPP_METHOD_NAME(5),
		java::io::ObjectOutput::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::io::ObjectOutput,"java/io/ObjectOutput")
J2CPP_DEFINE_METHOD(java::io::ObjectOutput,0,"close","()V")
J2CPP_DEFINE_METHOD(java::io::ObjectOutput,1,"flush","()V")
J2CPP_DEFINE_METHOD(java::io::ObjectOutput,2,"write","([B)V")
J2CPP_DEFINE_METHOD(java::io::ObjectOutput,3,"write","([BII)V")
J2CPP_DEFINE_METHOD(java::io::ObjectOutput,4,"write","(I)V")
J2CPP_DEFINE_METHOD(java::io::ObjectOutput,5,"writeObject","(Ljava/lang/Object;)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_OBJECTOUTPUT_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
