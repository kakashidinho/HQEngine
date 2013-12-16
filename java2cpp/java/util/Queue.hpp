/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.Queue
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_QUEUE_HPP_DECL
#define J2CPP_JAVA_UTIL_QUEUE_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace util { class Collection; } } }


#include <java/lang/Object.hpp>
#include <java/util/Collection.hpp>


namespace j2cpp {

namespace java { namespace util {

	class Queue;
	class Queue
		: public object<Queue>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)

		explicit Queue(jobject jobj)
		: object<Queue>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::util::Collection>() const;


		jboolean offer(local_ref< java::lang::Object >  const&);
		local_ref< java::lang::Object > poll();
		local_ref< java::lang::Object > remove();
		local_ref< java::lang::Object > peek();
		local_ref< java::lang::Object > element();
	}; //class Queue

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_QUEUE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_QUEUE_HPP_IMPL
#define J2CPP_JAVA_UTIL_QUEUE_HPP_IMPL

namespace j2cpp {



java::util::Queue::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::util::Queue::operator local_ref<java::util::Collection>() const
{
	return local_ref<java::util::Collection>(get_jobject());
}

jboolean java::util::Queue::offer(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Queue::J2CPP_CLASS_NAME,
		java::util::Queue::J2CPP_METHOD_NAME(0),
		java::util::Queue::J2CPP_METHOD_SIGNATURE(0), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::Queue::poll()
{
	return call_method<
		java::util::Queue::J2CPP_CLASS_NAME,
		java::util::Queue::J2CPP_METHOD_NAME(1),
		java::util::Queue::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Queue::remove()
{
	return call_method<
		java::util::Queue::J2CPP_CLASS_NAME,
		java::util::Queue::J2CPP_METHOD_NAME(2),
		java::util::Queue::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Queue::peek()
{
	return call_method<
		java::util::Queue::J2CPP_CLASS_NAME,
		java::util::Queue::J2CPP_METHOD_NAME(3),
		java::util::Queue::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Queue::element()
{
	return call_method<
		java::util::Queue::J2CPP_CLASS_NAME,
		java::util::Queue::J2CPP_METHOD_NAME(4),
		java::util::Queue::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::Object >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::util::Queue,"java/util/Queue")
J2CPP_DEFINE_METHOD(java::util::Queue,0,"offer","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Queue,1,"poll","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Queue,2,"remove","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Queue,3,"peek","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Queue,4,"element","()Ljava/lang/Object;")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_QUEUE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
