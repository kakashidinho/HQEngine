/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.view.animation.GridLayoutAnimationController
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_VIEW_ANIMATION_GRIDLAYOUTANIMATIONCONTROLLER_HPP_DECL
#define J2CPP_ANDROID_VIEW_ANIMATION_GRIDLAYOUTANIMATIONCONTROLLER_HPP_DECL


namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace view { namespace animation { class LayoutAnimationController; } } } }
namespace j2cpp { namespace android { namespace view { namespace animation { namespace LayoutAnimationController_ { class AnimationParameters; } } } } }
namespace j2cpp { namespace android { namespace view { namespace animation { class Animation; } } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/view/animation/Animation.hpp>
#include <android/view/animation/LayoutAnimationController.hpp>


namespace j2cpp {

namespace android { namespace view { namespace animation {

	class GridLayoutAnimationController;
	namespace GridLayoutAnimationController_ {

		class AnimationParameters;
		class AnimationParameters
			: public object<AnimationParameters>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)
			J2CPP_DECLARE_FIELD(0)
			J2CPP_DECLARE_FIELD(1)
			J2CPP_DECLARE_FIELD(2)
			J2CPP_DECLARE_FIELD(3)

			explicit AnimationParameters(jobject jobj)
			: object<AnimationParameters>(jobj)
			, column(jobj)
			, row(jobj)
			, columnsCount(jobj)
			, rowsCount(jobj)
			{
			}

			operator local_ref<android::view::animation::LayoutAnimationController_::AnimationParameters>() const;


			AnimationParameters();

			field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), jint > column;
			field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(1), J2CPP_FIELD_SIGNATURE(1), jint > row;
			field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(2), J2CPP_FIELD_SIGNATURE(2), jint > columnsCount;
			field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(3), J2CPP_FIELD_SIGNATURE(3), jint > rowsCount;
		}; //class AnimationParameters

	} //namespace GridLayoutAnimationController_

	class GridLayoutAnimationController
		: public object<GridLayoutAnimationController>
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
		J2CPP_DECLARE_FIELD(0)
		J2CPP_DECLARE_FIELD(1)
		J2CPP_DECLARE_FIELD(2)
		J2CPP_DECLARE_FIELD(3)
		J2CPP_DECLARE_FIELD(4)
		J2CPP_DECLARE_FIELD(5)
		J2CPP_DECLARE_FIELD(6)
		J2CPP_DECLARE_FIELD(7)
		J2CPP_DECLARE_FIELD(8)

		typedef GridLayoutAnimationController_::AnimationParameters AnimationParameters;

		explicit GridLayoutAnimationController(jobject jobj)
		: object<GridLayoutAnimationController>(jobj)
		{
		}

		operator local_ref<android::view::animation::LayoutAnimationController>() const;


		GridLayoutAnimationController(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		GridLayoutAnimationController(local_ref< android::view::animation::Animation > const&);
		GridLayoutAnimationController(local_ref< android::view::animation::Animation > const&, jfloat, jfloat);
		jfloat getColumnDelay();
		void setColumnDelay(jfloat);
		jfloat getRowDelay();
		void setRowDelay(jfloat);
		jint getDirection();
		void setDirection(jint);
		jint getDirectionPriority();
		void setDirectionPriority(jint);
		jboolean willOverlap();

		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), jint > DIRECTION_LEFT_TO_RIGHT;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(1), J2CPP_FIELD_SIGNATURE(1), jint > DIRECTION_RIGHT_TO_LEFT;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(2), J2CPP_FIELD_SIGNATURE(2), jint > DIRECTION_TOP_TO_BOTTOM;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(3), J2CPP_FIELD_SIGNATURE(3), jint > DIRECTION_BOTTOM_TO_TOP;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(4), J2CPP_FIELD_SIGNATURE(4), jint > DIRECTION_HORIZONTAL_MASK;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(5), J2CPP_FIELD_SIGNATURE(5), jint > DIRECTION_VERTICAL_MASK;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(6), J2CPP_FIELD_SIGNATURE(6), jint > PRIORITY_NONE;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(7), J2CPP_FIELD_SIGNATURE(7), jint > PRIORITY_COLUMN;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(8), J2CPP_FIELD_SIGNATURE(8), jint > PRIORITY_ROW;
	}; //class GridLayoutAnimationController

} //namespace animation
} //namespace view
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_VIEW_ANIMATION_GRIDLAYOUTANIMATIONCONTROLLER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_VIEW_ANIMATION_GRIDLAYOUTANIMATIONCONTROLLER_HPP_IMPL
#define J2CPP_ANDROID_VIEW_ANIMATION_GRIDLAYOUTANIMATIONCONTROLLER_HPP_IMPL

namespace j2cpp {




android::view::animation::GridLayoutAnimationController_::AnimationParameters::operator local_ref<android::view::animation::LayoutAnimationController_::AnimationParameters>() const
{
	return local_ref<android::view::animation::LayoutAnimationController_::AnimationParameters>(get_jobject());
}


android::view::animation::GridLayoutAnimationController_::AnimationParameters::AnimationParameters()
: object<android::view::animation::GridLayoutAnimationController_::AnimationParameters>(
	call_new_object<
		android::view::animation::GridLayoutAnimationController_::AnimationParameters::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController_::AnimationParameters::J2CPP_METHOD_NAME(0),
		android::view::animation::GridLayoutAnimationController_::AnimationParameters::J2CPP_METHOD_SIGNATURE(0)
	>()
)
, column(get_jobject())
, row(get_jobject())
, columnsCount(get_jobject())
, rowsCount(get_jobject())
{
}




J2CPP_DEFINE_CLASS(android::view::animation::GridLayoutAnimationController_::AnimationParameters,"android/view/animation/GridLayoutAnimationController$AnimationParameters")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController_::AnimationParameters,0,"<init>","()V")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController_::AnimationParameters,0,"column","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController_::AnimationParameters,1,"row","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController_::AnimationParameters,2,"columnsCount","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController_::AnimationParameters,3,"rowsCount","I")



android::view::animation::GridLayoutAnimationController::operator local_ref<android::view::animation::LayoutAnimationController>() const
{
	return local_ref<android::view::animation::LayoutAnimationController>(get_jobject());
}


android::view::animation::GridLayoutAnimationController::GridLayoutAnimationController(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::view::animation::GridLayoutAnimationController>(
	call_new_object<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(0),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}



android::view::animation::GridLayoutAnimationController::GridLayoutAnimationController(local_ref< android::view::animation::Animation > const &a0)
: object<android::view::animation::GridLayoutAnimationController>(
	call_new_object<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(1),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



android::view::animation::GridLayoutAnimationController::GridLayoutAnimationController(local_ref< android::view::animation::Animation > const &a0, jfloat a1, jfloat a2)
: object<android::view::animation::GridLayoutAnimationController>(
	call_new_object<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(2),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1, a2)
)
{
}


jfloat android::view::animation::GridLayoutAnimationController::getColumnDelay()
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(3),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(3), 
		jfloat
	>(get_jobject());
}

void android::view::animation::GridLayoutAnimationController::setColumnDelay(jfloat a0)
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(4),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}

jfloat android::view::animation::GridLayoutAnimationController::getRowDelay()
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(5),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(5), 
		jfloat
	>(get_jobject());
}

void android::view::animation::GridLayoutAnimationController::setRowDelay(jfloat a0)
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(6),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject(), a0);
}

jint android::view::animation::GridLayoutAnimationController::getDirection()
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(7),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(7), 
		jint
	>(get_jobject());
}

void android::view::animation::GridLayoutAnimationController::setDirection(jint a0)
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(8),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

jint android::view::animation::GridLayoutAnimationController::getDirectionPriority()
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(9),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(9), 
		jint
	>(get_jobject());
}

void android::view::animation::GridLayoutAnimationController::setDirectionPriority(jint a0)
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(10),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject(), a0);
}

jboolean android::view::animation::GridLayoutAnimationController::willOverlap()
{
	return call_method<
		android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_NAME(11),
		android::view::animation::GridLayoutAnimationController::J2CPP_METHOD_SIGNATURE(11), 
		jboolean
	>(get_jobject());
}



static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(0),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(0),
	jint
> android::view::animation::GridLayoutAnimationController::DIRECTION_LEFT_TO_RIGHT;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(1),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(1),
	jint
> android::view::animation::GridLayoutAnimationController::DIRECTION_RIGHT_TO_LEFT;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(2),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(2),
	jint
> android::view::animation::GridLayoutAnimationController::DIRECTION_TOP_TO_BOTTOM;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(3),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(3),
	jint
> android::view::animation::GridLayoutAnimationController::DIRECTION_BOTTOM_TO_TOP;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(4),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(4),
	jint
> android::view::animation::GridLayoutAnimationController::DIRECTION_HORIZONTAL_MASK;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(5),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(5),
	jint
> android::view::animation::GridLayoutAnimationController::DIRECTION_VERTICAL_MASK;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(6),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(6),
	jint
> android::view::animation::GridLayoutAnimationController::PRIORITY_NONE;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(7),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(7),
	jint
> android::view::animation::GridLayoutAnimationController::PRIORITY_COLUMN;

static_field<
	android::view::animation::GridLayoutAnimationController::J2CPP_CLASS_NAME,
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_NAME(8),
	android::view::animation::GridLayoutAnimationController::J2CPP_FIELD_SIGNATURE(8),
	jint
> android::view::animation::GridLayoutAnimationController::PRIORITY_ROW;


J2CPP_DEFINE_CLASS(android::view::animation::GridLayoutAnimationController,"android/view/animation/GridLayoutAnimationController")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,0,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,1,"<init>","(Landroid/view/animation/Animation;)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,2,"<init>","(Landroid/view/animation/Animation;FF)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,3,"getColumnDelay","()F")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,4,"setColumnDelay","(F)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,5,"getRowDelay","()F")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,6,"setRowDelay","(F)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,7,"getDirection","()I")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,8,"setDirection","(I)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,9,"getDirectionPriority","()I")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,10,"setDirectionPriority","(I)V")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,11,"willOverlap","()Z")
J2CPP_DEFINE_METHOD(android::view::animation::GridLayoutAnimationController,12,"getDelayForView","(Landroid/view/View;)J")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,0,"DIRECTION_LEFT_TO_RIGHT","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,1,"DIRECTION_RIGHT_TO_LEFT","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,2,"DIRECTION_TOP_TO_BOTTOM","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,3,"DIRECTION_BOTTOM_TO_TOP","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,4,"DIRECTION_HORIZONTAL_MASK","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,5,"DIRECTION_VERTICAL_MASK","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,6,"PRIORITY_NONE","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,7,"PRIORITY_COLUMN","I")
J2CPP_DEFINE_FIELD(android::view::animation::GridLayoutAnimationController,8,"PRIORITY_ROW","I")

} //namespace j2cpp

#endif //J2CPP_ANDROID_VIEW_ANIMATION_GRIDLAYOUTANIMATIONCONTROLLER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
