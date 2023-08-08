#ifndef TCH_CXX_WRAPPER_UTILS_H
#define TCH_CXX_WRAPPER_UTILS_H

#include <c10/util/intrusive_ptr.h>

/*
 *  用于创建包装torch常用的引用计数指针的类的宏
 * */
#define CREATE_CONTAINER_CLASS(CLASS_NAME_, WRAP_TYPE_) \
class CLASS_NAME_ { \
protected: \
     explicit CLASS_NAME_(c10::intrusive_ptr<WRAP_TYPE_> value) noexcept: inner(std::move(value)) {} \
 \
     explicit CLASS_NAME_() noexcept: inner() {} \
\
public: \
    c10::intrusive_ptr<WRAP_TYPE_> inner; \
 \
    explicit CLASS_NAME_(std::unique_ptr<WRAP_TYPE_> value) noexcept: inner(std::move(value)) {} \
 \
    CLASS_NAME_(const CLASS_NAME_ &container) { \
        inner = std::move(c10::intrusive_ptr<WRAP_TYPE_>(container.inner)); \
    } \
 \
    CLASS_NAME_ clone() const { \
        return CLASS_NAME_(*this); \
    } \
};


#endif //TCH_CXX_WRAPPER_UTILS_H
