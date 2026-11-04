# Project-wide warning, sanitizer and coverage settings applied per target so
# third-party code (OpenCV, included as SYSTEM) is never affected.
function(vtt_set_project_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
      -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual
      -Wnull-dereference -Wdouble-promotion -Wformat=2)
    if(VTT_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(VTT_ENABLE_SANITIZERS)
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    if(VTT_ENABLE_COVERAGE)
      target_compile_options(${target} PRIVATE --coverage -O0 -g)
      target_link_options(${target} PRIVATE --coverage)
    endif()
    if(VTT_NATIVE_ARCH)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(VTT_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()
endfunction()
