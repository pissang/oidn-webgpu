/* scriptc 0.3.2's Windows runtime uses MinGW CRT spellings. Keep their
 * implementations in the core archive, forwarding to the host's MSVC CRT.
 * The remaining POSIX aliases and C99 printf entry points are supplied by
 * Microsoft's compatibility libraries. No alternate CRT DLL is bundled. */
#include <sys/stat.h>

#define MSVC_DIRECTIVES \
    " /DEFAULTLIB:oldnames.lib /DEFAULTLIB:legacy_stdio_definitions.lib" \
    " /ALTERNATENAME:___chkstk_ms=__chkstk "

/* This translation unit is compiled with Zig's MinGW headers, where
 * #pragma comment(lib, ...) is ignored. MSVC consumes the COFF directives. */
__attribute__((section(".drectve"), used))
static const char directives[sizeof(MSVC_DIRECTIVES) - 1] = MSVC_DIRECTIVES;

int stat64i32(const char *path, struct _stat64i32 *out) {
    return _stat64i32(path, out);
}

/* MSVC does not ship the 128-bit unsigned division libcall used by the
 * precompiled BigInt runtime. Clang's Windows x64 ABI passes each operand
 * by reference and returns the quotient in XMM0, matching MinGW/compiler-rt.
 * Binary long division avoids a recursive __udivti3 call. Before iteration
 * bit, the remainder holds at most 127 numerator bits, so shifting is safe. */
typedef unsigned __int128 uint128;
uint128 __udivti3(uint128 numerator, uint128 denominator) {
    if (denominator == 0) __builtin_trap();
    uint128 quotient = 0;
    uint128 remainder = 0;
    for (int bit = 127; bit >= 0; --bit) {
        remainder = (remainder << 1) | ((numerator >> bit) & 1);
        if (remainder >= denominator) {
            remainder -= denominator;
            quotient |= (uint128)1 << bit;
        }
    }
    return quotient;
}
