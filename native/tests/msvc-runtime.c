/* Link and execute the core's MinGW compatibility surface from an MSVC host.
 * stdin: numerator lo/hi, denominator lo/hi, expected quotient lo/hi (hex). */
#include <emmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

extern __m128i __udivti3(__m128i, __m128i);
extern int stat64i32(const char *, struct _stat64i32 *);
/* MinGW dirent's public record layout; DIR itself remains opaque. */
struct dirent {
    long d_ino;
    unsigned short d_reclen, d_namlen;
    char d_name[260];
};
extern void *opendir(const char *);
extern struct dirent *readdir(void *);
extern int closedir(void *);

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    unsigned long long nlo, nhi, dlo, dhi, qlo, qhi;
    unsigned cases = 0;
    while (scanf("%llx %llx %llx %llx %llx %llx", &nlo, &nhi, &dlo, &dhi, &qlo, &qhi) == 6) {
        __m128i actual = __udivti3(_mm_set_epi64x(nhi, nlo), _mm_set_epi64x(dhi, dlo));
        uint64_t words[2];
        _mm_storeu_si128((__m128i *)words, actual);
        if (words[0] != qlo || words[1] != qhi) {
            fprintf(stderr, "128-bit division mismatch at case %u\n", cases);
            return 2;
        }
        ++cases;
    }
    if (cases < 100) return 3;
    char file[4096];
    snprintf(file, sizeof file, "%s\\oidn_webgpu.dll", argv[1]);
    struct _stat64i32 info;
    if (stat64i32(file, &info) || info.st_size <= 0) return 4;
    void *directory = opendir(argv[1]);
    if (!directory) return 5;
    int found = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (strcmp(entry->d_name, "oidn_webgpu.dll") == 0) {
            if (entry->d_namlen != strlen(entry->d_name)) return 6;
            found = 1;
        }
    }
    if (closedir(directory) || !found) return 7;
    printf("MSVC runtime compatibility: %u divisions, stat and directory access OK\n", cases);
    return 0;
}
