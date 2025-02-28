
inline float4 __attribute__((overloadable)) _convert_float16_sat(float4 source) {
    if (source.s0 > 65500) {
        printf("work %f\n", source.s0);
        source.s0 = 65500.f;
    }
    if (source.s0 < - 65500) {
        printf("work %f\n", source.s0);
        source.s0 = -65500.f;
    }

    if (source.s1 > 65500) {
        printf("work %f\n", source.s1);
        source.s1 = 65500.f;
    }
    if (source.s1 < - 65500) {
        printf("work %f\n", source.s1);
        source.s1 = -65500.f;
    }

    if (source.s2 > 65500) {
        printf("work %f\n", source.s2);
        source.s2 = 65500.f;
    }
    if (source.s2 < - 65500) {
        printf("work %f\n", source.s2);
        source.s2 = -65500.f;
    }

    if (source.s3 > 65500) {
        printf("work %f\n", source.s3);
        source.s3 = 65500.f;
    }

    if (source.s3 < - 65500) {
        printf("work %f\n", source.s3);
        source.s3 = -65500.f;
    }
    return source;
}

inline float _convert_float16_sat(float source) {
    if (source > 65500) {
        printf("work %f\n", source);
        return 65500.f;
    }
    if (source < - 65500) {
        printf("work %f\n", source);
        return -65500.f;
    }
    return source;
}

#ifdef intel_convert_as_bfloat16_float
#define _convert_as_bfloat16_float(val) intel_convert_as_bfloat16_float(val)
#else
inline float _convert_as_bfloat16_float(ushort source) {
    uint u = 0;
    //sign
    if ( (source>>15) ) { 
        u = 1 << 31;
    }
    //exponent
    u += ( ( (source >> 7) & 0b11111111)) << 23;
    //fraction 
    u += (source & 0b1111111) << 16;
    float* f = &u;
    return *f;
}
#endif

#ifdef intel_convert_bfloat16_as_ushort
#define _convert_bfloat16_as_ushort(val) intel_convert_bfloat16_as_ushort(val)
#else
inline ushort _convert_bfloat16_as_ushort(float source) {
    uint* in = &source;
    ushort u = 0;
    if ( (*in>>31) ) { 
        u = 1 << 15;
    }
    //exponent
    u += ( ( (*in >> 23) & 0b11111111)) << 7;
    //fraction
    u += (*in >> 16) & 0b1111111;
    return u;
}
#endif
