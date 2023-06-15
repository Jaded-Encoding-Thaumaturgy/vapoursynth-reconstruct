ReconstructData *d = static_cast<ReconstructData *>(instanceData);

if (activationReason == arInitial) {
    vsapi->requestFrameFilter(n, d->node, frameCtx);
    vsapi->requestFrameFilter(n, d->slope, frameCtx);
    vsapi->requestFrameFilter(n, d->weights, frameCtx);
    if constexpr (got_intercept)
        vsapi->requestFrameFilter(n, d->intercept, frameCtx);
} else if (activationReason == arAllFramesReady) {
    const VSFrame *node = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSFrame *slope = vsapi->getFrameFilter(n, d->slope, frameCtx);
    const VSFrame *weights = vsapi->getFrameFilter(n, d->weights, frameCtx);
    const VSFrame *intercept = got_intercept ? vsapi->getFrameFilter(n, d->intercept, frameCtx) : nullptr;

    const T *nodePlane = reinterpret_cast<const T *>(vsapi->getReadPtr(node, 0));
    const T *slopePlane = reinterpret_cast<const T *>(vsapi->getReadPtr(slope, 0));
    const T *weightsPlane = reinterpret_cast<const T *>(vsapi->getReadPtr(weights, 0));
    const T *interceptPlane =
        reinterpret_cast<const T *>(got_intercept ? vsapi->getReadPtr(intercept, 0) : nullptr);

    int width = vsapi->getFrameWidth(node, 0);
    int height = vsapi->getFrameHeight(node, 0);
    ptrdiff_t stride = vsapi->getStride(node, 0) / sizeof(T);

    VSFrame *output = vsapi->copyFrame(node, core);
    T *outputPlane = reinterpret_cast<T *>(vsapi->getWritePtr(output, 0));

    int r = d->radius;
    int ws = r * 2 + 1;
    float wa = 1.f / static_cast<float>(ws * ws * 1.0);

    if constexpr (simd) {
        int nw = width & ~0x7;

        SIMDT scale = MANGLE_NAME(set1)(wa);

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < nw; i += 8) {
                SIMDT value = MANGLE_NAME(load)(&(nodePlane[j * stride + i]));
                SIMDT acc = MANGLE_NAME(set1)(0.0f);

                int max_y = std::min(j + r, height - 1);

                for (int y = std::max(j - r, 0); y <= max_y; y++) {
                    int max_x = std::min(i + r + 1, nw - 1);

                    for (int x = std::max(i - r, 0); x <= max_x; x++) {
                        SIMDT slope = MANGLE_NAME(load)(&(slopePlane[y * stride + x]));
                        SIMDT weight = MANGLE_NAME(load)(&(weightsPlane[y * stride + x]));

                        if constexpr (got_intercept) {
                            SIMDT intercept = MANGLE_NAME(load)(&(interceptPlane[y * stride + x]));
                            SIMDT pred = MANGLE_NAME(add)(MANGLE_NAME(mul)(value, slope), intercept);
                            acc = MANGLE_NAME(add)(acc, MANGLE_NAME(mul)(pred, weight));
                        } else {
                            acc = MANGLE_NAME(add)(acc, MANGLE_NAME(mul)(MANGLE_NAME(mul)(value, slope), weight));
                        }
                    }
                }

                SIMDT final = MANGLE_NAME(mul)(acc, scale);
                MANGLE_NAME(store)(&(outputPlane[j * stride + i]), final);
            }
        }
    } else {
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                T value = nodePlane[j * stride + i];
                T acc = 0.0f;

                int max_y = std::min(j + r, height - 1);

                for (int y = std::max(j - r, 0); y <= max_y; y++) {
                    int max_x = std::min(i + r + 1, width - 1);

                    for (int x = std::max(i - r, 0); x <= max_x; x++) {
                        T slope = slopePlane[y * stride + x];
                        T weight = weightsPlane[y * stride + x];

                        if constexpr (got_intercept) {
                            T intercept = interceptPlane[y * stride + x];
                            acc += (value * slope + intercept) * weight;
                        } else {
                            acc += value * slope * weight;
                        }
                    }
                }

                outputPlane[j * stride + i] = acc / wa;
            }
        }
    }

    vsapi->freeFrame(node);
    vsapi->freeFrame(slope);
    vsapi->freeFrame(weights);
    vsapi->freeFrame(intercept);

    return output;
}

return nullptr;