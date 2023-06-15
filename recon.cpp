// Originally written by shane, updated by Setsugen no ao

#include "VSHelper4.h"
#include "VapourSynth4.h"
#include "kernel/cpufeatures.h"
#include "kernel/cpulevel.h"
#include "kernel/umHalf.h"

#include <algorithm>
#include <immintrin.h>
#include <memory>
#include <vector>

static const std::string reconstructName { "recon.Reconstruct: " };

static auto getSizeStr = [](VSVideoInfo vi) {
    return ("[" + std::to_string(vi.width) + "x" + std::to_string(vi.height) + "]");
};

struct ReconstructData {
        VSVideoInfo vi;
        VSNode *node, *slope, *weights, *intercept;
        uint8_t radius;
};

static void VS_CC reconstructFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ReconstructData *d = static_cast<ReconstructData *>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->slope);
    vsapi->freeNode(d->weights);
    vsapi->freeNode(d->intercept);
    delete d;
}

#define T float
#define SIMDT __m256
#define MANGLE_NAME(name) _mm256##_##name##_ps
template<bool simd, bool got_intercept>
static const VSFrame *VS_CC reconstructGetFrame_ps(
    int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core,
    const VSAPI *vsapi
) {
#include "recon_func.h"
}
#undef T
#undef SIMDT
#undef MANGLE_NAME

#define T float16
#define SIMDT __m256h
#define MANGLE_NAME(name) _mm256##_##name##_ph
template<bool simd, bool got_intercept>
static const VSFrame *VS_CC reconstructGetFrame_ph(
    int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core,
    const VSAPI *vsapi
) {
#ifdef VS_TARGET_CPU_FP16
#include "recon_func.h"
#else
    return nullptr;
#endif
}
#undef T
#undef SIMDT
#undef MANGLE_NAME

static void VS_CC reconstructCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<ReconstructData> d = std::make_unique<ReconstructData>();
    VSPlugin *self = static_cast<VSPlugin *>(userData);
    VSFilterGetFrame getFrame;
    int err;

#ifdef VS_TARGET_CPU_X86
    const CPUFeatures &f = *getCPUFeatures();
#define CPU_F16C (f.f16c)
#else
#define CPU_F16C (false)
#endif

    d->node = vsapi->mapGetNode(in, "node", 0, nullptr);
    d->slope = vsapi->mapGetNode(in, "slope", 0, nullptr);
    d->weights = vsapi->mapGetNode(in, "weights", 0, nullptr);
    d->intercept = vsapi->mapGetNode(in, "intercept", 0, &err);

    if (err)
        d->intercept = nullptr;

    d->vi = *vsapi->getVideoInfo(d->node);

    std::vector<std::pair<VSNode *, std::string>> nodes {
        {     d->node,      "node"},
        {    d->slope,     "slope"},
        {  d->weights,   "weights"},
        {d->intercept, "intercept"}
    };

    for (const auto &[node, name] : nodes) {
        if (!node)
            continue;

        const VSVideoInfo nodeVi = *vsapi->getVideoInfo(node);

        if (d->vi.width != nodeVi.width || d->vi.height != nodeVi.height) {
            std::string sizes = getSizeStr(nodeVi) + " and " + getSizeStr(d->vi);
            vsapi->mapSetError(
                out, (reconstructName + name + " must have the same dimensions, passed " + sizes + ".").c_str()
            );
            return;
        }

        if (!(nodeVi.format.sampleType == stFloat &&
              (CPU_F16C && nodeVi.format.bitsPerSample == 16 || nodeVi.format.bitsPerSample == 32))) {
            std::string bits { CPU_F16C ? "16-32 " : "32" };
            vsapi->mapSetError(out, (reconstructName + name + " must be " + bits + "bits float!").c_str());
            return;
        }

        if (nodeVi.format.bitsPerSample != d->vi.format.bitsPerSample) {
            std::string bits =
                std::to_string(nodeVi.format.bitsPerSample) + " and " + std::to_string(d->vi.format.bitsPerSample);
            vsapi->mapSetError(
                out, (reconstructName + name + " must have the same bitdepths, passed " + bits + ".").c_str()
            );
            return;
        }

        if (nodeVi.format.numPlanes != 1) {
            vsapi->mapSetError(out, (reconstructName + name + " must have only one plane!").c_str());
            return;
        }
    }

    d->radius = static_cast<uint8_t>(vsapi->mapGetInt(in, "radius", 0, &err));

    if (err)
        d->radius = 4;

    int opt = vsapi->mapGetIntSaturated(in, "opt", 0, &err);
    if (err)
        opt = 0;

    bool simd = vs_get_cpulevel(core, vsapi) > VS_CPU_LEVEL_NONE && (opt == 0 || opt == 2);

    bool float_16 = d->vi.format.bitsPerSample == 16;

    if (d->intercept) {
        if (simd)
            getFrame = float_16 ? reconstructGetFrame_ph<true, true> : reconstructGetFrame_ps<true, true>;
        else
            getFrame = float_16 ? reconstructGetFrame_ph<false, true> : reconstructGetFrame_ps<false, true>;
    } else {
        if (simd)
            getFrame = float_16 ? reconstructGetFrame_ph<true, false> : reconstructGetFrame_ps<true, false>;
        else
            getFrame = float_16 ? reconstructGetFrame_ph<false, false> : reconstructGetFrame_ps<false, false>;
    }

    std::vector<VSFilterDependency> deps = {
        {   d->node, rpGeneral},
        {  d->slope, rpGeneral},
        {d->weights, rpGeneral},
    };

    if (d->intercept)
        deps.push_back({ d->intercept, rpGeneral });

    vsapi->createVideoFilter(
        out, "Reconstruct", &d->vi, getFrame, reconstructFree, fmParallel, deps.data(), static_cast<int>(deps.size()),
        d.get(), core
    );
    d.release();
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        "dev.setsugen.reconstruct", "recon", "Chroma reconstruction plugin.", VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION, 0, plugin
    );

    vspapi->registerFunction(
        "Reconstruct", "node:vnode;slope:vnode;weights:vnode;intercept:vnode:opt;radius:int:opt;opt:int:opt;",
        "clip:vnode;", reconstructCreate, (void *) plugin, plugin
    );
}
