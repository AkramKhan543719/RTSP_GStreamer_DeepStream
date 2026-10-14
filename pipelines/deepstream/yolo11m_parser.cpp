#include <algorithm>
#include <cmath>
#include <vector>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo11m(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects)
{
    if (layers.empty() || !layers[0].buffer)
        return false;

    const float* data = static_cast<const float*>(layers[0].buffer);
    const int classes = 80;
    const int boxes = 8400;
    const float threshold = params.perClassPreclusterThreshold.empty()
        ? 0.25f : params.perClassPreclusterThreshold[0];

    for (int i = 0; i < boxes; ++i) {
        float score = 0.0f;
        int cls = 0;

        for (int c = 0; c < classes; ++c) {
            float s = data[(c + 4) * boxes + i];
            if (s > score) {
                score = s;
                cls = c;
            }
        }

        if (score < threshold)
            continue;

        float cx = data[i];
        float cy = data[boxes + i];
        float w  = data[2 * boxes + i];
        float h  = data[3 * boxes + i];

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = cls;
        obj.detectionConfidence = score;
        obj.left = std::max(0.0f, cx - w / 2.0f);
        obj.top = std::max(0.0f, cy - h / 2.0f);
        obj.width = std::min(static_cast<float>(network.width), cx + w / 2.0f) - obj.left;
        obj.height = std::min(static_cast<float>(network.height), cy + h / 2.0f) - obj.top;

        if (obj.width > 0 && obj.height > 0)
            objects.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11m);
