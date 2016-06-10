/*
 Copyright 2015 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <assert.h>
#include <cstdio>

#include "loader.hpp"

#if HAS_IMGLIB
#include "image.hpp"
#endif

#if HAS_VIDLIB
#include "video.hpp"
#endif

#if HAS_AUDLIB
#include "audio.hpp"
#endif

#include "api.hpp"

Media* Media::create(MediaParams* params, MediaParams* ingestParams, int id) {
    switch (params->_mtype) {
    case IMAGE:
#if HAS_IMGLIB
        return new Image(reinterpret_cast<ImageParams*>(params),
                         reinterpret_cast<ImageIngestParams*>(ingestParams),
                         id);
#else
        {
            string message = "OpenCV " UNSUPPORTED_MEDIA_MESSAGE;
            throw std::runtime_error(message);
        }
#endif
    case VIDEO:
#if HAS_VIDLIB
        return new Video(reinterpret_cast<VideoParams*>(params), id);
#else
        {
            string message = "Video " UNSUPPORTED_MEDIA_MESSAGE;
            throw std::runtime_error(message);
        }
#endif
    case AUDIO:
#if HAS_AUDLIB
        return new Audio(reinterpret_cast<AudioParams*>(params), id);
#else
        {
            string message = "Audio " UNSUPPORTED_MEDIA_MESSAGE;
            throw std::runtime_error(message);
        }
#endif
    default:
        throw std::runtime_error("Unknown media type");
    }
    return 0;
}
