/*
 * Copyright 2015 Nervana Systems Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <cuda.h>

#define CUDA_CHECK( fn ) do { \
  CUresult status = (fn); \
  if ( CUDA_SUCCESS != status ) { \
    const char* errstr; \
    cuGetErrorString(status, &errstr); \
    printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

int main(int argc, char* argv[]) {

  char deviceName[32];
  int devCount, ordinal, major, minor;
  int maxMajor = 0, maxMinor = 0;
  CUdevice hDevice;

  // Initialize the Driver API and find a device
  CUDA_CHECK( cuInit(0) );
  CUDA_CHECK( cuDeviceGetCount(&devCount) );
  for (ordinal = 0; ordinal < devCount; ordinal++) {
    CUDA_CHECK( cuDeviceGet(&hDevice, ordinal) );
    CUDA_CHECK( cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice) );
    CUDA_CHECK( cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice) );
    CUDA_CHECK( cuDeviceGetName(deviceName, sizeof(deviceName), hDevice) );
    if (major >= maxMajor) {
      maxMajor = major;
      if (minor > maxMinor) {
        maxMinor = minor;
      }
    }
  }
  if (maxMajor == 0) {
    // no CUDA capable devices found.
    printf("0\n");
  } else {
    printf("%d.%d\n", maxMajor, maxMinor);
  }
  exit(0);
}
