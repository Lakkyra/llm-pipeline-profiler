#ifndef MAIN_H
#define MAIN_H

#include <string>
#include "hardware_prober.cuh"

// Public C++ APIs to start/stop the background sampling thread
void start_sampler(const std::string& socket_path, int interval_ms);
void stop_sampler();

// Public C++ APIs to probe empirical/theoretical hardware ceilings
HardwareProbeResult probe_hardware(int device_id = 0);
HardwareProbeResult lookup_hardware_specs(const std::string& gpu_name);

#endif // MAIN_H
