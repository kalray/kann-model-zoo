#define CL_HPP_ENABLE_EXCEPTIONS
#include <iostream>
#include <CL/cl_mppa.h>
#include <CL/cl2.hpp>
#include <assert.h>

int getDefaultDevice(cl::Device &device)
{
    cl_int err;
    cl::Platform platform;
    std::vector<cl::Device> devices;
    // 1. Access OpenCL platform
    std::cout << "[app] Accessing default platform..." << std::endl;
    err = cl::Platform::get(&platform);
    if (err != CL_SUCCESS) {
        std::cerr << "[app] Failed to find a cl::Platform" << std::endl;
        return EXIT_FAILURE;
    }
    // 2. Access the devices available on the platform
    std::cout
        << "[app] Accessing all acceleration devices available on the platform..."
        << std::endl;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    if (err != CL_SUCCESS || devices.size() == 0) {
        std::cerr << "[app] failed to get devices" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "[app] Found " << std::to_string(devices.size()) << " devices"
              << std::endl;
    // 3. Extract the first device in list
    std::cout << "[app] Accessing the first device in the list" << std::endl;
    device = devices[0];
    return EXIT_SUCCESS;
}

void getDefaultSubDevice(cl::Device &device, int nbr_clusters,
                         cl::Device &sub_device)
{
    std::vector<cl::Device> sub_devices;
    // Check compatibility between requested compute units and device's max
    // compute units
    cl_int nbr_max_compute_units;
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &nbr_max_compute_units);
    assert(nbr_clusters <= (int)nbr_max_compute_units);

    // Create subdevice partition with nbr_clusters
    cl_device_partition_property part_properties[3] = {
        CL_DEVICE_PARTITION_BY_COUNTS, nbr_clusters, 0};
    // Create subdevices
    device.createSubDevices(part_properties, &sub_devices);

    // Extract the first sub-device
    sub_device = sub_devices[0];
}

void getSubDevice(cl::Device &device, int first_cluster, int nbr_clusters,
                  cl::Device &sub_device)
{
    if (first_cluster == 0) {
        // Fallback to default mode
        getDefaultSubDevice(device, nbr_clusters, sub_device);
        return;
    }
    // Check consistency of args
    cl_int nbr_max_compute_units;
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &nbr_max_compute_units);
    assert(nbr_clusters + first_cluster <= (int)nbr_max_compute_units);
    std::vector<cl::Device> sub_devices;
    // We partition our device into 2 sub-devices.
    // The first sub-device has compute units from id 0 to first_cluster-1
    // The second sub-device has compute units from id first_cluster to
    // max_compute_units-1
    cl_device_partition_property part_properties[5] = {
        CL_DEVICE_PARTITION_BY_COUNTS, first_cluster, nbr_clusters,
        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
    // Create subdevices
    device.createSubDevices(part_properties, &sub_devices);
    // We should have created 2 subdevices
    assert(sub_devices.size() == 2);
    // The sub_device of interest is the second one in the vector.
    sub_device = sub_devices[1];
}
