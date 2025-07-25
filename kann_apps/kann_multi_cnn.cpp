/**
 *****************************************************************************
 *
 * Copyright 2020 Kalray
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************
 */
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl_mppa.h>
#include <CL/cl2.hpp>

#include <chrono>
#include <thread>
#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <kann_service/kann_service.hpp>

#include "utils.hpp"
#include "cl_utils.hpp"

#define PARAMS_PATH_BASE_IDX 1
#define IO_DIR_BASE_IDX      2
#define FIRST_NETWORK_IDX    PARAMS_PATH_BASE_IDX
#define NET_ARGS_SIZE        2


int threaded_kann_generic(
    const std::string &io_dir,
    const cl::Context &context,
    const cl::Device &sub_device,
    std::unique_ptr<KaNN::Service> kann_service,
    const std::string &thread_name
)
{
    cl_command_queue_properties command_queue_flags = CL_QUEUE_PROFILING_ENABLE;
    cl::CommandQueue command_queue;

    // 1. Create command queue
    std::cout << "[" << thread_name << "] Creating the command queue..." << std::endl;
    command_queue = cl::CommandQueue(context, sub_device, command_queue_flags);

    data_storage local_storage;
    try {
        std::cout << "[" << thread_name << "] Initialising device..." << std::endl;
        kann_service->initDevice(context, sub_device, command_queue);
    } catch (std::exception &e) {
        std::cerr << "[" << thread_name << "] Failed to initialise KaNN::Service"
                  << std::endl;
        std::cerr << "Catching exception :\n    " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Manage Input of the NN
    // getInputDescr() returns the NN input descriptors list.
    // In order to be opened in the same order than
    // external input provider, we need to sort the opening.
    // This is important to avoid deadlock on fifo (mkfifo) fd.
    auto input_list = kann_service->getInputDescr();
    std::sort(input_list.begin(), input_list.end(),
              [](const KaNN::IOBufferDescr *&a, const KaNN::IOBufferDescr *&b) {
                  return a->getName() < b->getName();
              });
    for (auto io : input_list) {
        std::vector<char> buf(io->getSize());
        std::string path = io_dir + "/" + io->getName();
        auto stream = std::ifstream(path, std::ifstream::binary);
        if (!stream.good()) {
            std::cerr << "[app] Failed to open input " << path << std::endl;
            return EXIT_FAILURE;
        }
        cl::Buffer mppa_buf = cl::Buffer(context, CL_MEM_READ_ONLY, io->getSize());
        local_storage.inputs_outputs.insert({io, mppa_buf});
        try {
            local_storage.local_inputs.emplace_back(
                input_storage(std::move(buf), mppa_buf, std::move(stream)));
            } catch (std::exception &e) {
            std::cerr << "[" << thread_name << "] Failed to create IO " << io->getName() << ": "
                << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Manage Output of the NN
    // getOutputDescr() returns the NN output descriptors list.
    // In order to be opened in the same order than
    // external output reader, we need to sort the opening.
    // This is important to avoid deadlock on fifo (mkfifo) fd.
    auto output_list = kann_service->getOutputDescr();
    std::sort(output_list.begin(), output_list.end(),
              [](const KaNN::IOBufferDescr *&a, const KaNN::IOBufferDescr *&b) {
                  return a->getName() < b->getName();
              });
    for (auto io : output_list) {
        std::vector<char> buf(io->getSize());
        std::string path = io_dir + "/" + io->getName();
        if (system(("mkdir -p $(dirname " + io_dir + "/" + io->getName() + ")").c_str()) != 0) {
            std::cerr << "[" << thread_name << "] Failed to create directory" << std::endl;
            return EXIT_FAILURE;
        }
        auto stream = std::ofstream(path, std::ofstream::binary);
        if (!stream.good()) {
            std::cerr <<"[" << thread_name << "] Failed to open output " << path
                << std::endl;
            return EXIT_FAILURE;
        }
        cl::Buffer mppa_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, io->getSize());
        local_storage.inputs_outputs.insert({io, mppa_buf});
        try {
            local_storage.local_outputs.emplace_back(
                output_storage(std::move(buf), mppa_buf, std::move(stream)));
            } catch (std::exception &e) {
            std::cerr << "[" << thread_name << "] Failed to create IO " << io->getName() << ": "
                << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Decode all frames and write the results
    bool success = true;
    int frame_idx = 1;
    std::chrono::time_point<std::chrono::steady_clock> start;
    while(update_inputs(local_storage, &command_queue, success, start)) {
        cl::Event evt;
        try {
            kann_service->processFrame(local_storage.inputs_outputs, nullptr, &evt);
            evt.wait();
        } catch (std::exception &e) {
            std::cerr << "[" << thread_name << "] Failed to process_frame or wait evt" << std::endl;
            std::cerr << "Catching exception : " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        std::chrono::time_point<std::chrono::steady_clock> end;
        if (!update_outputs(local_storage, &command_queue, end)) {
            std::cerr << "[" << thread_name << "] Error encountereed while writing outputs" << std::endl;
            return EXIT_FAILURE;
        }
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "[" << thread_name << "] Performance of frame " << frame_idx << ": "
                  << (1000 * elapsed_seconds.count()) << " ms - "
                  << (1.0 / elapsed_seconds.count()) << " fps" << std::endl;
        frame_idx++;
        start = std::chrono::steady_clock::now();
    }
    if (!success) {
        std::cerr << "[" << thread_name << "] Failed to read new input or to send new input to device" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        kann_service->terminateDevice();
    } catch (std::exception &e) {
        std::cerr << "[" << thread_name << "] Failed to finish and delete instance of KaNN::Service."
        << std::endl << "Catching exception : " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[app] Exiting" << std::endl;
    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
    if (argc == 1 || (argc % NET_ARGS_SIZE) != 1) {
        std::cout << "Usage:\n" << argv[0] << " <path to .kann file>" << " <inputs_outputs_dir>" << std::endl;
        return EXIT_FAILURE;
    }
    const unsigned nb_networks = (argc - FIRST_NETWORK_IDX) / NET_ARGS_SIZE;
    std::vector<std::string> params_path;
    std::vector<std::string> io_dir;

    for (unsigned net_idx = 0; net_idx < nb_networks; net_idx++) {
        params_path.push_back(std::string(argv[PARAMS_PATH_BASE_IDX + net_idx * NET_ARGS_SIZE]));
        io_dir.push_back(std::string(argv[IO_DIR_BASE_IDX + net_idx * NET_ARGS_SIZE]) + "/");
    }

    // Get default device
    cl::Device device;
    getDefaultDevice(device);

    // Create a context on this device
    std::cout << "[app] Creating a context on the device" << std::endl;
    cl::Context context = cl::Context(device);

    int last_used_cluster = 0;
    cl_int nbr_max_compute_units;
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &nbr_max_compute_units);

    std::vector<std::thread> threads(nb_networks);
    for (size_t thread_idx = 0; thread_idx < threads.size(); thread_idx++) {
        // Initialize the KaNN instance with the parameters
        KaNN::Service *kann_service;
        std::cout << "[app] creating instance of KaNN::Service." << std::endl;
        kann_service = new KaNN::Service(params_path[thread_idx]);

        int nbr_clusters = (int)kann_service->getNbrClusters();
        if (last_used_cluster + nbr_clusters > nbr_max_compute_units) {
            last_used_cluster = 0;
        }
        int first_cluster = last_used_cluster;
        last_used_cluster += nbr_clusters;

        cl::Device sub_device;
        getSubDevice(device, first_cluster, nbr_clusters, sub_device);
        threads[thread_idx] = std::thread(
            [&io_dir, thread_idx, &context, sub_device, kann_service]() {
                std::string thread_name = kann_service->getName() + "_" + std::to_string(thread_idx);
                threaded_kann_generic(
                    io_dir[thread_idx], context, sub_device,
                    std::unique_ptr<KaNN::Service>(kann_service),
                    thread_name);
                }  // End of thread lambda
                // End of threads creation
        );
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
    std::cout << "[app] exiting" << std::endl;
    return EXIT_SUCCESS;
}
