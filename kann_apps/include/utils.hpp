#include <fstream>
#include <iostream>
#include <unordered_map>


struct io_storage {
    std::vector<char> host_buf;
    cl::Buffer mppa_buf;
    io_storage(std::vector<char> &&_host_buf, const cl::Buffer &_mppa_buf)
        : host_buf(std::move(_host_buf)), mppa_buf(_mppa_buf)
    {
    }
};

struct input_storage : io_storage {
    std::ifstream stream;
    input_storage(std::vector<char> &&_host_buf, const cl::Buffer &_mppa_buf,
                  std::ifstream &&_stream)
        : io_storage(std::move(_host_buf), _mppa_buf),
          stream(std::move(_stream))
    {
    }
};

struct output_storage : io_storage {
    std::ofstream stream;
    output_storage(std::vector<char> &&_host_buf, const cl::Buffer &_mppa_buf,
                   std::ofstream &&_stream)
        : io_storage(std::move(_host_buf), _mppa_buf),
          stream(std::move(_stream))
    {
    }
};

struct data_storage {
    std::unordered_map<const KaNN::IOBufferDescr*, cl::Buffer> inputs_outputs;
    std::vector<input_storage> local_inputs;
    std::vector<output_storage> local_outputs;
};

// Read the input files, update the buffers and return a flag that is false
// if there are no data available.
bool update_inputs(
    data_storage &in,
    cl::CommandQueue *command_queue,
    bool &success,
    std::chrono::time_point<std::chrono::steady_clock> &start)
{
    bool inputs_ready = true;
    for (input_storage &in_storage : in.local_inputs) {
        in_storage.stream.read(in_storage.host_buf.data(), in_storage.host_buf.size());
        // Check that we did not read past the eof.
        if (in_storage.stream.eof()) {
            inputs_ready = false;
            break;
        }
        // Check that we are not in error
        if (in_storage.stream.fail()) {
            inputs_ready = false;
            success = false;
            std::cerr << "[app] failed to read input." << std::endl;
            break;
        }
        cl_int err = command_queue->enqueueWriteBuffer(
            in_storage.mppa_buf,
            CL_FALSE,
            0,
            in_storage.host_buf.size(),
            in_storage.host_buf.data());
        if (err != CL_SUCCESS) {
            inputs_ready = false;
            success = false;
            std::cerr << "[app] failed to write to opencl buffer." << std::endl;
            break;
        }
    }
    command_queue->finish();
    start = std::chrono::steady_clock::now();
    return inputs_ready;
}

// Write the output to the output files and return a success flag.
bool update_outputs(
    data_storage &out,
    cl::CommandQueue *command_queue,
    std::chrono::time_point<std::chrono::steady_clock> &end
)
{
    bool success = true;
    for (output_storage &storage : out.local_outputs) {
        cl_int err = command_queue->enqueueReadBuffer(
            storage.mppa_buf,
            CL_FALSE,
            0,
            storage.host_buf.size(),
            storage.host_buf.data());
        if (err != CL_SUCCESS) {
            success = false;
            std::cerr << "[app] failed to write to opencl buffer." << std::endl;
            break;
        }
    }
    command_queue->finish();
    end = std::chrono::steady_clock::now();

    for (output_storage &storage : out.local_outputs) {
        storage.stream.write(storage.host_buf.data(), storage.host_buf.size());

        // Check that we are not in error
        if (storage.stream.fail()) {
            success = false;
            std::cerr << "[app] failed to write output." << std::endl;
            break;
        }
        storage.stream.flush();
    }
    return success;
}