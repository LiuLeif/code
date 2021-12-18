#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_dummy_1;
class kernel_dummy_2;
class kernel_dummy_3;

int main(int argc, char *argv[]) {
    sycl::queue queue_gpu(
        sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    sycl::queue queue_cpu(
        sycl::host_selector{}, {sycl::property::queue::enable_profiling()});

    sycl::event event;
    {
        // NOTE: buffer 出作用域时 sycl runtime 会加上一个隐含的 barrier
        int a = 0;
        sycl::buffer<int32_t, 1> dummy_buffer(&a, sycl::range<1>(1));
        event = queue_cpu.submit([&](sycl::handler &handle) {
            auto dummy_acc =
                dummy_buffer.get_access<sycl::access::mode::read>(handle);
            handle.single_task<class kernel_dummy_1>([=]() {
                // delay
                int x = 1;
                for (int i = 0; i < 100000000; i++) {
                    x += i;
                }
                printf("kernel_dummy_1\n");
            });
        });
    }
    queue_cpu.submit([&](sycl::handler &handle) {
        handle.single_task<class kernel_dummy_3>(
            [=]() { printf("kernel_dummy_3\n"); });
    });
    // NOTE: 使用 event.wait 来协调两个不同的 queue, 针对同一个 queue 一般是不需
    // 要用 event.wait 的, 因为同一个 queue 从`概念`上是顺序执行的
    event.wait();
    queue_gpu.submit([&](sycl::handler &handle) {
        handle.single_task<class kernel_dummy_2>(
            [=]() { printf("kernel_dummy_2\n"); });
    });
    // NOTE: event 还可以用来获得 submit/start/end 时间
    auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();

    std::cout << "kernel_dummy_1 elapsed time: " << (end - start) / 1.0e6
              << " ms\n";

    queue_gpu.wait();
    queue_cpu.wait();

    return 0;
}
