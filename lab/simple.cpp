//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl::sycl;

const float a = 1.0f;
const float b = 0.1f;
const float c = 0.01f;
const float d = 1.5f;

int main() {
    const float dt = 0.01f; // time step
    const int N = 1000; // number of time steps

    std::vector<float> prey(N);
    std::vector<float> predator(N);

    prey[0] = 10.0f;
    predator[0] = 5.0f;

    // Create buffers
    buffer<float, 1> prey_buf(prey.data(), range<1>(N));
    buffer<float, 1> predator_buf(predator.data(), range<1>(N));

    queue q;

    q.submit([&](handler& cgh) {
        auto prey_acc = prey_buf.get_access<access::mode::read_write>(cgh);
        auto predator_acc = predator_buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class simulation_kernel>(range<1>(N-1), [=](id<1> i) {
            // Lotka-Volterra equations
            prey_acc[i+1] = prey_acc[i] + dt * (a * prey_acc[i] - b * prey_acc[i] * predator_acc[i]);
            predator_acc[i+1] = predator_acc[i] + dt * (-c * predator_acc[i] + d * prey_acc[i] * predator_acc[i]);

            // Ensure non-negative populations
            if (prey_acc[i+1] < 0.0f) {
                prey_acc[i+1] = 0.0f;
            }
            if (predator_acc[i+1] < 0.0f) {
                predator_acc[i+1] = 0.0f;
            }
        });
    });

    q.wait();

    std::cout << "Final prey population: " << prey_buf.get_access<access::mode::read>()[N-1] << std::endl;
    std::cout << "Final predator population: " << predator_buf.get_access<access::mode::read>()[N-1] << std::endl;

    return 0;
}

