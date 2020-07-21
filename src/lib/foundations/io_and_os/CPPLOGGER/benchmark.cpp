/*
 * main.cpp
 *
 *  Created on: May 27, 2020
 *      Author: sajeeb
 */

#include <assert.h>
#include <iostream>
#include <thread>

#include "CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

FILE* null_file = fopen("/dev/null", "w");

void bench(size_t iters, const char* msg) {
	using std::chrono::duration;
	using std::chrono::duration_cast;
	using std::chrono::high_resolution_clock;

	auto start = high_resolution_clock::now();
	for(size_t i=0; i<iters; ++i) {
		logger_info("Hello logger: msg number %ld", i);
	}

	auto delta = high_resolution_clock::now() - start;
	auto delta_d = duration_cast<duration<double>>(delta).count();

	logger_output_stream(stdout);
	logger_enable_global(true);
	logger_info("%-16s| Elapsed: %04.2f secs | Throughput: %'d/sec", msg, delta_d, int(iters/delta_d));
//	assert(logger::queue._last.load()->stream == stdout);
}

void single_threaded(size_t iters) {
	logger_info("*******************************************************************");
	logger_info("Single threaded benchmark with %'ld iterations", iters);
	logger_info("*******************************************************************");

	logger_output_stream(null_file);
	bench(iters, "basic");

	logger_output_stream(null_file);
	logger_enable_global(false);
	bench(iters, "disabled");
}

void multi_thread_bench(size_t n_threads, size_t iters, string type) {
	using std::chrono::duration;
	using std::chrono::duration_cast;
	using std::chrono::high_resolution_clock;

	std::thread _threads[n_threads];
	for (size_t i=0; i<n_threads; ++i) _threads[i] = std::thread([&]{
        for (int j = 0; j < int(iters / n_threads); j++) {
			logger_info_mt("Hello logger: msg number %d", j);
        }
    });

	auto start = high_resolution_clock::now();

	for (auto &t : _threads) {
		t.join();
	};

	auto delta = high_resolution_clock::now() - start;
	auto delta_d = duration_cast<duration<double>>(delta).count();

	logger_output_stream(stdout);
	logger_enable_global(true);
	logger_info("%-16s| Elapsed: %04.2f secs | Throughput: %'d/sec", type.c_str(), delta_d, int(iters/delta_d));
}

void multi_threaded(size_t threads, size_t iters) {
	logger_info("*******************************************************************");
	logger_info("%'d thread benchmark with %'ld iterations", threads, iters);
	logger_info("*******************************************************************");

	logger_output_stream(null_file);
	multi_thread_bench(threads, iters, "basic");

	logger_output_stream(null_file);
	logger_enable_global(false);
	multi_thread_bench(threads, iters, "disabled");
}

int main(int argc, char** argv) {
	logger_print_file(false);
	logger_print_line(false);
//	logger_print_log_type(false);

	setlocale(LC_NUMERIC, "");
	unsigned int iters = 1000000;
	size_t threads = 10;

	for (uint8_t i=0; i<argc; ++i) {
		if (strcmp(argv[i], "--threads") == 0) {
			threads = atoi(argv[i+1]);
			i += 1;

		} else if (strcmp(argv[i], "--iters") == 0) {
			iters = atoi(argv[i+1]);
			i += 1;
		}
	}

	single_threaded(iters);
	logger_info("");
	multi_threaded(threads, iters);

	fclose(null_file);
}
