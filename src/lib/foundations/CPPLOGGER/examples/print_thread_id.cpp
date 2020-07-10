/*
 * print_thread_id.cpp
 *
 *  Created on: May 30, 2020
 *      Author: sajeeb
 */

#include <iostream>
#include <chrono>
#include <thread>
#include "CPPLOGGER_ASYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

void print_thread(uint id) {
	for (int i=0; i<2; ++i) {
		logger_info_mt("%s: %03d | %03d", logger_red("Message from thread"), id, i);
		logger_info_mt("Message from thread: %03d | %03d", id, i+1);
	}
}

int main() {
	logger_print_file(false);
	logger_print_line(false);

	const uint n_threads = 200;
	std::thread threads[n_threads];
	logger_print_thread_id(true);
	for (uint8_t i=0; i<n_threads; ++i)
		threads[i] = std::thread(print_thread, i+1);
	for (std::thread& t : threads) t.join();

	logger_async_info("Done executing.");
}
