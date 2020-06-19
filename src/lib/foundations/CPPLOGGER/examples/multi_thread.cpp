/*
 * multi_thread.cpp
 *
 *  Created on: May 30, 2020
 *      Author: sajeeb
 */

#include "../CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

void print_thread(uint id) {
	logger_info_mt("%s: %03d", logger_red("Message from thread"), id);
	logger_info_mt("Message from thread: %03d", id);
}

int main() {
	const uint n_threads = 200;
	std::thread threads[n_threads];
	for (uint8_t i=0; i<n_threads; ++i)
		threads[i] = std::thread(print_thread, i+1);
	for (std::thread& t : threads) t.join();
}
