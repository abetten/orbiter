/*
 * time_stamp.cpp
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

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This is printed without a time stamp.");

	logger_print_timestamps(true);
	logger_info("This is printed with millisecond time stamp resolution.");

	logger_timestamp_resolution(logger_microsecond);
	logger_info("This is printed with microsecond time stamp resolution.");

	logger_timestamp_resolution(logger_nanosecond);
	logger_info("This is printed with nanosecond time stamp resolution.");

}
