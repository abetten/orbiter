/*
 * basic_usage.cpp
 *
 *  Created on: May 29, 2020
 *      Author: sajeeb
 */

#include "../CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info("-------------------------------------------------------------");
	logger_info("Welcome to CPPLOGGER!");
	logger_info("-------------------------------------------------------------");

	logger_info("The following shows some basic usage.");

	logger_error("This is an error message with some %s", "argument");
	logger_warning("This is a warning message with 2 args: (%.3f, %.4f)", 1.2323745, 2.5);
	logger_info("This is an info message with three arguments: (%03d, %03d, %03d)", 0, 50, 100);
	logger_info("");

}
