/*
 * log_type.cpp
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

	logger_info("This message is printed with log type enabled.");

	logger_print_log_type(false);
	logger_info("This message is printed without the log type.");

}
