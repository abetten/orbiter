/*
 * no_file_line.cpp
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

	logger_info("This message has both line number and file");

	logger_print_file(false);
	logger_print_line(false);
	logger_info("This message does not have either a file or a line number.");

}
