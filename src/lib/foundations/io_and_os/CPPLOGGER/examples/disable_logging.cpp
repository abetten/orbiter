/*
 * disable_logging.cpp
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

	logger_info("This is before disabling global logging.");

	// This disables logging in the current translation unit.
	// To disable global logging use: logger_enable_global(false)
	logger_enable(false);
	logger_info("This should not be printed.");

	logger_enable(true);
	logger_info("This is after enabling global logging.");

}
