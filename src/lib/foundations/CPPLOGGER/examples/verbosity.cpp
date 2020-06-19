/*
 * verbosity.cpp
 *
 *  Created on: Jun 1, 2020
 *      Author: sajeeb
 */

#include "../CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {
	int verbose_level = 0;

	logger_info(verbose_level, 1, "This should only be printed at verbose level 1.");

	logger_info("Current verbose level: %d", verbose_level);
	logger_info("Incrementing verbose level by one: %d", ++verbose_level);
	logger_info("Current verbose level: %d", verbose_level);

	logger_info(verbose_level, 1, "This should only be printed at verbose level 1.");
}
