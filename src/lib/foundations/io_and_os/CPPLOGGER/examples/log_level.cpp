/*
 * log_level.cpp
 *
 *  Created on: Jun 14, 2020
 *      Author: sajeeb
 */

#include <CPPLOGGER_SYNC.h>

logger_init();

int main() {

	logger_log_level(logger_loglevel_warning);
	logger_info("This message should not be printed.");
	logger_error("This message should not be printed");
	logger_warning("This warning message should be printed");

	logger_log_level(logger_loglevel_error);
	logger_info("This info message should not be printed.");
	logger_error("This error message should be printed");
	logger_warning("This warning message should not be printed");

	logger_log_level(logger_loglevel_info);
	logger_info("This info message should be printed.");
	logger_error("This error message should not be printed");
	logger_warning("This warning message should not be printed");

	logger_log_level(logger_loglevel_all);
	logger_info("This info message should be printed.");
	logger_error("This error message should be printed");
	logger_warning("This warning message should be printed");

}
