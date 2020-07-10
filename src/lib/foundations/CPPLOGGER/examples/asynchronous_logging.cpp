/*
 * asynchronous_logging.cpp
 *
 *  Created on: Jun 8, 2020
 *      Author: sajeeb
 */

#include "CPPLOGGER_ASYNC.h"

logger_init();

int main() {
	logger_async_info("This message is not asynchronously logged.");
	logger_async_info("This message is asynchronously logged");

	logger_error("This error message is not asynchronously logged.");
	logger_async_error("This error message is asynchronously logged");

	logger_warning("This warning message is not asynchronously logged.");
	logger_async_warning("This warning message is asynchronously logged");
}
