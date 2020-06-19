/*
 * CPPLOGGER_ASYNC.cpp
 *
 *  Created on: Jun 12, 2020
 *      Author: sajeeb
 */

/**
 * macro to initialize the extern variables in logger
 */
#define logger_init() \
namespace logger { \
	bool _print_timestamps_ = false; \
	bool _print_thread_id_ = false; \
	bool _print_log_type_ = true; \
	bool _print_file_ = true; \
	bool _print_line_ = true; \
	bool _enable_global_ = true; \
	bool _flush_immediately_ = false; \
	mutex critical_section; \
	va_list __args__; \
	FILE* _output_stream_ = stdout; \
	lock_free_queue queue;\
	timestamp_resolution _resolution = logger::timestamp_resolution::millisecond;\
	log_level _loglevel_global_ = logger::log_level::all;\
}

#define logger_output_stream(v) \
		logger::_output_stream_ = v;\
		logger::queue.set_output_stream((FILE*)v)

#include "CPPLOGGER_SYNC.h"

namespace logger {

	#define logger_async_info(...) logger::_info_async_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_async_error(...) logger::_error_async_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_async_warning(...) logger::_warning_async_(__FILE__, __LINE__, __VA_ARGS__)

	/**
	 *
	 */
	extern lock_free_queue queue;

	/**
	 * wrapper function containing the boilerplate code for multi-threaded
	 * logging
	 */
	#define __CPPLOGGER_ASYNC__(fmt, color, type) {\
		va_list __args__; \
		va_start(__args__, fmt); \
		lock_free_queue::node* _node = queue.get();\
		char* b = _node->data;\
		if (_print_log_type_) { \
			b += sprintf(b, "[" color type ANSI_RESET "]"); \
		} \
		if (_print_timestamps_) { \
			b += __print_timestamp_sprintf__(b);\
		} \
		if (_print_thread_id_) { \
			std::stringstream ss; ss << std::this_thread::get_id(); \
			const unsigned long long int id = std::stoull(ss.str()); \
			b += sprintf(b, "[%04lld]", id); \
		} \
		if (_print_log_type_ || _print_timestamps_ || _print_thread_id_) { \
			b += sprintf(b, ": "); \
		} \
		if (_print_file_) { \
			b += sprintf(b, "%s:", _file_); \
		} \
		if (_print_line_) { \
			b += sprintf(b, "%d:", line); \
		} \
		if (_print_file_ || _print_line_) { \
			b += sprintf(b, ": "); \
		} \
		b += vsprintf(b, fmt, __args__); \
		b += sprintf(b, ANSI_RESET "\n"); \
		va_end(__args__);\
		_node->done = true;\
	}

	/**
	 *
	 */

	inline void _info_async_(const char* _file_, const int line, const int cvl, const int avl, const char* fmt, ...) {
		if (_enable_global_ && _enable_translation_uint_ && cvl >= avl) {
			if (_loglevel_global_ == all || _loglevel_global_ == info) {
				__CPPLOGGER_ASYNC__(fmt, ANSI_GREEN, "INFO");
			}
		}
	}

	inline void _info_async_(const char* _file_, const int line, const char* fmt, ...) {
		if (_enable_global_ && _enable_translation_uint_) {
			if (_loglevel_global_ == all || _loglevel_global_ == info) {
				__CPPLOGGER_ASYNC__(fmt, ANSI_GREEN, "INFO");

			}
		}
	}

	/**
	 *
	 */

	inline void _error_async_(const char* _file_, const int line, const int cvl, const int avl, const char* fmt, ...) {
		if (_enable_global_ && _enable_translation_uint_ && cvl >= avl) {
			if (_loglevel_global_ == all || _loglevel_global_ == error) {
				__CPPLOGGER_ASYNC__(fmt, ANSI_RED, "ERROR");
			}
		}
	}

	inline void _error_async_(const char* _file_, const int line, const char* fmt, ...) {
		if (_enable_global_ && _enable_translation_uint_) {
			if (_loglevel_global_ == all || _loglevel_global_ == error) {
				__CPPLOGGER_ASYNC__(fmt, ANSI_RED, "ERROR");
			}
		}
	}

	/**
	 *
	 */

	inline void _warning_async_(const char* _file_, const int line, const int cvl, const int avl, const char* fmt, ...) {
		if (_enable_global_ && _enable_translation_uint_ && cvl >= avl) {
			if (_loglevel_global_ == all || _loglevel_global_ == warning) {
				__CPPLOGGER_ASYNC__(fmt, ANSI_BLUE, "WARN");
			}
		}
	}

	inline void _warning_async_(const char* _file_, const int line, const char* fmt, ...) {
		if (_enable_global_ && _enable_translation_uint_) {
			if (_loglevel_global_ == all || _loglevel_global_ == warning) {
				__CPPLOGGER_ASYNC__(fmt, ANSI_BLUE, "WARN");
			}
		}
	}

}
