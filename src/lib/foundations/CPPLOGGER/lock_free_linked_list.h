/*
 * lock_free_queue.h
 *
 *  Created on: Jun 5, 2020
 *      Author: sajeeb
 */

#ifndef LOCK_FREE_LINKED_LIST_H_
#define LOCK_FREE_LINKED_LIST_H_

#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <atomic>
#include <mutex>
#include <thread>
#include <assert.h>
#include <condition_variable>

//#define __LOGGER_DEBUGGER__
#include "logger_debugger.h"

using std::atomic;
using std::thread;
using std::condition_variable;
using std::mutex;

#define __force_inline__ __attribute__((always_inline)) inline

class lock_free_queue {
public:
	typedef struct node {
		char data[256];
		node* next = nullptr;
		bool done = false;
		FILE* stream;
		node(FILE* os) : stream(os) {}
	} node;

	atomic<node*> _first;
	atomic<node*> _last;
	atomic<int> num_nodes;
	atomic<bool> __kill__;
	atomic<bool> __terminated__;
	uint flush_interval = pow(2, 13);
	atomic<uint> current_flush_count;
	atomic<FILE*> _output_stream_;
	thread printer;
	condition_variable cv, cv1;
	mutex mtx, mtx1;

	lock_free_queue() {
		num_nodes.store(0);
		current_flush_count.store(0);
		__terminated__.store(false);
		__kill__.store(false);

		_first.store(NULL);
		_last.store(NULL);

		set_output_stream(stdout);
		printer = thread(lock_free_queue::run, std::ref(*this));
		printer.detach();
	}

	~lock_free_queue() {
		__LOGGER_DEBUGGER__("In Destructor.");
		__LOGGER_DEBUGGER__("Terminating print thread");
		__kill__.store(true);
		while (!__terminated__.load()) {
			running = true;
			cv.notify_one();
		}
		__LOGGER_DEBUGGER__("Print thread terminated.");
		__LOGGER_DEBUGGER__("Number of nodes remaining: %d", num_nodes.load());
		flush_all();
	}

	__force_inline__
	node* get() {
		return append();
	}

	__force_inline__
	void set_output_stream(FILE* os) {
//		__LOGGER_DEBUGGER__("Value before setting stream: %p", _output_stream_.load());
		_output_stream_.store(os);
//		setbuf(_output_stream_.load(), NULL);
//		__LOGGER_DEBUGGER__("Value after setting stream: %p", _output_stream_.load());
	}


private:
	atomic<bool> running;
	/**
	 * Only gets called in the get function
	 */
	__force_inline__
	node* append() {
		if (num_nodes % flush_interval == 0) {
			running = false;
			std::unique_lock<mutex> lock(mtx1);
			cv1.wait_for(lock,  std::chrono::nanoseconds(0), [&]{
				return (num_nodes % flush_interval);
			});
		}
		running = true;
//		__LOGGER_DEBUGGER__("Appending node..");
		node* n = new node(_output_stream_);
		++num_nodes;
//		__LOGGER_DEBUGGER__("new node: %p", n);
		node* prev = _last.exchange(n);
//		__LOGGER_DEBUGGER__("Checking previous node pointer: %p", prev);
		(prev) ? prev->next = n : _first = n;
//		__LOGGER_DEBUGGER__("new first: %p", _first.load());
//		__LOGGER_DEBUGGER__("Done appending node.");
//		if (++current_flush_count % flush_interval) cv.notify_one();
//		cv.notify_one();
		cv1.notify_all();
		return n;
	}

	/**
	 * Gets called in flush and run method
	 */
	__force_inline__
	void flush(node* n) {
//		__LOGGER_DEBUGGER__("flush");
//		__LOGGER_DEBUGGER__("Setting contents of _first to: %p", n->next);
		_first.store(n->next);
//		__LOGGER_DEBUGGER__("Printing contents of node: %p", n);
//		setbuf(n->stream, NULL);
		fprintf(n->stream, "%s", n->data);
//		__LOGGER_DEBUGGER__("Printed: %s to %p", n->data, n->stream);
//		__LOGGER_DEBUGGER__("DELETE: %p", n);
		delete n;
		--num_nodes;
	}

	/**
	 * Gets called only in the destructor
	 */
	__force_inline__
	void flush_all() {
//		__LOGGER_DEBUGGER__("flush_all..");
		std::lock_guard<mutex> lock(mtx);
		node* f;
		node* prev_last_node = _last;
		while ((f = _first) != nullptr) {
			flush(f);
		}
		_last.compare_exchange_strong(prev_last_node, f);
		running = true;
	}

	/**
	 *
	 */
	__force_inline__
	static void run(lock_free_queue& q) {
		std::unique_lock<std::mutex> lock(q.mtx);
//		q.cv.wait_for(lock, std::chrono::seconds(1), [&]{
		q.cv.wait(lock, [&]{
//			if (q.num_nodes < 8192) return false;
			node* f = q._first;
//			__LOGGER_DEBUGGER__("f: %p, l: %p", f, l);
			for (;f != q._last && f != NULL && f->done;) {
//				__LOGGER_DEBUGGER__("Printer thread flushing node: %p", f);
				fprintf(f->stream, "%s", f->data);
				const node* n = f;
				f = f->next;
				delete n;
				--q.num_nodes;
			}
			while (!q.running) q.cv1.notify_all();
			q._first = f;
//			__LOGGER_DEBUGGER__("num_nodes: %d", q.num_nodes.load());
//			__LOGGER_DEBUGGER__("Printer thread done flushing");
			return (q.__kill__.load()) ? true : false;
		});
		while (!q.__terminated__.load()) q.__terminated__.store(true);
	}
};

#endif /* LOCK_FREE_LINKED_LIST_H_ */
