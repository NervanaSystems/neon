/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <assert.h>

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>

using std::thread;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::lock_guard;

class ThreadPool {
public:
    explicit ThreadPool(int count)
    : _count(count), _done(false) {
        _stopped = new bool[count];
        for (int i = 0; i < count; i++) {
            _stopped[i] = false;
        }
    }

    virtual ~ThreadPool() {
        for (auto t : _threads) {
            t->join();
            delete t;
        }
        delete[] _stopped;
    }

    virtual void start() {
        for (int i = 0; i < _count; i++) {
            _threads.push_back(new thread(&ThreadPool::run, this, i));
        }
    }

    virtual void stop() {
        _done = true;
    }

    bool stopped() {
        for (int i = 0; i < _count; i++) {
            if (_stopped[i] == false) {
                return false;
            }
        }
        return true;
    }

    void join() {
        for (auto t : _threads) {
            t->join();
        }
    }

protected:
    virtual void work(int id) = 0;

    void run(int id) {
        while (_done == false) {
            work(id);
        }
        _stopped[id] = true;
    }

protected:
    int                         _count;
    vector<thread*>             _threads;
    bool                        _done;
    bool*                       _stopped;
};
