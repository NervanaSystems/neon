# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Shared-memory based IPC for accepting data from third-party applications.
"""

import numpy as np
import posix_ipc as ipc
import mmap
import struct
import logging

logger = logging.getLogger(__name__)


class Message(object):
    def __init__(self, suffix, header_format):
        self.header_size = struct.calcsize(header_format)
        self.header_format = header_format

        self.shmem_name = '/neon-shmem-' + suffix
        self.empty_sem_name = '/neon-empty-sem-' + suffix
        self.fill_sem_name = '/neon-fill-sem-' + suffix
        self.mutex_name = '/neon-mutex-' + suffix

        self.memory = None
        self.mapfile = None
        self.empty_sem = None
        self.fill_sem = None
        self.mutex = None

    def create(self, data_size):
        self.data_size = data_size
        self.shmem_size = self.header_size + self.data_size
        self.memory, self.mapfile = self.create_shmem(self.shmem_name,
                                                      self.shmem_size)
        self.empty_sem = self.create_sem(self.empty_sem_name, 1)
        self.fill_sem = self.create_sem(self.fill_sem_name, 0)
        self.mutex = self.create_sem(self.mutex_name, 1)

    def destroy(self):
        self.destroy_shmem(self.memory, self.mapfile)
        for item in [self.empty_sem, self.fill_sem, self.mutex]:
            self.destroy_sem(item)

    def open(self):
        self.memory, self.mapfile = self.open_shmem(self.shmem_name)
        self.data_size = self.memory.size
        self.empty_sem = self.open_sem(self.empty_sem_name)
        self.fill_sem = self.open_sem(self.fill_sem_name)
        self.mutex = self.open_sem(self.mutex_name)

    def close(self):
        self.close_shmem(self.memory, self.mapfile)
        for item in [self.empty_sem, self.fill_sem, self.mutex]:
            self.close_sem(item)

    def create_sem(self, name, initial_value):
        try:
            sem = ipc.Semaphore(name, ipc.O_CREX, 0660, initial_value)
        except ipc.ExistentialError:
            logger.warning('Deleting semaphore %s', name)
            self.destroy_sem(self.open_sem(name))
            sem = ipc.Semaphore(name, ipc.O_CREX, 0660, initial_value)
        return sem

    def destroy_sem(self, sem):
        if sem is not None:
            sem.unlink()

    def create_shmem(self, name, size):
        try:
            memory = ipc.SharedMemory(name, ipc.O_CREX, 0660, size)
        except ipc.ExistentialError:
            logger.warning('Deleting shared memory %s', name)
            self.destroy_shmem(*self.open_shmem(name))
            memory = ipc.SharedMemory(name, ipc.O_CREX, 0660, size)
        mapfile = mmap.mmap(memory.fd, memory.size)
        return memory, mapfile

    def destroy_shmem(self, memory, mapfile):
        self.close_shmem(memory, mapfile)
        if memory is not None:
            memory.unlink()

    def open_sem(self, name):
        return ipc.Semaphore(name, 0)

    def close_sem(self, sem):
        if sem is not None:
            sem = sem.close()

    def open_shmem(self, name):
        memory = ipc.SharedMemory(name,  0)
        mapfile = mmap.mmap(memory.fd, 0)
        return memory, mapfile

    def close_shmem(self, memory, mapfile):
        if memory is not None:
            memory.close_fd()
        if mapfile is not None:
            mapfile.close()

    def send(self, data, header):
        self.empty_sem.acquire()
        self.mutex.acquire()
        self.mapfile.seek(0)
        if len(header) != 0:
            packed_header = struct.pack(self.header_format, *header)
            self.mapfile.write(packed_header)
        self.mapfile.write(np.getbuffer(data))
        self.mutex.release()
        self.fill_sem.release()

    def receive(self):
        self.fill_sem.acquire()
        self.mutex.acquire()
        self.mapfile.seek(0)
        if self.header_size != 0:
            packed_header = self.mapfile.read(self.header_size)
            header = struct.unpack(self.header_format, packed_header)
        else:
            header = ()
        buf = self.mapfile.read(self.data_size)
        data = np.frombuffer(buf, dtype=np.uint8)
        self.mutex.release()
        self.empty_sem.release()
        return data, header


class Endpoint(object):
    def __init__(self, **kwargs):
        self.req_name = 'req'
        self.res_name = 'res'
        self.req_header_format = ''
        self.res_header_format = ''
        self.__dict__.update(kwargs)
        self.request = Message(self.req_name, self.req_header_format)
        self.response = Message(self.res_name, self.res_header_format)


class Server(Endpoint):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)
        self.start()

    def start(self):
        self.request.create(self.req_size)
        self.response.create(self.res_size)
        logger.info('Started shared-memory server')

    def stop(self):
        self.request.destroy()
        self.response.destroy()

    def send(self, data, header=()):
        self.response.send(data, header)

    def receive(self):
        return self.request.receive()


class Client(Endpoint):
    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
        self.start()

    def start(self):
        self.request.open()
        self.response.open()
        logger.info('Started shared-memory client')

    def stop(self):
        self.request.close()
        self.response.close()

    def send(self, data, header=()):
        self.request.send(data, header)

    def receive(self):
        return self.response.receive()
