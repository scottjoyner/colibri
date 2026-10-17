#!/usr/bin/env python3
"""Bounded session-fenced two-reader executor for shadow SSD I/O only.

Never part of a live model runtime. A batch admits <=2 reads, each a 4MiB
read into a temporary aligned buffer. Session/generation revocation prevents
later admissions and drains active reads before returning.
"""
from __future__ import annotations
import concurrent.futures
import dataclasses
import threading
from typing import Callable

@dataclasses.dataclass(frozen=True)
class ReadKey:
    path: str
    offset: int

class AdmissionDenied(RuntimeError):
    pass

class BoundedShadowReader:
    def __init__(self, *, session: str, pin: str, generation: int,
                 read_fn: Callable, max_workers: int=2,
                 bytes_per_read: int=4*1024*1024, max_bytes: int=8*1024*1024):
        if (not session or len(pin)!=40 or
            any(c not in "0123456789abcdef" for c in pin) or
            generation<0 or max_workers!=2 or bytes_per_read<=0 or
            max_bytes!=2*bytes_per_read):
            raise ValueError("NO_GO invalid fixed two-reader ownership or budget")
        self.session=session
        self.pin=pin
        self.generation=generation
        self.max_workers=max_workers
        self.bytes_per_read=bytes_per_read
        self.max_bytes=max_bytes
        self.read_fn=read_fn
        self._lock=threading.RLock()
        self._enabled=True
        self._shutdown=False
        self._batch_active=False
        self._pool=concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.submitted=0
        self.completed=0

    def read_batch(self, *, session: str, pin: str, generation: int,
                   requests: list[tuple[int,int]])->list[float]:
        if len(requests)>self.max_workers or len(requests)*self.bytes_per_read>self.max_bytes:
            raise AdmissionDenied("NO_GO batch exceeds inflight reservation")
        with self._lock:
            if (not self._enabled or self._shutdown or
                session!=self.session or pin!=self.pin or generation!=self.generation):
                raise AdmissionDenied("NO_GO stale or revoked session/pin/generation")
            if self._batch_active:
                raise AdmissionDenied("NO_GO another two-reader batch owns the reservation")
            self._batch_active=True
            # Only ONE bounded batch may be admitted at a time. New generation
            # admission cannot overlap it, and revoke prevents further batches.
            futures=[self._pool.submit(self.read_fn,fd,off) for fd,off in requests]
            self.submitted+=len(futures)
        try:
            result=[f.result(timeout=10) for f in futures]
            with self._lock:
                # If the session was revoked mid-read, discard the completed
                # timing rather than returning stale work to a caller.
                if not self._enabled:
                    raise AdmissionDenied("NO_GO session revoked before read completion")
                self.completed+=len(futures)
            return result
        except BaseException:
            with self._lock:self._enabled=False
            for f in futures:f.cancel()
            # Do not discard any mmap/FD while a native read can still be active.
            concurrent.futures.wait(futures)
            raise
        finally:
            with self._lock:self._batch_active=False

    def revoke(self)->None:
        with self._lock:self._enabled=False

    def close(self)->None:
        with self._lock:
            if self._shutdown:return
            self._shutdown=True
            self._enabled=False
        # Tasks cannot be safely abandoned while their O_DIRECT mmap exists.
        self._pool.shutdown(wait=True,cancel_futures=True)

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
