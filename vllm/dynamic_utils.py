import threading
from contextlib import contextmanager


class ForegroundBackgroundGate:
    """
    Foreground (前台 / 优先任务):
      - 延迟敏感
      - 永远不被阻塞
      - 可以并发执行

    Background (后台 / 辅助任务):
      - 只能在没有任何 Foreground 运行时执行
      - Foreground 到来时，必须等待
      - 默认互斥（一次只运行一个 Background）
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._active_foreground = 0
        self._background_running = False
        self._exclusive_background_running = False
        self._fair_background_waiters = 0
        self._fair_background_running = False
        self._fair_background_owner: int | None = None

    @contextmanager
    def foreground(self):
        """前台任务：无条件进入"""
        with self._cond:
            self._active_foreground += 1

        try:
            yield
        finally:
            with self._cond:
                self._active_foreground -= 1
                if self._active_foreground == 0:
                    # 前台清空，唤醒后台
                    self._cond.notify_all()

    @contextmanager
    def migration_foreground(self):
        """Foreground section that yields to an active migration transfer."""
        with self._cond:
            while (self._exclusive_background_running
                   or self._fair_background_running
                   or self._fair_background_waiters > 0):
                self._cond.wait()
            self._active_foreground += 1

        try:
            yield
        finally:
            with self._cond:
                self._active_foreground -= 1
                if self._active_foreground == 0:
                    self._cond.notify_all()

    @contextmanager
    def background(self):
        """后台任务：仅在系统空闲时运行"""
        with self._cond:
            while self._active_foreground > 0 or self._background_running:
                self._cond.wait()
            self._background_running = True

        try:
            yield
        finally:
            with self._cond:
                self._background_running = False
                self._cond.notify_all()

    @contextmanager
    def fair_background(self):
        """Run one background chunk before newly arriving forwards."""
        owner = threading.get_ident()
        nested = False
        with self._cond:
            if self._fair_background_owner == owner:
                nested = True
            else:
                self._fair_background_waiters += 1
                try:
                    while (self._active_foreground > 0
                           or self._background_running):
                        self._cond.wait()
                    self._background_running = True
                    self._fair_background_running = True
                    self._fair_background_owner = owner
                finally:
                    self._fair_background_waiters -= 1

        try:
            yield
        finally:
            if not nested:
                with self._cond:
                    self._fair_background_owner = None
                    self._fair_background_running = False
                    self._background_running = False
                    self._cond.notify_all()

    @contextmanager
    def exclusive_background(self):
        """Background section that also blocks migration-aware foreground."""
        with self._cond:
            while self._active_foreground > 0 or self._background_running:
                self._cond.wait()
            self._background_running = True
            self._exclusive_background_running = True

        try:
            yield
        finally:
            with self._cond:
                self._exclusive_background_running = False
                self._background_running = False
                self._cond.notify_all()
