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