import threading

from vllm.dynamic_utils import ForegroundBackgroundGate


def test_exclusive_background_blocks_new_migration_foreground():
    gate = ForegroundBackgroundGate()
    exclusive_entered = threading.Event()
    release_exclusive = threading.Event()
    forward_entered = threading.Event()

    def unmap():
        with gate.exclusive_background():
            exclusive_entered.set()
            assert release_exclusive.wait(timeout=2)

    def forward():
        with gate.migration_foreground():
            forward_entered.set()

    unmap_thread = threading.Thread(target=unmap)
    unmap_thread.start()
    assert exclusive_entered.wait(timeout=2)

    forward_thread = threading.Thread(target=forward)
    forward_thread.start()
    assert not forward_entered.wait(timeout=0.05)

    release_exclusive.set()
    unmap_thread.join(timeout=2)
    forward_thread.join(timeout=2)

    assert not unmap_thread.is_alive()
    assert not forward_thread.is_alive()
    assert forward_entered.is_set()


def test_waiting_fair_background_blocks_new_migration_foreground():
    gate = ForegroundBackgroundGate()
    release_first_forward = threading.Event()
    fair_entered = threading.Event()
    later_forward_entered = threading.Event()

    def first_forward():
        with gate.migration_foreground():
            assert release_first_forward.wait(timeout=2)

    def fair_background():
        with gate.fair_background():
            fair_entered.set()

    def later_forward():
        with gate.migration_foreground():
            later_forward_entered.set()

    first = threading.Thread(target=first_forward)
    first.start()
    background = threading.Thread(target=fair_background)
    background.start()

    for _ in range(100):
        with gate._cond:
            if gate._fair_background_waiters:
                break
        threading.Event().wait(0.005)
    else:
        raise AssertionError("fair background waiter was not registered")

    later = threading.Thread(target=later_forward)
    later.start()
    assert not later_forward_entered.wait(timeout=0.05)

    release_first_forward.set()
    assert fair_entered.wait(timeout=2)
    first.join(timeout=2)
    background.join(timeout=2)
    later.join(timeout=2)

    assert not first.is_alive()
    assert not background.is_alive()
    assert not later.is_alive()
    assert later_forward_entered.is_set()


def test_fair_background_is_reentrant_for_weight_chunks():
    gate = ForegroundBackgroundGate()
    with gate.fair_background():
        with gate.fair_background():
            assert gate._fair_background_running
    assert not gate._fair_background_running
