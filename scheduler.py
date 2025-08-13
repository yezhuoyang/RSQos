from kernel import *
from process import *
from instruction import *
from hardware import *
from instruction import *
from syscall import *



class Scheduler:
    def __init__(self, kernel_instance: Kernel, hardware_instance: virtualHardware):
        self._kernel = kernel_instance
        self._hardware = hardware_instance
        self._virtual_hardware_mapping = virtualHardwareMapping(hardware_instance)


    def schedule(self):
        """
        Schedule processes for execution.
        The input is a kernel with a process queue.
        The output is a single process instance with virtual hardware mapping.
        """
        processes = self._kernel.get_processes()
        if not processes:
            return None
        pass
