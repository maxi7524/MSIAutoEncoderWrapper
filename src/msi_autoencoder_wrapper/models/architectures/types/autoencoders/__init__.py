"""
Initialization package tracking clean network topology subcomponents
"""

import pkgutil
import importlib
import sys

# Automated strategy subcomponents discovery block
## Scan package directory layout structure for dynamic registrations loops
for _, module_name, _ in pkgutil.iter_modules(__path__):
    full_module_name = f"{__name__}.{module_name}"
    if full_module_name not in sys.modules:
        importlib.import_module(full_module_name)