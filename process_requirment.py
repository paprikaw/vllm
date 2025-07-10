# SPDX-License-Identifier: Apache-2.0

import glob

requires_files = glob.glob('requirements/*.txt')
requires_files += ["pyproject.toml"]
for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    if "torch" in "".join(lines).lower() or "mamba" in "".join(lines).lower() or "timm" in "".join(lines).lower() or "encodec" in "".join(lines).lower() or "vocos" in "".join(lines).lower():
        print("removed:")
        with open(file, 'w') as f:
            for line in lines:
                if 'torch' not in line.lower() and 'mamba' not in line.lower() and 'timm' not in line.lower() and 'encodec' not in line.lower() and 'vocos' not in line.lower():
                    f.write(line)
                else:
                    print(line.strip())
    print(f"<<< done cleaning {file}")
    print()