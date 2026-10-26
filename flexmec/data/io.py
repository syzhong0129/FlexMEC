import pickle

def try_read_pkl(pkl_path: str):
    try:
        import sys
        try:
            import numpy as np
            # Support pickle files created with newer NumPy module paths.
            sys.modules.setdefault('numpy._core', np.core)
            sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
            sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
            sys.modules.setdefault('numpy._core.umath', np.core.umath)
        except ImportError:
            pass

        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"直接pickle读取失败: {e}")

    # Fall back to pandas for pickle compatibility.
    try:
        import pandas as pd  # type: ignore
        try:
            return pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"pandas读取失败: {e}")
    except Exception as e:
        print(f"pandas导入失败: {e}")

    raise Exception("无法读取PKL文件")
