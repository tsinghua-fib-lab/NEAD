import hashlib

def my_hash(x, bits=32):
    """ 直接用 hash(x) 得到的值在不同的 Python 运行环境中可能不一样，所以改用 sha256 来保证可复现性 """
    h = hashlib.sha256(str(x).encode()).digest()
    return int.from_bytes(h[:bits // 8], "little") # 取前 bits 位的数据