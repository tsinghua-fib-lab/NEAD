import hashlib

def my_hash(x, bits=32):
    """Return a reproducible SHA-256-based hash instead of Python's salted hash."""
    h = hashlib.sha256(str(x).encode()).digest()
    return int.from_bytes(h[:bits // 8], "little") # Use the first `bits` bits.
