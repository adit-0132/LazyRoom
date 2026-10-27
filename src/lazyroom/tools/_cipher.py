"""ImageToolbox-compatible file encryption (port of ``feature/cipher``).

On-disk format (``AndroidCryptographyManager.EncryptedFileHeader``, version 1)::

    "ITBXCRYP" | u8 version=1 | u8 kdf=1 (PBKDF2WithHmacSHA256) | i32 iterations (BE)
    | u16 key size in bits (128/256, BE) | u8 salt size (=16) | u8 nonce size | u16 cipher-name size (BE)
    | salt | nonce | cipher name (JCA transformation, ASCII) | ciphertext

The key is PBKDF2-HMAC-SHA256(password UTF-8, salt, iterations, key size).  For AEAD
ciphers the encoded header is the additional authenticated data.  For ``PBE*`` ciphers
the PBKDF2 output is hex-encoded and used as the *password* of the BouncyCastle PBE
scheme named by the cipher (with the header salt and iteration count).  Files without
the magic are "legacy" files: key = MD5 / SHA-256 of the password and IV = first block
of the key (decryption only, like ImageToolbox).

ImageToolbox lists whatever ``Cipher`` algorithms the device's security providers expose
(Conscrypt + Android BC + bundled BouncyCastle), filtered by ``CipherType.BROKEN``.  The
table below is that list reconstructed from BouncyCastle 1.77 and Conscrypt; behaviour of
every supported entry was checked against files produced by the real BouncyCastle
provider (see tests/test_file_tools.py).  ``cryptography`` is an optional dependency.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import struct
import warnings
from dataclasses import dataclass
from typing import Any

from ..core.errors import ErrorCode, LazyRoomError

MAGIC = b"ITBXCRYP"
VERSION = 1
KDF_PBKDF2_SHA256 = 1
FIXED_HEADER_SIZE = 20
PBKDF2_ITERATIONS = 210_000
SALT_SIZE = 16
GCM_NONCE_SIZE = 12
GCM_TAG_SIZE = 16
MAX_CIPHER_NAME_SIZE = 256
MAX_NONCE_SIZE = 64
DEFAULT_ALGORITHM = "AES/GCM/NoPadding"
RANDOM_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DECRYPT_FAILED = "Failed to decrypt file: invalid password or corrupted encrypted data"

# CipherType.BROKEN / AndroidCryptographyManager.MD5List (verbatim)
MD5_LIST = ["SEED", "NOEKEON", "HC128", "AES_128/CBC/NOPADDING", "AES_128/CBC/PKCS5PADDING", "AES_128/GCM/NOPADDING",
            "DESEDE", "GRAIN128", "SM4", "TEA", "ZUC-128", "AES_128/ECB/NOPADDING", "AES_128/ECB/PKCS5PADDING",
            "AES_128/GCM/NOPADDING"]


@dataclass(frozen=True)
class Impl:
    alg: str                    # aes | camellia | sm4 | 3des | des | blowfish | idea | seed | rc2 | rc4
    mode: str                   # gcm | ecb | cbc | ctr | stream
    pad: bool = False           # PKCS#7 padding
    key_bytes: int | None = None  # bytes of the derived key the provider actually uses (None = all)
    pbe: tuple[str, str, int, int] | None = None  # (scheme pkcs12|pkcs5s1|openssl, digest, key bytes, iv bytes)


@dataclass(frozen=True)
class CipherDef:
    cipher: str                 # JCA name stored in the header (CipherType.cipher)
    name: str                   # ImageToolbox display name (CipherType.name)
    impl: Impl | None
    note: str = ""
    provider: str = "BC"


def _pbe12(digest: str, alg: str, key: int, iv: int, mode: str = "cbc", pad: bool = True) -> Impl:
    return Impl(alg, mode, pad, pbe=("pkcs12", digest, key, iv))


_NO_ARIA = "ARIA is not implemented by the 'cryptography' package"
_IV_BUG = ("broken in ImageToolbox itself: its IV size rule gives 8 bytes for this 16-byte-block CBC cipher, so "
           "BouncyCastle encrypts with a random IV that is not stored and the file cannot be decrypted")
_NA = "not implemented by the 'cryptography' package"


def _na(cipher: str, name: str | None = None, why: str = _NA) -> CipherDef:
    return CipherDef(cipher, name or cipher, None, why)


# Order = ImageToolbox's CipherType.entries (default first, then sorted by display name).
CIPHERS: list[CipherDef] = [
    CipherDef("AES/GCM/NoPadding", "AES/GCM/NoPadding", Impl("aes", "gcm"), "ImageToolbox default", "default"),
    CipherDef("AES", "AES", Impl("aes", "ecb", True), "BouncyCastle default mode: ECB/PKCS7Padding"),
    CipherDef("AES/CTR/NOPADDING", "AES/CTR/NOPADDING", Impl("aes", "ctr"), "", "Conscrypt"),
    CipherDef("AES/ECB/PKCS5PADDING", "AES/ECB/PKCS5PADDING", Impl("aes", "ecb", True), "", "Conscrypt"),
    CipherDef("AES/GCM/NOPADDING", "AES/GCM/NOPADDING", Impl("aes", "gcm"), "", "Conscrypt"),
    CipherDef("AES_256/ECB/PKCS5PADDING", "AES_256/ECB/PKCS5PADDING", Impl("aes", "ecb", True), "", "Conscrypt"),
    CipherDef("ARC4", "ARC4", Impl("rc4", "stream"), "RC4 with the 256-bit derived key; IV ignored"),
    _na("ARIA", why=_NO_ARIA),
    _na("1.2.410.200046.1.1.3", "ARIA128/CFB", _NO_ARIA),
    _na("1.2.410.200046.1.1.4", "ARIA128/OFB", _NO_ARIA),
    _na("1.2.410.200046.1.1.8", "ARIA192/CFB", _NO_ARIA),
    _na("1.2.410.200046.1.1.6", "ARIA192/ECB", _NO_ARIA),
    _na("1.2.410.200046.1.1.9", "ARIA192/OFB", _NO_ARIA),
    _na("1.2.410.200046.1.1.13", "ARIA256/CFB", _NO_ARIA),
    _na("1.2.410.200046.1.1.11", "ARIA256/ECB", _NO_ARIA),
    _na("1.2.410.200046.1.1.14", "ARIA256/OFB", _NO_ARIA),
    _na("ARIAGCM", why=_NO_ARIA),
    _na("1.2.410.200046.1.1.1", "ASN1-module", "ARIA-128/ECB (ImageToolbox's OID mapping labels it 'ASN1-module'); "
        + _NO_ARIA),
    CipherDef("BLOWFISH", "BLOWFISH", Impl("blowfish", "ecb", True)),
    CipherDef("CAMELLIA", "CAMELLIA", Impl("camellia", "ecb", True)),
    _na("1.2.392.200011.61.1.1.1.2", "CAMELLIA-128/CBC", _IV_BUG),
    _na("1.2.392.200011.61.1.1.1.3", "CAMELLIA-192/CBC", _IV_BUG),
    _na("1.2.392.200011.61.1.1.1.4", "CAMELLIA-256/CBC", _IV_BUG),
    _na("CAST6"),
    CipherDef("DESEDE", "DESEDE", Impl("3des", "ecb", True), "two-key Triple DES (128-bit derived key)"),
    CipherDef("DESEDE/CBC/PKCS5PADDING", "DESEDE/CBC/PKCS5PADDING", Impl("3des", "cbc", True), "", "Conscrypt"),
    CipherDef("1.2.840.113549.3.7", "DESede/CBC/NoPadding", Impl("3des", "cbc", True),
              "BouncyCastle pads this OID with PKCS#7 despite the mapped name"),
    CipherDef("GCM", "GCM", Impl("aes", "gcm"), "BouncyCastle alias of AES/GCM"),
    _na("GOST28147"),
    CipherDef("IDEA", "IDEA", Impl("idea", "ecb", True, key_bytes=16), "BouncyCastle uses the first 128 key bits"),
    _na("NOEKEON"),
    _na("PBEWITHMD2ANDDES", why="MD2 is not available in OpenSSL 3"),
    CipherDef("PBEWITHMD5AND128BITAES-CBC-OPENSSL", "PBEWITHMD5AND128BITAES-CBC-OPENSSL",
              Impl("aes", "cbc", True, pbe=("openssl", "md5", 16, 16))),
    CipherDef("PBEWITHMD5AND192BITAES-CBC-OPENSSL", "PBEWITHMD5AND192BITAES-CBC-OPENSSL",
              Impl("aes", "cbc", True, pbe=("openssl", "md5", 24, 16))),
    CipherDef("PBEWITHMD5AND256BITAES-CBC-OPENSSL", "PBEWITHMD5AND256BITAES-CBC-OPENSSL",
              Impl("aes", "cbc", True, pbe=("openssl", "md5", 32, 16))),
    CipherDef("PBEWITHMD5ANDDES", "PBEWITHMD5ANDDES", Impl("des", "cbc", True, pbe=("pkcs5s1", "md5", 8, 8))),
    _na("PBEWITHMD5ANDRC2", why="64-bit RC2 keys are not supported by 'cryptography' (128-bit only)"),
    CipherDef("PBEWITHSHA1ANDDES", "PBEWITHSHA1ANDDES", Impl("des", "cbc", True, pbe=("pkcs5s1", "sha1", 8, 8))),
    _na("PBEWITHSHA1ANDRC2", why="64-bit RC2 keys are not supported by 'cryptography' (128-bit only)"),
    CipherDef("PBEWITHSHA256AND128BITAES-CBC-BC", "PBEWITHSHA256AND128BITAES-CBC-BC", _pbe12("sha256", "aes", 16, 16)),
    CipherDef("PBEWITHSHA256AND192BITAES-CBC-BC", "PBEWITHSHA256AND192BITAES-CBC-BC", _pbe12("sha256", "aes", 24, 16)),
    CipherDef("PBEWITHSHA256AND256BITAES-CBC-BC", "PBEWITHSHA256AND256BITAES-CBC-BC", _pbe12("sha256", "aes", 32, 16)),
    CipherDef("PBEWITHSHAAND128BITAES-CBC-BC", "PBEWITHSHAAND128BITAES-CBC-BC", _pbe12("sha1", "aes", 16, 16)),
    CipherDef("PBEWITHSHAAND128BITRC2-CBC", "PBEWITHSHAAND128BITRC2-CBC", _pbe12("sha1", "rc2", 16, 8)),
    CipherDef("PBEWITHSHAAND128BITRC4", "PBEWITHSHAAND128BITRC4", _pbe12("sha1", "rc4", 16, 0, "stream", False)),
    CipherDef("PBEWITHSHAAND192BITAES-CBC-BC", "PBEWITHSHAAND192BITAES-CBC-BC", _pbe12("sha1", "aes", 24, 16)),
    CipherDef("PBEWITHSHAAND2-KEYTRIPLEDES-CBC", "PBEWITHSHAAND2-KEYTRIPLEDES-CBC", _pbe12("sha1", "3des", 16, 8)),
    CipherDef("PBEWITHSHAAND256BITAES-CBC-BC", "PBEWITHSHAAND256BITAES-CBC-BC", _pbe12("sha1", "aes", 32, 16)),
    CipherDef("PBEWITHSHAAND3-KEYTRIPLEDES-CBC", "PBEWITHSHAAND3-KEYTRIPLEDES-CBC", _pbe12("sha1", "3des", 24, 8)),
    _na("PBEWITHSHAAND40BITRC2-CBC", why="40-bit RC2 keys are not supported by 'cryptography' (128-bit only)"),
    CipherDef("PBEWITHSHAAND40BITRC4", "PBEWITHSHAAND40BITRC4", _pbe12("sha1", "rc4", 5, 0, "stream", False)),
    CipherDef("PBEWITHSHAANDIDEA-CBC", "PBEWITHSHAANDIDEA-CBC", _pbe12("sha1", "idea", 16, 8)),
    _na("PBEWITHSHAANDTWOFISH-CBC", why="Twofish is " + _NA),
    _na("1.2.840.113549.3.2", "RC2/CBC/PKCS5Padding",
        "RC2 with the 256-bit derived key is not supported by 'cryptography' (128-bit keys only)"),
    _na("RC5"),
    _na("RC6"),
    CipherDef("RIJNDAEL", "RIJNDAEL", Impl("aes", "ecb", True), "Rijndael-128 = AES"),
    CipherDef("SEED", "SEED", Impl("seed", "ecb", True)),
    _na("SERPENT"),
    _na("SHACAL-2"),
    _na("SKIPJACK"),
    CipherDef("SM4", "SM4", Impl("sm4", "ecb", True)),
    _na("1.3.6.1.4.1.11591.13.2.2", "Serpent-128/CBC", _IV_BUG),
    _na("1.3.6.1.4.1.11591.13.2.4", "Serpent-128/CFB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.1", "Serpent-128/ECB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.3", "Serpent-128/OFB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.22", "Serpent-192/CBC", _IV_BUG),
    _na("1.3.6.1.4.1.11591.13.2.24", "Serpent-192/CFB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.21", "Serpent-192/ECB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.23", "Serpent-192/OFB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.42", "Serpent-256/CBC", _IV_BUG),
    _na("1.3.6.1.4.1.11591.13.2.44", "Serpent-256/CFB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.41", "Serpent-256/ECB", "Serpent is " + _NA),
    _na("1.3.6.1.4.1.11591.13.2.43", "Serpent-256/OFB", "Serpent is " + _NA),
    _na("TEA"),
    _na("THREEFISH-256"),
    _na("TNEPRES"),
    _na("TWOFISH"),
    _na("XTEA"),
]

_BY_CIPHER = {c.cipher: c for c in CIPHERS}
_BY_NAME = {c.name: c for c in CIPHERS}
_BY_KEY: dict[str, CipherDef] = {}
for _c in CIPHERS:  # case-insensitive fallback; the first entry wins ("AES/GCM/NoPadding" before "...NOPADDING")
    _BY_KEY.setdefault(_c.name.upper(), _c)
    _BY_KEY.setdefault(_c.cipher.upper(), _c)
SUPPORTED_NAMES = [c.name for c in CIPHERS if c.impl is not None]


def lookup(algorithm: str) -> CipherDef:
    """Find a table entry by ImageToolbox display name or JCA cipher string (exact, then case-insensitive)."""
    a = algorithm.strip()
    d = _BY_NAME.get(a) or _BY_CIPHER.get(a) or _BY_KEY.get(a.upper())
    if d is None:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Unknown cipher algorithm '{algorithm}'",
                              details={"supported": SUPPORTED_NAMES}, hint="See the cipher_algorithms tool.")
    return d


# ---------------------------------------------------------------------------
# Exact ports of AndroidCryptographyManager's name-based rules
# ---------------------------------------------------------------------------

def is_gcm(name: str) -> bool:
    return "GCM" in name.upper()


def is_authenticated(name: str) -> bool:
    u = name.upper()
    return is_gcm(name) or "CCM" in u or "POLY1305" in u


def key_size_bits(name: str) -> int:
    return 128 if any(m.upper() in name.upper() for m in MD5_LIST) else 256


def legacy_hash(name: str) -> str:
    return "md5" if any(m.upper() in name.upper() for m in MD5_LIST) else "sha256"


def iv_size(name: str) -> int:
    if name in ("AESRFC5649WRAP", "AESWRAPPAD", "AES_128/KWP/NoPadding", "AES_192/KWP/NoPadding",
                "AES_256/KWP/NoPadding", "ARIAWRAPPAD"):
        return 4
    if (name in ("AESWRAP", "AES_128/KW/NoPadding", "AES_192/KW/NoPadding", "CAMELLIAWRAP", "AES_256/KW/NoPadding",
                 "ARIAWRAP", "CHACHA", "DESEDEWRAP", "GRAINV1", "SALSA20")
            or ("CBC" in name and name != "SEED/CBC") or "WRAP" in name.upper()):
        return 8
    if (name in ("AES/ECB/PKCS7PADDING", "AES/GCM-SIV/NOPADDING", "AES128/CCM", "AES192/CCM", "AES256/CCM",
                 "AES_256/GCM-SIV/NOPADDING", "AES_256/GCM/NOPADDING", "GRAIN128", "AES_128/CBC/NOPADDING",
                 "AES_128/CBC/PKCS5PADDING", "AES_128/GCM/NOPADDING") or "CCM" in name or "CHACHA" in name):
        return 12
    if name == "XSALSA20":
        return 24
    if name == "ZUC-256":
        return 25
    if "DSTU7624" in name and "512" in name:
        return 64
    return 16


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

@dataclass
class Header:
    iterations: int
    key_size: int  # bits
    salt: bytes
    nonce: bytes
    cipher: str

    def encode(self) -> bytes:
        cb = self.cipher.encode("utf-8")
        return (MAGIC + struct.pack(">BBiHBBH", VERSION, KDF_PBKDF2_SHA256, self.iterations, self.key_size,
                                    len(self.salt), len(self.nonce), len(cb)) + self.salt + self.nonce + cb)


def has_magic(data: bytes) -> bool:
    return data[:len(MAGIC)] == MAGIC


def decode_header(data: bytes) -> tuple[Header, int]:
    """``EncryptedFileHeader.decode`` with the same validation rules."""
    def bad(why: str) -> LazyRoomError:
        return LazyRoomError(ErrorCode.DECODE_FAILED, DECRYPT_FAILED, details={"reason": f"invalid header: {why}"})

    if len(data) < FIXED_HEADER_SIZE or not has_magic(data):
        raise bad("too short or missing magic")
    version, kdf, iterations, key_size, salt_n, nonce_n, cipher_n = struct.unpack_from(">BBiHBBH", data, 8)
    if version != VERSION:
        raise bad(f"unsupported version {version}")
    if kdf != KDF_PBKDF2_SHA256:
        raise bad(f"unsupported key derivation {kdf}")
    size = FIXED_HEADER_SIZE + salt_n + nonce_n + cipher_n
    if not 1 <= iterations <= PBKDF2_ITERATIONS:
        raise bad("iteration count out of range")
    if key_size not in (128, 256):
        raise bad("key size must be 128 or 256")
    if salt_n != SALT_SIZE or nonce_n > MAX_NONCE_SIZE or not 1 <= cipher_n <= MAX_CIPHER_NAME_SIZE:
        raise bad("field size out of range")
    if size >= len(data):
        raise bad("no ciphertext")
    p = FIXED_HEADER_SIZE
    salt, nonce = data[p:p + salt_n], data[p + salt_n:p + salt_n + nonce_n]
    raw = data[p + salt_n + nonce_n:size]
    try:
        cipher = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise bad("cipher name is not UTF-8") from None
    if not all(0x20 <= ord(ch) <= 0x7E for ch in cipher):
        raise bad("cipher name is not printable ASCII")
    if is_gcm(cipher) and nonce_n != GCM_NONCE_SIZE:
        raise bad("GCM nonce must be 12 bytes")
    return Header(iterations, key_size, salt, nonce, cipher), size


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def password_bytes(password: str) -> bytes:
    """UTF-8 bytes of a password; the error never quotes the offending character."""
    try:
        return password.encode("utf-8")
    except UnicodeEncodeError:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "The password contains characters that are not valid "
                              "Unicode (lone surrogates)", details={"param": "password"}) from None


def pbkdf2(password: str, salt: bytes, iterations: int, key_bits: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password_bytes(password), salt, iterations, key_bits // 8)


def _h(digest: str) -> Any:
    return getattr(hashlib, digest)


def pkcs12_kdf(digest: str, password: bytes, salt: bytes, iterations: int, ident: int, n: int) -> bytes:
    """RFC 7292 appendix B.2 (BouncyCastle ``PKCS12ParametersGenerator``)."""
    h = _h(digest)
    u, v = h().digest_size, h().block_size
    D = bytes([ident]) * v

    def fill(b: bytes) -> bytes:
        if not b:
            return b""
        k = v * ((len(b) + v - 1) // v)
        return (b * (k // len(b) + 1))[:k]

    buf = bytearray(fill(salt) + fill(password))
    out = b""
    while len(out) < n:
        A = h(D + bytes(buf)).digest()
        for _ in range(iterations - 1):
            A = h(A).digest()
        out += A
        if len(out) >= n:
            break
        B = int.from_bytes((A * (v // u + 1))[:v], "big")
        for j in range(0, len(buf), v):
            chunk = (int.from_bytes(buf[j:j + v], "big") + B + 1) % (1 << (8 * v))
            buf[j:j + v] = chunk.to_bytes(v, "big")
    return out[:n]


def _pbe_password_bytes(scheme: str, password: str) -> bytes:
    if scheme == "pkcs12":  # PKCS12PasswordToBytes: UTF-16BE + two zero bytes
        return (password.encode("utf-16-be") + b"\x00\x00") if password else b""
    return bytes(ord(ch) & 0xFF for ch in password)  # PKCS5PasswordToBytes (8-bit)


def pbe_key_iv(pbe: tuple[str, str, int, int], password: str, salt: bytes, iterations: int) -> tuple[bytes, bytes]:
    scheme, digest, kn, ivn = pbe
    pw = _pbe_password_bytes(scheme, password)
    if scheme == "pkcs12":
        return (pkcs12_kdf(digest, pw, salt, iterations, 1, kn),
                pkcs12_kdf(digest, pw, salt, iterations, 2, ivn) if ivn else b"")
    if scheme == "pkcs5s1":  # PBKDF1: H^c(P || S); key and IV split one digest
        h = _h(digest)
        dk = h(pw + salt).digest()
        for _ in range(iterations - 1):
            dk = h(dk).digest()
        return dk[:kn], dk[kn:kn + ivn]
    if scheme == "openssl":  # EVP_BytesToKey with count 1 (BouncyCastle ignores the iteration count)
        h = _h(digest)
        out, prev = b"", b""
        while len(out) < kn + ivn:
            prev = h(prev + pw + salt).digest()
            out += prev
        return out[:kn], out[kn:kn + ivn]
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# cryptography backend
# ---------------------------------------------------------------------------

def _crypto() -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # noqa: F401
    except ImportError:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                              "File encryption needs the optional 'cryptography' package",
                              hint="pip install cryptography") from None
    import cryptography.hazmat.primitives.ciphers as c

    return c


def _algorithm(alg: str, key: bytes) -> Any:
    c = _crypto()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from cryptography.hazmat.decrepit.ciphers import algorithms as old
        except ImportError:  # cryptography < 43
            old = c.algorithms
        if alg == "aes":
            return c.algorithms.AES(key)
        if alg == "camellia":
            return c.algorithms.Camellia(key)
        if alg == "sm4":
            return c.algorithms.SM4(key)
        if alg == "3des":
            return old.TripleDES(key)
        if alg == "des":  # single DES = Triple DES with K1 = K2 = K3
            return old.TripleDES(key * 3)
        if alg == "blowfish":
            return old.Blowfish(key)
        if alg == "idea":
            return old.IDEA(key)
        if alg == "seed":
            return old.SEED(key)
        if alg == "rc2":
            return old.RC2(key)
        if alg == "rc4":
            return old.ARC4(key)
    raise ValueError(alg)


def _pkcs7_pad(data: bytes, bs: int) -> bytes:
    n = bs - len(data) % bs
    return data + bytes([n]) * n


def _pkcs7_unpad(data: bytes, bs: int) -> bytes:
    if not data or len(data) % bs:
        raise ValueError("bad length")
    n = data[-1]
    if not 1 <= n <= bs or not hmac.compare_digest(data[-n:], bytes([n]) * n):
        raise ValueError("bad padding")
    return data[:-n]


class _Failed(Exception):
    pass


def run_cipher(impl: Impl, key: bytes, iv: bytes, data: bytes, *, encrypt: bool, aad: bytes = b"") -> bytes:
    c = _crypto()
    from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm

    if impl.key_bytes:
        key = key[:impl.key_bytes]
    try:
        alg = _algorithm(impl.alg, key)
        if impl.mode == "gcm":
            if encrypt:
                e = c.Cipher(alg, c.modes.GCM(iv)).encryptor()
                e.authenticate_additional_data(aad)
                return e.update(data) + e.finalize() + e.tag
            if len(data) < GCM_TAG_SIZE:
                raise _Failed("ciphertext shorter than the tag")
            d = c.Cipher(alg, c.modes.GCM(iv, data[-GCM_TAG_SIZE:])).decryptor()
            d.authenticate_additional_data(aad)
            return d.update(data[:-GCM_TAG_SIZE]) + d.finalize()
        mode = {"ecb": lambda: c.modes.ECB(), "cbc": lambda: c.modes.CBC(iv), "ctr": lambda: c.modes.CTR(iv),
                "stream": lambda: None}[impl.mode]()
        cipher = c.Cipher(alg, mode)
        bs = alg.block_size // 8 if impl.mode != "stream" else 1
        if encrypt:
            payload = _pkcs7_pad(data, bs) if impl.pad else data
            if impl.mode in ("ecb", "cbc") and len(payload) % bs:
                raise LazyRoomError(ErrorCode.INVALID_PARAMS,
                                      f"This unpadded block cipher needs a data length that is a multiple of {bs} bytes")
            e = cipher.encryptor()
            return e.update(payload) + e.finalize()
        if impl.mode in ("ecb", "cbc") and len(data) % bs:
            raise _Failed("ciphertext length is not a multiple of the block size")
        d = cipher.decryptor()
        out = d.update(data) + d.finalize()
        return _pkcs7_unpad(out, bs) if impl.pad else out
    except InvalidTag:
        raise _Failed("authentication tag mismatch") from None
    except UnsupportedAlgorithm as exc:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                              f"The installed cryptography/OpenSSL build does not provide {impl.alg.upper()}",
                              details={"reason": str(exc)},
                              hint="Legacy ciphers need OpenSSL's legacy provider; pick another algorithm.") from None
    except ValueError as exc:
        raise _Failed(str(exc)) from None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_random_string(length: int) -> str:
    """``generateRandomString``: SecureRandom picks from [0-9A-Za-z]."""
    return "".join(secrets.choice(RANDOM_CHARS) for _ in range(length))


def _supported(d: CipherDef) -> Impl:
    if d.impl is None:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"Cipher '{d.name}' is not supported: {d.note}",
                              details={"cipher": d.cipher}, hint="See the cipher_algorithms tool.")
    return d.impl


def _versioned_cipher(d: CipherDef, impl: Impl, password: str, header: Header) -> tuple[bytes, bytes]:
    key = pbkdf2(password, header.salt, header.iterations, header.key_size)
    if impl.pbe:
        return pbe_key_iv(impl.pbe, key.hex(), header.salt, header.iterations)
    # IvParameterSpec(nonce) is accepted for CBC/CTR (16 or 8 byte nonce = block size); ECB/stream ignore it
    return key, header.nonce


def encrypt(data: bytes, password: str, algorithm: str = DEFAULT_ALGORITHM, *,
            iterations: int = PBKDF2_ITERATIONS) -> bytes:
    d = lookup(algorithm)
    impl = _supported(d)
    nonce = secrets.token_bytes(GCM_NONCE_SIZE if is_gcm(d.name) else iv_size(d.name))
    header = Header(iterations, key_size_bits(d.name), secrets.token_bytes(SALT_SIZE), nonce, d.cipher)
    enc_header = header.encode()
    key, iv = _versioned_cipher(d, impl, password, header)
    aad = enc_header if is_authenticated(d.name) else b""
    try:
        return enc_header + run_cipher(impl, key, iv, data, encrypt=True, aad=aad)
    except _Failed as exc:
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Encryption failed: {exc}") from None


def decrypt(data: bytes, password: str, legacy_algorithm: str = DEFAULT_ALGORITHM) -> tuple[bytes, dict[str, Any]]:
    """Decrypt an ImageToolbox file.  Returns (plaintext, info)."""
    if has_magic(data):
        header, size = decode_header(data)
        d = _BY_CIPHER.get(header.cipher) or _BY_KEY.get(header.cipher.upper())
        if d is None:
            raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                                  f"The file was encrypted with '{header.cipher}', which LazyRoom does not know",
                                  details={"cipher": header.cipher}, hint="See the cipher_algorithms tool.")
        impl = _supported(d)
        key, iv = _versioned_cipher(d, impl, password, header)
        aad = data[:size] if is_authenticated(d.name) else b""
        try:
            out = run_cipher(impl, key, iv, data[size:], encrypt=False, aad=aad)
        except _Failed as exc:
            raise LazyRoomError(ErrorCode.DECODE_FAILED, DECRYPT_FAILED, details={"reason": str(exc)}) from None
        return out, {"format": "versioned", "algorithm": d.name, "cipher": d.cipher, "iterations": header.iterations,
                     "key_bits": header.key_size, "authenticated": is_authenticated(d.name),
                     "padding_checked": impl.pad}
    d = lookup(legacy_algorithm)
    impl = _supported(d)
    if impl.pbe:  # createKey(PBEKeySpec(password)) + PBEParameterSpec(encoded, encoded.size)
        enc = _pbe_password_bytes(impl.pbe[0], password)
        key, iv = pbe_key_iv(impl.pbe, password, enc, len(enc))
    else:
        key = _h(legacy_hash(d.name))(password_bytes(password)).digest()
        bs = 0 if impl.mode == "stream" else (8 if impl.alg in ("3des", "des", "blowfish", "idea", "rc2") else 16)
        iv = key[:bs]  # IvParameterSpec(encodedKey, 0, blockSize); ECB / RC4 ignore it
    try:
        out = run_cipher(impl, key, iv, data, encrypt=False)
    except _Failed as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, DECRYPT_FAILED, details={"reason": str(exc)}) from None
    return out, {"format": "legacy", "algorithm": d.name, "cipher": d.cipher,
                 "authenticated": impl.mode == "gcm", "padding_checked": impl.pad}


def describe_algorithms(probe: bool = True) -> list[dict[str, Any]]:
    out = []
    for d in CIPHERS:
        e: dict[str, Any] = {"name": d.name, "cipher": d.cipher, "supported": d.impl is not None,
                             "provider": d.provider}
        if d.impl is not None:
            i = d.impl
            e.update({"algorithm": i.alg, "mode": i.mode, "padding": "PKCS7" if i.pad else "none",
                      "authenticated": is_authenticated(d.name), "key_bits": key_size_bits(d.name),
                      "nonce_bytes": GCM_NONCE_SIZE if is_gcm(d.name) else iv_size(d.name)})
            if i.pbe:
                e["pbe"] = {"scheme": i.pbe[0], "digest": i.pbe[1], "key_bytes": i.pbe[2], "iv_bytes": i.pbe[3]}
            if probe:
                e["available"] = _probe(i)
        if d.note:
            e["note" if d.impl is not None else "reason"] = d.note
        out.append(e)
    return out


def _probe(impl: Impl) -> bool:
    try:
        key = bytes(range(32))[:impl.pbe[2]] if impl.pbe else bytes(range(32))
        if impl.alg in ("3des",):
            key = key[:24]
        elif impl.alg in ("seed", "sm4", "idea", "rc2"):
            key = key[:16]
        elif impl.alg == "des":
            key = key[:8]
        iv = bytes(GCM_NONCE_SIZE) if impl.mode == "gcm" else bytes(16 if impl.alg in ("aes", "camellia", "sm4", "seed") else 8)
        run_cipher(impl, key, iv, b"probe", encrypt=True)
        return True
    except LazyRoomError as e:
        if e.code == ErrorCode.BACKEND_UNAVAILABLE and "optional 'cryptography'" in e.message:
            raise
        return False
    except Exception:
        return False
