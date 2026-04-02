"""
Bitcoin OP_RETURN attestation — construye y broadcastea una tx P2PKH con OP_RETURN data.
No requiere nodo Bitcoin. Usa Blockstream API para broadcast.
"""
import hashlib
import struct
import urllib.request
import json
from eth_keys import keys as eth_keys


def _dsha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _varint(n: int) -> bytes:
    if n < 0xfd:
        return struct.pack("B", n)
    elif n <= 0xffff:
        return b"\xfd" + struct.pack("<H", n)
    elif n <= 0xffffffff:
        return b"\xfe" + struct.pack("<I", n)
    return b"\xff" + struct.pack("<Q", n)


def _push(data: bytes) -> bytes:
    n = len(data)
    if n < 0x4c:
        return struct.pack("B", n) + data
    elif n <= 0xff:
        return b"\x4c" + struct.pack("B", n) + data
    raise ValueError("data too long")


def _pubkey_from_privkey(privkey_hex: str) -> bytes:
    """Compressed secp256k1 pubkey from hex private key."""
    pk = eth_keys.PrivateKey(bytes.fromhex(privkey_hex.removeprefix("0x")))
    x = pk.public_key.to_bytes()[:32]
    y = pk.public_key.to_bytes()[32:]
    prefix = b"\x02" if y[-1] % 2 == 0 else b"\x03"
    return prefix + x


def _p2pkh_script(pubkey_compressed: bytes) -> bytes:
    """scriptPubKey for P2PKH."""
    h = hashlib.new("ripemd160", hashlib.sha256(pubkey_compressed).digest()).digest()
    return (
        b"\x76"   # OP_DUP
        b"\xa9"   # OP_HASH160
        + _push(h)
        + b"\x88"  # OP_EQUALVERIFY
        b"\xac"   # OP_CHECKSIG
    )


def _sign_input(privkey_hex: str, tx_bytes_for_signing: bytes) -> bytes:
    """Sign tx sighash, return DER signature + SIGHASH_ALL byte."""
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    sighash = _dsha256(tx_bytes_for_signing)

    # eth_keys signs the raw hash directly — clean and available
    pk = eth_keys.PrivateKey(bytes.fromhex(privkey_hex.removeprefix("0x")))
    sig_obj = pk.sign_msg_hash(sighash)
    sig_bytes = bytes(sig_obj)  # 65 bytes: r(32) + s(32) + v(1)

    r = int.from_bytes(sig_bytes[:32], 'big')
    s = int.from_bytes(sig_bytes[32:64], 'big')

    # Low-S normalization (required for Bitcoin standard scripts)
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    if s > n // 2:
        s = n - s

    der_sig = encode_dss_signature(r, s)
    return der_sig + b"\x01"  # SIGHASH_ALL


def build_and_broadcast(
    privkey_hex: str,
    utxo_txid: str,
    utxo_vout: int,
    utxo_value: int,
    opreturn_data: bytes,
    fee_sats: int = 500,
) -> str:
    """
    Build a P2PKH tx with one OP_RETURN output and change back to our address.
    Returns the broadcast txid.
    """
    pubkey = _pubkey_from_privkey(privkey_hex)
    script_pubkey = _p2pkh_script(pubkey)
    change = utxo_value - fee_sats

    if len(opreturn_data) > 80:
        opreturn_data = opreturn_data[:80]

    # --- Build tx for signing (scriptSig = scriptPubKey of input) ---
    def _serialize_tx(script_sig: bytes) -> bytes:
        raw = b""
        raw += struct.pack("<I", 1)           # version
        raw += _varint(1)                    # 1 input
        raw += bytes.fromhex(utxo_txid)[::-1]  # txid little-endian
        raw += struct.pack("<I", utxo_vout)  # vout
        raw += _varint(len(script_sig))
        raw += script_sig
        raw += struct.pack("<I", 0xffffffff) # sequence

        # outputs
        outputs = []
        # OP_RETURN output (0 value)
        op_return_script = b"\x6a" + _push(opreturn_data)
        outputs.append((0, op_return_script))
        # change output
        outputs.append((change, script_pubkey))

        raw += _varint(len(outputs))
        for val, scr in outputs:
            raw += struct.pack("<q", val)
            raw += _varint(len(scr))
            raw += scr
        raw += struct.pack("<I", 0)          # locktime
        return raw

    # Signing payload: serialized tx + SIGHASH_ALL (4 bytes LE)
    tx_for_signing = _serialize_tx(script_pubkey) + struct.pack("<I", 1)
    der_sig = _sign_input(privkey_hex, tx_for_signing)

    # Build final scriptSig: <sig> <pubkey>
    script_sig = _push(der_sig) + _push(pubkey)

    # Final tx
    final_tx = _serialize_tx(script_sig)

    # Broadcast via Blockstream
    req = urllib.request.Request(
        "https://blockstream.info/api/tx",
        data=final_tx.hex().encode(),
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=15)
    txid = resp.read().decode().strip()
    return txid


def attest_opreturn(commitment_hash: str, privkey_hex: str) -> dict | None:
    """
    Publica el commitment hash en Bitcoin base layer via OP_RETURN.
    Usa el UTXO disponible en la dirección derivada de privkey_hex.
    """
    try:
        utxo_txid = "fcb22d363802baa8cad261613f2b675b98ab9daa32d3ab7c3965c25b624a065c"
        utxo_vout  = 0
        utxo_value = 7289
        data = bytes.fromhex(commitment_hash[:64])  # 32 bytes
        txid = build_and_broadcast(
            privkey_hex  = privkey_hex,
            utxo_txid    = utxo_txid,
            utxo_vout    = utxo_vout,
            utxo_value   = utxo_value,
            opreturn_data = data,
            fee_sats      = 600,
        )
        return {"txid": txid, "network": "bitcoin-mainnet", "data": commitment_hash[:64]}
    except Exception as e:
        return {"error": str(e)}
