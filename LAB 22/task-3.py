import socket
from concurrent.futures import ThreadPoolExecutor
def scan_port(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex((ip, port))
        if result == 0:
            print(f"Port {port} is open on {ip}")
        else:
            print(f"Port {port} is closed on {ip}")

def scan_ports(ip, ports):
    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(lambda port: scan_port(ip, port), ports)

if __name__ == "__main__":
    target_ip = input("Enter the IP address to scan: ")
    port_range = range(1, 1025)  # Scanning ports 1 to 1024
    scan_ports(target_ip, port_range)