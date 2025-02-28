# 🚀 Network Emulation: Client-Server with Wireless & NetEm

This guide walks you through setting up a **client-server network** using **Linux network namespaces**, **virtual Ethernet (veth) pairs**, **wireless interfaces (`wlp3s0:1` & `wlp3s0:2`)**, and **network emulation using `tc netem`**.

---

## 🔹 Step 1: Create Network Namespaces
We create two separate namespaces:
- `client` → Represents the client
- `server` → Represents the server

### 🛠 Commands:
```bash
sudo ip netns add client
sudo ip netns add server
```
Verify:
```bash
ip netns list
```

---

## 🔹 Step 2: Create Virtual Network Interfaces
We create a **virtual Ethernet (veth) pair** to link the namespaces.

### 🛠 Commands:
```bash
sudo ip link add veth-client type veth peer name veth-client
sudo ip link add veth-server type veth peer name veth-server

```

Assign each interface to a namespace:
```bash
sudo ip link set veth-client netns client
sudo ip link set veth-server netns server
```

---

## 🔹 Step 3: Assign IP Addresses
### 🛠 Commands:
```bash
sudo ip netns exec client ip addr add 192.168.1.10/24 dev veth-client
sudo ip netns exec server ip addr add 192.168.1.20/24 dev veth-server
```

Bring up interfaces:
```bash
sudo ip netns exec client ip link set veth-client up
sudo ip netns exec server ip link set veth-server up
```

Enable loopback interfaces:
```bash
sudo ip netns exec client ip link set lo up
sudo ip netns exec server ip link set lo up
```

Enable IP forwarding:
```bash
sudo sysctl -w net.ipv4.ip_forward=1
```

---

## 🔹 Step 4: Verify Connectivity
### 🛠 Commands:
```bash
sudo ip netns exec client ping -c 4 192.168.1.20
sudo ip netns exec server ping -c 4 192.168.1.10
```
If you receive **64 bytes from ...**, the setup is correct.

---

## 🔹 Step 5: Apply Network Emulation (`tc netem`)
We add **latency, jitter, and packet loss**.

### 🛠 Commands:
```bash
sudo ip netns exec client tc qdisc add dev veth-client root netem delay 10ms 5ms loss 1%
sudo ip netns exec server tc qdisc add dev veth-server root netem delay 10ms 5ms loss 1%
```

Verify:
```bash
sudo ip netns exec client tc qdisc show dev veth-client
sudo ip netns exec server tc qdisc show dev veth-server
```

---

## 🔹 Step 6: Attach Wireless Interfaces
We assign `wlp3s0:1` and `wlp3s0:2` to namespaces.

### 🛠 Commands:
```bash
sudo ip link set wlp3s0:1 netns client
sudo ip link set wlp3s0:2 netns server
```

Bring them up:
```bash
sudo ip netns exec client ip link set wlp3s0:1 up
sudo ip netns exec server ip link set wlp3s0:2 up
```

Verify:
```bash
sudo ip netns exec client ip addr show wlp3s0:1
sudo ip netns exec server ip addr show wlp3s0:2
```

---

## 🔹 Step 7: Generate Traffic with `iperf3`
Start an **iperf3 server** in the `server` namespace:
```bash
sudo ip netns exec server iperf3 -s -B 192.168.1.20
```
Run a **client test** from `client`:
```bash
sudo ip netns exec client iperf3 -c 192.168.1.20 -B 192.168.1.10 -t 30
```

---

## 🔹 Step 8: Debugging & Cleanup
### 🛠 Verify `iperf3` Results:
```bash
sudo ip netns exec client tc -s qdisc show dev veth-client
sudo ip netns exec server tc -s qdisc show dev veth-server
```

### 🛠 Remove `tc netem` Rules:
```bash
sudo ip netns exec client tc qdisc del dev veth-client root
sudo ip netns exec server tc qdisc del dev veth-server root
```

---

## 🔹 Summary
✅ **Created namespaces (`client`, `server`)**  
✅ **Setup veth pair & assigned IPs**  
✅ **Tested connectivity with `ping`**  
✅ **Applied network degradation (`tc netem`)**  
✅ **Attached wireless interfaces (`wlp3s0:1`, `wlp3s0:2`)**  
✅ **Generated traffic using `iperf3`**  

---

## 🔥 Next Steps
- **Experiment with more `tc netem` parameters**
- **Monitor traffic over time**
- **Automate this setup with a script**

🚀 Enjoy your custom network emulation! 🎯


