import Foundation
import CoreBluetooth
import Combine

struct DiscoveredPeripheral: Identifiable, Hashable {
    let id: UUID
    let name: String
    let peripheral: CBPeripheral
    let rssi: Int

    func hash(into hasher: inout Hasher) { hasher.combine(id) }
    static func == (lhs: DiscoveredPeripheral, rhs: DiscoveredPeripheral) -> Bool { lhs.id == rhs.id }
}

enum BLEConnectionState: String {
    case disconnected
    case scanning
    case connecting
    case connected
    case disconnecting
}

@MainActor
final class BLEManager: NSObject, ObservableObject {

    static let emgServiceUUID = CBUUID(string: "FFE0")
    static let emgCharacteristicUUID = CBUUID(string: "FFE1")

    @Published var connectionState: BLEConnectionState = .disconnected
    @Published var discoveredPeripherals: [DiscoveredPeripheral] = []
    @Published var connectedPeripheralName: String?

    private var emgBuffer: [Double] = []
    private let bufferLock = NSLock()

    private var centralManager: CBCentralManager!
    private var connectedPeripheral: CBPeripheral?
    private var emgCharacteristic: CBCharacteristic?

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: .global(qos: .userInitiated))
    }

    func startScanning() {
        guard centralManager.state == .poweredOn else { return }
        discoveredPeripherals.removeAll()
        connectionState = .scanning
        centralManager.scanForPeripherals(
            withServices: [Self.emgServiceUUID],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
    }

    func stopScanning() {
        centralManager.stopScan()
        if connectionState == .scanning { connectionState = .disconnected }
    }

    func connect(_ peripheral: CBPeripheral) {
        stopScanning()
        connectionState = .connecting
        connectedPeripheral = peripheral
        peripheral.delegate = self
        centralManager.connect(peripheral, options: nil)
    }

    func disconnect() {
        guard let peripheral = connectedPeripheral else { return }
        connectionState = .disconnecting
        if let char = emgCharacteristic {
            peripheral.setNotifyValue(false, for: char)
        }
        centralManager.cancelPeripheralConnection(peripheral)
    }

    /// Drains the accumulated EMG sample buffer (thread-safe).
    func drainEMGBuffer() -> [Double] {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        let samples = emgBuffer
        emgBuffer.removeAll(keepingCapacity: true)
        return samples
    }

    /// Parses raw BLE bytes into EMG sample doubles.
    /// Assumes little-endian 16-bit signed integers, scaled to millivolts.
    private func parseEMGData(_ data: Data) -> [Double] {
        var samples: [Double] = []
        let bytesPerSample = 2
        let count = data.count / bytesPerSample
        for i in 0..<count {
            let offset = i * bytesPerSample
            let raw = data.subdata(in: offset..<offset + bytesPerSample)
                .withUnsafeBytes { $0.load(as: Int16.self) }
            let value = Double(Int16(littleEndian: raw))
            // Scale ADC counts → millivolts (3.3V ref, 12-bit ADC, gain 1000)
            let millivolts = (value / 2048.0) * (3300.0 / 1000.0)
            samples.append(millivolts)
        }
        return samples
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEManager: CBCentralManagerDelegate {

    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in
            if central.state != .poweredOn {
                connectionState = .disconnected
            }
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager,
        didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any],
        rssi RSSI: NSNumber
    ) {
        let name = peripheral.name ?? "Unknown EMG Device"
        let discovered = DiscoveredPeripheral(
            id: peripheral.identifier,
            name: name,
            peripheral: peripheral,
            rssi: RSSI.intValue
        )
        Task { @MainActor in
            if !discoveredPeripherals.contains(where: { $0.id == discovered.id }) {
                discoveredPeripherals.append(discovered)
            }
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            connectionState = .connected
            connectedPeripheralName = peripheral.name ?? "EMG Sensor"
        }
        peripheral.discoverServices([BLEManager.emgServiceUUID])
    }

    nonisolated func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        Task { @MainActor in
            connectionState = .disconnected
            connectedPeripheral = nil
            connectedPeripheralName = nil
            emgCharacteristic = nil
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        Task { @MainActor in
            connectionState = .disconnected
            connectedPeripheral = nil
        }
    }
}

// MARK: - CBPeripheralDelegate

extension BLEManager: CBPeripheralDelegate {

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services where service.uuid == BLEManager.emgServiceUUID {
            peripheral.discoverCharacteristics([BLEManager.emgCharacteristicUUID], for: service)
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        for char in characteristics where char.uuid == BLEManager.emgCharacteristicUUID {
            Task { @MainActor in
                emgCharacteristic = char
            }
            peripheral.setNotifyValue(true, for: char)
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard characteristic.uuid == BLEManager.emgCharacteristicUUID,
              let data = characteristic.value else { return }
        let samples = parseEMGData(data)
        bufferLock.lock()
        emgBuffer.append(contentsOf: samples)
        bufferLock.unlock()
    }
}
