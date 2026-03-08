import Foundation
import Combine

enum StreamingConnectionState: String {
    case disconnected
    case connecting
    case connected
    case error
}

/// Server response types parsed from WebSocket JSON messages.
struct ServerResult: Identifiable {
    let id = UUID()
    let taskType: String
    let timestampMs: Int64
    let output: [String: Any]
    let confidence: Double
    let narrative: String
}

struct DeepAnalysisResult: Identifiable {
    let id = UUID()
    let taskType: String
    let output: [String: Any]
    let clinicalNarrative: String
    let raw: [String: Any]
}

@MainActor
final class StreamingService: NSObject, ObservableObject {

    @Published var connectionState: StreamingConnectionState = .disconnected
    @Published var latestResult: ServerResult?
    @Published var deepAnalysisResult: DeepAnalysisResult?
    @Published var deepAnalysisInProgress = false
    @Published var lastError: String?
    @Published var serverSessionId: String?

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private var pingTimer: Timer?

    var baseURL: String = "ws://localhost:8000"

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }

    // MARK: - Connection

    func connect(sessionId: String) {
        guard connectionState != .connected, connectionState != .connecting else { return }
        connectionState = .connecting
        lastError = nil

        let urlString = "\(baseURL)/ws/session/\(sessionId)"
        guard let url = URL(string: urlString) else {
            connectionState = .error
            lastError = "Invalid URL: \(urlString)"
            return
        }

        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()
        connectionState = .connected
        startReceiving()
        startPing()
    }

    func disconnect() {
        stopPing()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        connectionState = .disconnected
        serverSessionId = nil
    }

    // MARK: - Sending

    func sendSessionStart(taskType: String, sessionId: String) {
        let message = SessionStartMessage(taskType: taskType, sessionId: sessionId)
        sendCodable(message)
    }

    func sendSensorData(emg: [Double], imu: IMUData?) {
        let message = SensorDataMessage(emg: emg, imu: imu)
        sendCodable(message)
    }

    func sendDeepAnalysisRequest(taskType: String, question: String?) {
        let message = DeepAnalysisRequestMessage(taskType: taskType, question: question)
        sendCodable(message)
        deepAnalysisInProgress = true
    }

    func sendSessionEnd() {
        let message = SessionEndMessage()
        sendCodable(message)
    }

    // MARK: - Private

    private func sendCodable<T: Codable>(_ value: T) {
        guard let data = try? JSONEncoder().encode(value),
              let jsonString = String(data: data, encoding: .utf8) else { return }
        webSocketTask?.send(.string(jsonString)) { [weak self] error in
            if let error {
                Task { @MainActor [weak self] in
                    self?.lastError = error.localizedDescription
                }
            }
        }
    }

    private func startReceiving() {
        webSocketTask?.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self else { return }
                switch result {
                case .success(let message):
                    self.handleMessage(message)
                    self.startReceiving()
                case .failure(let error):
                    self.connectionState = .error
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .string(let text):
            guard let d = text.data(using: .utf8) else { return }
            data = d
        case .data(let d):
            data = d
        @unknown default:
            return
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        switch type {
        case "session_started":
            serverSessionId = json["session_id"] as? String

        case "result":
            let result = ServerResult(
                taskType: json["task_type"] as? String ?? "",
                timestampMs: json["timestamp_ms"] as? Int64 ?? 0,
                output: json["output"] as? [String: Any] ?? [:],
                confidence: json["confidence"] as? Double ?? 0,
                narrative: json["narrative"] as? String ?? ""
            )
            latestResult = result

        case "deep_analysis_started":
            deepAnalysisInProgress = true

        case "deep_analysis":
            let result = DeepAnalysisResult(
                taskType: json["task_type"] as? String ?? "",
                output: json["output"] as? [String: Any] ?? [:],
                clinicalNarrative: json["clinical_narrative"] as? String ?? "",
                raw: json
            )
            deepAnalysisResult = result
            deepAnalysisInProgress = false

        case "session_ended":
            serverSessionId = nil

        case "error":
            lastError = json["message"] as? String

        default:
            break
        }
    }

    private func startPing() {
        pingTimer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in
            self?.webSocketTask?.sendPing { error in
                if let error {
                    Task { @MainActor [weak self] in
                        self?.connectionState = .error
                        self?.lastError = "Ping failed: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    private func stopPing() {
        pingTimer?.invalidate()
        pingTimer = nil
    }
}

// MARK: - URLSessionWebSocketDelegate

extension StreamingService: URLSessionWebSocketDelegate {

    nonisolated func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        Task { @MainActor [weak self] in
            self?.connectionState = .connected
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        Task { @MainActor [weak self] in
            self?.connectionState = .disconnected
        }
    }
}
